"""
Advanced RAG pipeline built with LangGraph.

Techniques implemented
----------------------
1. HyDE  – Hypothetical Document Embeddings: generate a fake "perfect answer"
           and use *its* embedding for retrieval alongside the raw query.
2. Multi-Query – expand the original question into N variations to increase
                 recall across semantically different phrasings.
3. Reciprocal Rank Fusion (RRF) – merge ranked lists from every query variant
                                   without needing score normalisation.
4. Cross-Encoder Re-ranking – score (query, passage) pairs with a fine-tuned
                               cross-encoder and re-order the fused list.
5. Contextual Compression – ask the LLM to distil each retrieved passage down
                             to only the sentences that are relevant to the
                             question before building the final prompt.

Pipeline graph (LangGraph)
--------------------------
START → expand_query → retrieve → rerank → compress → generate → END
"""

import json
import operator
import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sentence_transformers import CrossEncoder

from .vector_engine import MultiModalVectorEngine


# ──────────────────────────────────────────────────────────────────────────────
#  State schema
# ──────────────────────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    query: str
    hyde_document: str
    expanded_queries: List[str]
    text_results: List[Dict[str, Any]]
    image_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    compressed_context: str
    answer: str
    sources: List[Dict[str, Any]]
    # pipeline_trace accumulates strings from every node
    pipeline_trace: Annotated[List[str], operator.add]


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline class
# ──────────────────────────────────────────────────────────────────────────────


class AdvancedRAGPipeline:
    def __init__(
        self,
        vector_engine: MultiModalVectorEngine,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen2:0.5b",
    ):
        self.vector_engine = vector_engine
        self.ollama_host = ollama_host
        self.model_name = model

        print(f"  Connecting to Ollama model: {model}")
        self.llm = ChatOllama(
            model=model, base_url=ollama_host, temperature=0.7
        )

        print("  Loading cross-encoder reranker (ms-marco-MiniLM-L-6-v2) …")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
        )

        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    #  Graph construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        builder = StateGraph(RAGState)

        builder.add_node("expand_query", self._expand_query)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("rerank", self._rerank)
        builder.add_node("compress", self._compress)
        builder.add_node("generate", self._generate)

        builder.add_edge(START, "expand_query")
        builder.add_edge("expand_query", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "compress")
        builder.add_edge("compress", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    # ------------------------------------------------------------------ #
    #  Nodes                                                               #
    # ------------------------------------------------------------------ #

    # 1. Query expansion: HyDE + multi-query
    def _expand_query(self, state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # --- HyDE --------------------------------------------------------
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a short, factual paragraph that directly answers the question "
            "below. Output ONLY the paragraph.\n\nQuestion: {query}"
        )
        try:
            hyde_doc: str = (hyde_prompt | self.llm | StrOutputParser()).invoke(
                {"query": query}
            )
        except Exception:
            hyde_doc = query

        # --- Multi-query -------------------------------------------------
        mq_prompt = ChatPromptTemplate.from_template(
            "Generate 3 distinct search-query reformulations of the question below "
            "to maximise document recall.\n"
            "Return ONLY a JSON array of 3 strings, e.g.: "
            '["q1", "q2", "q3"]\n\nQuestion: {query}'
        )
        expanded = [query]
        try:
            raw = (mq_prompt | self.llm | StrOutputParser()).invoke({"query": query})
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                variants = json.loads(m.group())
                expanded += [v for v in variants if isinstance(v, str) and v != query]
        except Exception:
            pass

        return {
            "hyde_document": hyde_doc,
            "expanded_queries": expanded,
            "pipeline_trace": [
                f"HyDE doc generated; {len(expanded)} query variants produced"
            ],
        }

    # 2. Retrieval with RRF fusion
    def _retrieve(self, state: RAGState) -> Dict[str, Any]:
        hyde_doc = state.get("hyde_document", "") or state["query"]
        all_queries = list(dict.fromkeys(state["expanded_queries"] + [hyde_doc]))

        K = 60  # RRF constant
        text_scores: Dict[str, float] = {}
        text_docs: Dict[str, Dict] = {}
        image_scores: Dict[str, float] = {}
        image_docs: Dict[str, Dict] = {}

        for q in all_queries:
            for rank, doc in enumerate(self.vector_engine.search_text(q, n_results=6)):
                did = doc["id"]
                text_scores[did] = text_scores.get(did, 0.0) + 1.0 / (K + rank + 1)
                text_docs[did] = doc

            for rank, doc in enumerate(self.vector_engine.search_images(q, n_results=3)):
                did = doc["id"]
                image_scores[did] = image_scores.get(did, 0.0) + 1.0 / (K + rank + 1)
                image_docs[did] = doc

        def _sort(scores, docs, top_n):
            return sorted(
                [{**docs[i], "rrf_score": scores[i]} for i in docs],
                key=lambda x: x["rrf_score"],
                reverse=True,
            )[:top_n]

        text_results = _sort(text_scores, text_docs, 8)
        image_results = _sort(image_scores, image_docs, 3)

        return {
            "text_results": text_results,
            "image_results": image_results,
            "pipeline_trace": [
                f"RRF fusion over {len(all_queries)} queries → "
                f"{len(text_results)} text + {len(image_results)} image results"
            ],
        }

    # 3. Cross-encoder re-ranking (text only; images are kept as-is)
    def _rerank(self, state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        text_results = state["text_results"]
        image_results = state["image_results"]

        if text_results:
            pairs = [(query, doc["text"]) for doc in text_results]
            scores = self.reranker.predict(pairs)
            for doc, s in zip(text_results, scores):
                doc["rerank_score"] = float(s)
            text_results = sorted(
                text_results, key=lambda x: x["rerank_score"], reverse=True
            )[:5]

        reranked = text_results + image_results

        return {
            "reranked_results": reranked,
            "pipeline_trace": [
                f"Cross-encoder reranked {len(text_results)} text passages"
            ],
        }

    # 4. Contextual compression
    def _compress(self, state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        reranked = state["reranked_results"]

        compress_prompt = ChatPromptTemplate.from_template(
            "Extract and return ONLY the sentences from the passage below that are "
            "directly relevant to the question. Preserve exact wording. "
            "If nothing is relevant, return an empty string.\n\n"
            "Question: {query}\n\nPassage:\n{passage}"
        )
        compress_chain = compress_prompt | self.llm | StrOutputParser()

        parts: List[str] = []
        for doc in reranked[:6]:
            if doc["modality"] == "text":
                try:
                    compressed = compress_chain.invoke(
                        {"query": query, "passage": doc["text"]}
                    )
                    if compressed.strip():
                        meta = doc["metadata"]
                        parts.append(
                            f"[{meta.get('title', 'Document')}]\n{compressed.strip()}"
                        )
                except Exception:
                    parts.append(doc["text"])
            else:
                meta = doc["metadata"]
                parts.append(
                    f"[Image: {meta.get('file_name', 'image')}]\n"
                    f"Caption: {doc['text']}"
                )

        context = "\n\n---\n\n".join(parts)

        return {
            "compressed_context": context,
            "pipeline_trace": [
                f"Contextual compression applied to {len(reranked[:6])} passages"
            ],
        }

    # 5. Answer generation
    def _generate(self, state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        context = state.get("compressed_context", "")
        reranked = state["reranked_results"]

        if not context.strip():
            return {
                "answer": (
                    "I could not find relevant information in the knowledge base "
                    "to answer your question."
                ),
                "sources": [],
                "pipeline_trace": ["Generation skipped — no context found"],
            }

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise AI assistant. Answer questions using ONLY "
                    "the provided context. Cite source titles when relevant. "
                    "If the context does not contain the answer, say so clearly.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {query}\n\n"
                    "Provide a clear, well-structured answer.",
                ),
            ]
        )

        try:
            answer: str = (gen_prompt | self.llm | StrOutputParser()).invoke(
                {"context": context, "query": query}
            )
        except Exception as exc:
            answer = f"Generation error: {exc}"

        # Build source list for the UI
        sources: List[Dict[str, Any]] = []
        seen_titles: set = set()
        for doc in reranked[:6]:
            meta = doc["metadata"]
            if doc["modality"] == "text":
                title = meta.get("title", "Document")
                if title not in seen_titles:
                    seen_titles.add(title)
                    sources.append(
                        {
                            "type": "text",
                            "title": title,
                            "file": meta.get("file_name", ""),
                            "score": round(
                                doc.get("rerank_score", doc.get("rrf_score", 0)), 3
                            ),
                        }
                    )
            else:
                fname = meta.get("file_name", "image")
                if fname not in seen_titles:
                    seen_titles.add(fname)
                    sources.append(
                        {
                            "type": "image",
                            "title": fname,
                            "file": fname,
                            "thumbnail": meta.get("thumbnail", ""),
                            "score": round(doc.get("rrf_score", 0), 3),
                        }
                    )

        return {
            "answer": answer,
            "sources": sources,
            "pipeline_trace": [f"Answer generated using {len(sources)} sources"],
        }

    # ------------------------------------------------------------------ #
    #  Public run interface                                                #
    # ------------------------------------------------------------------ #

    def run(self, query: str) -> Dict[str, Any]:
        initial: RAGState = {
            "query": query,
            "hyde_document": "",
            "expanded_queries": [],
            "text_results": [],
            "image_results": [],
            "reranked_results": [],
            "compressed_context": "",
            "answer": "",
            "sources": [],
            "pipeline_trace": [],
        }
        result = self.graph.invoke(initial)

        all_retrieved = result["text_results"] + result["image_results"]
        all_retrieved.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        rrf_results = [
            {
                "title": doc["metadata"].get("title") or doc["metadata"].get("file_name", ""),
                "file": doc["metadata"].get("file_name", ""),
                "rrf_score": round(doc.get("rrf_score", 0), 4),
                "modality": doc.get("modality", "text"),
            }
            for doc in all_retrieved
        ]

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "pipeline_trace": result["pipeline_trace"],
            "hyde_document": result.get("hyde_document", ""),
            "expanded_queries": result.get("expanded_queries", []),
            "rrf_results": rrf_results,
        }

    def check_llm_connection(self) -> bool:
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
