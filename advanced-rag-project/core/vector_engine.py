from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from PIL import Image
import chromadb
import uuid


class _TextEF(EmbeddingFunction):
    """SentenceTransformer embedding function for text (384-dim)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"  Loading text embedding model: {model_name} ...")
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()

    def encode(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


class _ClipEF(EmbeddingFunction):
    """CLIP embedding function for images and cross-modal text search (512-dim)."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        print(f"  Loading CLIP model: {model_name} ...")
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(list(input), show_progress_bar=False).tolist()

    def encode_text(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()

    def encode_image(self, image: Image.Image) -> List[float]:
        return self._model.encode(image, show_progress_bar=False).tolist()

    @property
    def model(self) -> SentenceTransformer:
        return self._model


class MultiModalVectorEngine:
    """
    Manages two ChromaDB collections:
      - text_docs  : text chunks from PDFs / Markdown (all-MiniLM-L6-v2, 384-dim)
      - image_docs : images indexed by CLIP caption embedding (clip-ViT-B-32, 512-dim)
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)

        self._text_ef = _TextEF()
        self._clip_ef = _ClipEF()

        self.text_collection = self._client.get_or_create_collection(
            name="text_docs",
            embedding_function=self._text_ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self._client.get_or_create_collection(
            name="image_docs",
            embedding_function=self._clip_ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Public properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def clip_model(self) -> SentenceTransformer:
        return self._clip_ef.model

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def add_text(self, text: str, metadata: Dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())
        # ChromaDB metadata values must be str/int/float/bool
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.text_collection.add(ids=[doc_id], documents=[text], metadatas=[safe_meta])
        return doc_id

    def add_image(
        self,
        caption: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> str:
        doc_id = str(uuid.uuid4())
        safe_meta = {k: str(v) for k, v in metadata.items()}
        self.image_collection.add(
            ids=[doc_id],
            documents=[caption],
            embeddings=[embedding],
            metadatas=[safe_meta],
        )
        return doc_id

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def search_text(self, query: str, n_results: int = 6) -> List[Dict[str, Any]]:
        n = min(n_results, self.text_collection.count())
        if n == 0:
            return []
        raw = self.text_collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return self._fmt(raw, "text")

    def search_images(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        n = min(n_results, self.image_collection.count())
        if n == 0:
            return []
        q_emb = self._clip_ef.encode_text(query)
        raw = self.image_collection.query(
            query_embeddings=[q_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return self._fmt(raw, "image")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _fmt(self, raw: Dict, modality: str) -> List[Dict[str, Any]]:
        if not raw["ids"] or not raw["ids"][0]:
            return []
        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            dist = raw["distances"][0][i]
            results.append(
                {
                    "id": doc_id,
                    "text": raw["documents"][0][i],
                    "metadata": raw["metadatas"][0][i],
                    "score": max(0.0, 1.0 - dist),
                    "modality": modality,
                    "rank": i + 1,
                }
            )
        return results

    def get_stats(self) -> Dict[str, int]:
        return {
            "text_chunks": self.text_collection.count(),
            "images": self.image_collection.count(),
            "total": self.text_collection.count() + self.image_collection.count(),
        }

    def is_empty(self) -> bool:
        return self.get_stats()["total"] == 0
