"""
Flask application — Advanced Multimodal RAG
Port: 127.0.0.1:5353
"""

import json
import sys
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from werkzeug.utils import secure_filename

# Make the project root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.document_processor import DocumentProcessor
from core.image_processor import ImageProcessor
from core.rag_pipeline import AdvancedRAGPipeline
from core.vector_engine import MultiModalVectorEngine

# ──────────────────────────────────────────────────────────────────────────────
#  Paths & constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
IMAGES_DIR = BASE_DIR / "images"
CHROMA_DIR = BASE_DIR / "chroma_db"
THUMB_DIR = BASE_DIR / "static" / "thumbnails"

for d in (DOCS_DIR, IMAGES_DIR, CHROMA_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen2:0.5b"

SUPPORTED_DOCS = {".pdf", ".md", ".txt"}
SUPPORTED_IMGS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ──────────────────────────────────────────────────────────────────────────────
#  Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ──────────────────────────────────────────────────────────────────────────────
#  Lazy component initialisation (done once on first request)
# ──────────────────────────────────────────────────────────────────────────────

_vector_engine: MultiModalVectorEngine | None = None
_doc_processor: DocumentProcessor | None = None
_img_processor: ImageProcessor | None = None
_rag_pipeline: AdvancedRAGPipeline | None = None


def _init():
    global _vector_engine, _doc_processor, _img_processor, _rag_pipeline
    if _vector_engine is not None:
        return

    print("\n[init] Loading MultiModal Vector Engine …")
    _vector_engine = MultiModalVectorEngine(persist_dir=str(CHROMA_DIR))

    print("[init] Loading Document Processor …")
    _doc_processor = DocumentProcessor()

    print("[init] Loading Image Processor …")
    _img_processor = ImageProcessor(
        ollama_host=OLLAMA_HOST,
        thumbnail_dir=str(THUMB_DIR),
    )

    print("[init] Building Advanced RAG Pipeline …")
    _rag_pipeline = AdvancedRAGPipeline(
        vector_engine=_vector_engine,
        ollama_host=OLLAMA_HOST,
        model=MODEL,
    )

    # Auto-ingest on first run
    if _vector_engine.is_empty():
        print("[init] Knowledge base empty — ingesting docs/ and images/ …")
        stats = _doc_processor.process_directory(str(DOCS_DIR), _vector_engine)
        print(f"[init] Text ingestion: {stats}")
        img_count = _ingest_images(str(IMAGES_DIR))
        print(f"[init] Images ingested: {img_count}")

    print("[init] Ready.\n")


def _ingest_images(directory: str) -> int:
    count = 0
    for fp in Path(directory).rglob("*"):
        if fp.suffix.lower() in SUPPORTED_IMGS:
            result = _img_processor.process_image(
                str(fp), clip_model=_vector_engine.clip_model
            )
            if result and result["embedding"]:
                _vector_engine.add_image(
                    caption=result["caption"],
                    embedding=result["embedding"],
                    metadata={
                        "file_path": str(fp),
                        "file_name": result["file_name"],
                        "thumbnail": result["thumbnail"],
                        "content_type": "image",
                    },
                )
                count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    _init()
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    _init()
    stats = _vector_engine.get_stats()
    llm_ok = _rag_pipeline.check_llm_connection()
    return jsonify(
        {
            "status": "ready",
            "llm_connected": llm_ok,
            "model": MODEL,
            "ollama_host": OLLAMA_HOST,
            "stats": stats,
        }
    )


@app.route("/api/documents")
def api_documents():
    _init()
    docs = sorted(
        f.name for f in DOCS_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_DOCS
    )
    images = sorted(
        f.name for f in IMAGES_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_IMGS
    )
    return jsonify({"documents": docs, "images": images})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    _init()
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    result = _rag_pipeline.run(query)
    return jsonify(result)


@app.route("/api/chat/stream", methods=["POST"])
def api_chat_stream():
    _init()
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def _generate():
        result = _rag_pipeline.run(query)

        # 1. Pipeline internals (HyDE doc, expanded queries, RRF scores)
        yield f"data: {json.dumps({'type': 'pipeline_details', 'hyde_document': result['hyde_document'], 'expanded_queries': result['expanded_queries'], 'rrf_results': result['rrf_results']})}\n\n"

        # 2. Pipeline trace
        yield f"data: {json.dumps({'type': 'trace', 'steps': result['pipeline_trace']})}\n\n"

        # 2. Stream answer tokens
        for word in result["answer"].split():
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        # 3. Sources
        yield f"data: {json.dumps({'type': 'sources', 'sources': result['sources']})}\n\n"

        # 4. Done sentinel
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/upload", methods=["POST"])
def api_upload():
    _init()
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()

    if ext in SUPPORTED_DOCS:
        save_path = DOCS_DIR / filename
        file.save(str(save_path))
        if ext == ".pdf":
            chunks = _doc_processor.process_pdf(str(save_path))
        else:
            chunks = _doc_processor.process_text_file(str(save_path))
        for chunk in chunks:
            _vector_engine.add_text(chunk["text"], chunk["metadata"])
        return jsonify(
            {
                "message": f"Indexed {len(chunks)} chunks from {filename}",
                "type": "document",
                "chunks": len(chunks),
            }
        )

    if ext in SUPPORTED_IMGS:
        save_path = IMAGES_DIR / filename
        file.save(str(save_path))
        result = _img_processor.process_image(
            str(save_path), clip_model=_vector_engine.clip_model
        )
        if result and result["embedding"]:
            _vector_engine.add_image(
                caption=result["caption"],
                embedding=result["embedding"],
                metadata={
                    "file_path": str(save_path),
                    "file_name": result["file_name"],
                    "thumbnail": result["thumbnail"],
                    "content_type": "image",
                },
            )
            return jsonify(
                {
                    "message": f"Indexed image {filename}",
                    "type": "image",
                    "caption": result["caption"],
                    "thumbnail": result["thumbnail"],
                }
            )
        return jsonify({"error": "Failed to process image"}), 500

    return jsonify({"error": f"Unsupported file type: {ext}"}), 400


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init()
    app.run(host="127.0.0.1", port=5353, debug=False)
