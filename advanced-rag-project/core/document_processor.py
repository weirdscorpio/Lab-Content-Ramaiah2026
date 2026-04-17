import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF


class DocumentProcessor:
    """
    Ingests PDFs and Markdown/text files into the vector store.

    Chunking strategy: sentence-aware overlapping windows — sentences are
    grouped until chunk_size is hit, then the last ~chunk_overlap characters
    of sentences are carried forward as overlap into the next chunk.
    """

    SUPPORTED_TEXT = {".pdf", ".md", ".txt"}

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_directory(self, directory: str, vector_engine) -> Dict[str, int]:
        """Walk *directory*, ingest every supported file into *vector_engine*."""
        stats = {"files": 0, "chunks": 0}
        path = Path(directory)
        if not path.exists():
            return stats

        for fp in sorted(path.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in self.SUPPORTED_TEXT:
                chunks = self._dispatch(str(fp))
                for chunk in chunks:
                    vector_engine.add_text(chunk["text"], chunk["metadata"])
                stats["files"] += 1
                stats["chunks"] += len(chunks)

        return stats

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from every page of a PDF, then chunk it."""
        doc = fitz.open(file_path)
        pages_text: List[str] = []

        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        doc.close()

        full_text = "\n".join(pages_text)
        return self._to_chunks(full_text, file_path, content_type="pdf")

    def process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Read a Markdown or plain-text file and chunk it."""
        text = Path(file_path).read_text(encoding="utf-8")
        return self._to_chunks(text, file_path, content_type="text")

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _dispatch(self, file_path: str) -> List[Dict[str, Any]]:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.process_pdf(file_path)
        return self.process_text_file(file_path)

    def _to_chunks(
        self, text: str, file_path: str, content_type: str
    ) -> List[Dict[str, Any]]:
        file_name = Path(file_path).name
        title = self._extract_title(text) or file_name
        raw_chunks = self._chunk(text)

        return [
            {
                "text": chunk,
                "metadata": {
                    "source": file_path,
                    "file_name": file_name,
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "content_type": content_type,
                },
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def _extract_title(self, text: str) -> Optional[str]:
        for line in text.strip().splitlines()[:8]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line and len(line) < 120 and not line.startswith("#"):
                return line
        return None

    def _chunk(self, text: str) -> List[str]:
        """Sentence-aware overlapping chunker."""
        # Split on sentence boundaries
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

        chunks: List[str] = []
        window: List[str] = []
        size = 0

        for sent in sentences:
            if size + len(sent) > self.chunk_size and window:
                chunks.append(" ".join(window))
                # Build overlap tail
                tail: List[str] = []
                tail_size = 0
                for s in reversed(window):
                    tail_size += len(s)
                    tail.insert(0, s)
                    if tail_size >= self.chunk_overlap:
                        break
                window = tail
                size = sum(len(s) for s in window)

            window.append(sent)
            size += len(sent)

        if window:
            chunks.append(" ".join(window))

        # Drop micro-chunks that carry no information
        return [c for c in chunks if len(c) > 60]
