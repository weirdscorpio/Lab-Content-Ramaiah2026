import base64
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from PIL import Image


class ImageProcessor:
    """
    Processes image files:
      1. Generates a caption via LLaVA (Ollama) — falls back to filename if unavailable.
      2. Creates a JPEG thumbnail for the UI.
      3. Returns a CLIP embedding via the provided clip_model.

    Supported formats: JPEG, PNG, WEBP, BMP, GIF.
    """

    SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        thumbnail_dir: str = "./static/thumbnails",
    ):
        self.ollama_host = ollama_host
        self.thumbnail_dir = Path(thumbnail_dir)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self._vision_model: Optional[str] = self._detect_vision_model()
        if self._vision_model:
            print(f"  Vision model available: {self._vision_model}")
        else:
            print("  No vision model found — image captions will use filename only.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_image(
        self, file_path: str, clip_model=None
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a dict with caption, thumbnail URL, dimensions, and CLIP embedding,
        or None if the file cannot be processed.
        """
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED:
            return None

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
                width, height = img_rgb.size

                # Thumbnail
                thumb_name = f"{path.stem}_thumb.jpg"
                thumb_path = self.thumbnail_dir / thumb_name
                thumb = img_rgb.copy()
                thumb.thumbnail((300, 300), Image.LANCZOS)
                thumb.save(str(thumb_path), "JPEG", quality=85)

                # CLIP embedding (needs original size, re-open)
                embedding: Optional[list] = None
                if clip_model is not None:
                    orig = img_rgb.copy()
                    embedding = clip_model.encode(orig, show_progress_bar=False).tolist()

            caption = self._caption(file_path) or f"Image: {path.name}"

            return {
                "file_path": str(file_path),
                "file_name": path.name,
                "caption": caption,
                "thumbnail": f"/static/thumbnails/{thumb_name}",
                "width": width,
                "height": height,
                "embedding": embedding,
            }

        except Exception as exc:
            print(f"  [ImageProcessor] Failed to process {file_path}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _detect_vision_model(self) -> Optional[str]:
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = m.get("name", "")
                    if any(v in name.lower() for v in ["llava", "bakllava", "moondream"]):
                        return name
        except Exception:
            pass
        return None

    def _caption(self, file_path: str) -> Optional[str]:
        if not self._vision_model:
            return None
        try:
            with open(file_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()

            payload = {
                "model": self._vision_model,
                "prompt": (
                    "Describe this image concisely but thoroughly. "
                    "Include the main subject, visible text, colours, and context."
                ),
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.1},
            }
            resp = requests.post(
                f"{self.ollama_host}/api/generate", json=payload, timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as exc:
            print(f"  [ImageProcessor] Caption failed: {exc}")
        return None
