from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable, List, Literal, Optional

from .gemini_client import build_gemini_client

EmbeddingDevice = Literal["auto", "cpu", "cuda"]


@dataclass
class Embedder:
    model_name: str
    device: EmbeddingDevice = "auto"

    _model: object | None = None
    _gemini_client: object | None = None

    def _uses_gemini(self) -> bool:
        model = (self.model_name or "").strip().lower()
        return model.startswith("gemini-embedding")

    def _resolve_device(self) -> str:
        """
        Resolve which device to run embeddings on.

        - cpu: always CPU
        - cuda: require CUDA if available, otherwise fall back to CPU
        - auto: use CUDA if available, else CPU
        """
        dev = (self.device or "auto").strip().lower()
        if dev == "cpu":
            return "cpu"

        # Try to detect CUDA availability. Keep this import local so we don't
        # make torch a hard import at module import time.
        cuda_ok = False
        try:
            import torch  # noqa: WPS433

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if dev == "cuda":
            return "cuda" if cuda_ok else "cpu"
        # auto
        return "cuda" if cuda_ok else "cpu"

    def _load_local(self):
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            device = self._resolve_device()
            try:
                self._model = SentenceTransformer(self.model_name, device=device)
            except Exception:
                # Safety fallback: if CUDA init fails for any reason, fall back to CPU
                # rather than breaking the working system.
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _gemini_dimension(self) -> int:
        raw = (os.getenv("EMBEDDING_DIMENSION", "3072") or "3072").strip()
        try:
            dim = int(raw)
        except Exception:
            dim = 3072
        return max(128, min(3072, dim))

    def _load_gemini_client(self):
        if self._gemini_client is not None:
            return self._gemini_client

        self._gemini_client = build_gemini_client(os.getenv("GEMINI_API_KEY", ""))
        return self._gemini_client

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum((x * x) for x in vec))
        if norm <= 0.0:
            return vec
        return [x / norm for x in vec]

    @staticmethod
    def _extract_embedding_values(resp) -> List[List[float]]:
        values: list[list[float]] = []
        embeddings = getattr(resp, "embeddings", None)
        if embeddings:
            for item in embeddings:
                item_values = getattr(item, "values", None)
                if item_values:
                    values.append([float(x) for x in item_values])
        if values:
            return values

        embedding = getattr(resp, "embedding", None)
        if embedding is not None:
            item_values = getattr(embedding, "values", None)
            if item_values:
                return [[float(x) for x in item_values]]
        raise ValueError("Gemini embedding yaniti parse edilemedi.")

    def _embed_via_gemini(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []

        from google.genai import types

        client = self._load_gemini_client()
        dim = self._gemini_dimension()
        out: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            resp = client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=dim,
                ),
            )
            out.extend(self._extract_embedding_values(resp))
        return [self._normalize(vec) for vec in out]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if self._uses_gemini():
            return self._embed_via_gemini(texts_list, task_type="RETRIEVAL_DOCUMENT")

        model = self._load_local()
        # normalize_embeddings improves cosine similarity behavior.
        embs = model.encode(texts_list, normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self._uses_gemini():
            return self._embed_via_gemini([text], task_type="RETRIEVAL_QUERY")[0]
        return self.embed_texts([text])[0]

