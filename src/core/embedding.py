from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import mimetypes
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional

from .models import Chunk

logger = logging.getLogger(__name__)

EmbeddingDevice = Literal["auto", "cpu", "cuda"]


@dataclass
class Embedder:
    model_name: str
    device: EmbeddingDevice = "auto"
    output_dimension: int = 3072

    _model: object | None = None
    _gemini_client: object | None = None
    _query_cache: OrderedDict = field(default_factory=OrderedDict)
    _query_cache_max: int = 128

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
                logger.warning("CUDA init failed for SentenceTransformer, falling back to CPU", exc_info=True)
                self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _gemini_dimension(self) -> int:
        return max(128, min(3072, int(self.output_dimension or 3072)))

    def _load_gemini_client(self):
        if self._gemini_client is not None:
            return self._gemini_client

        from .gemini_client import build_gemini_client

        self._gemini_client = build_gemini_client(
            os.getenv("GEMINI_API_KEY", ""),
            model_name=self.model_name,
        )
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

    @staticmethod
    def _is_retryable_error(e: Exception) -> bool:
        """Check if a Gemini API error is retryable (rate limit, server error)."""
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "quota" in err_str:
            return True
        if "500" in err_str or "503" in err_str or "502" in err_str:
            return True
        if "timeout" in err_str or "timed out" in err_str:
            return True
        return False

    def _embed_via_gemini(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts:
            return []

        import time
        from google.genai import types

        client = self._load_gemini_client()
        dim = self._gemini_dimension()
        out: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            last_err: Exception | None = None
            for attempt in range(1, 5):
                try:
                    resp = client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=dim,
                        ),
                    )
                    out.extend(self._extract_embedding_values(resp))
                    break
                except Exception as e:
                    last_err = e
                    if attempt >= 4 or not self._is_retryable_error(e):
                        raise
                    wait = min(30.0, 2.0 * (2 ** (attempt - 1)))
                    logger.warning(
                        "Gemini embedding rate-limited (attempt %d/4), retrying in %.1fs: %s",
                        attempt, wait, e,
                    )
                    time.sleep(wait)
            else:
                if last_err:
                    raise last_err
        return [self._normalize(vec) for vec in out]

    def _embed_chunk_via_gemini(self, chunk: Chunk, task_type: str) -> List[float]:
        from google.genai import types

        client = self._load_gemini_client()
        dim = self._gemini_dimension()
        text_payload = (chunk.text or "").strip() or f"{chunk.file_name} page {chunk.page_start}"

        if chunk.modality == "visual" and chunk.image_path and "embedding-2" in (self.model_name or "").lower():
            try:
                path = Path(chunk.image_path)
                mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
                image_bytes = path.read_bytes()
                resp = client.models.embed_content(
                    model=self.model_name,
                    contents=[
                        types.Part.from_text(text=text_payload),
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    ],
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=dim,
                    ),
                )
                return self._normalize(self._extract_embedding_values(resp)[0])
            except Exception:
                # Fall back to text-only embedding when multimodal embedding is unavailable.
                logger.debug("Multimodal embedding failed for chunk, falling back to text-only", exc_info=True)

        return self._embed_via_gemini([text_payload], task_type=task_type)[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        texts_list = list(texts)
        if self._uses_gemini():
            return self._embed_via_gemini(texts_list, task_type="RETRIEVAL_DOCUMENT")

        model = self._load_local()
        # normalize_embeddings improves cosine similarity behavior.
        embs = model.encode(texts_list, normalize_embeddings=True)
        return embs.tolist()

    def embed_chunks(self, chunks: Iterable[Chunk]) -> List[List[float]]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []
        if self._uses_gemini():
            text_positions: list[int] = []
            text_payloads: list[str] = []
            out: list[List[float] | None] = [None] * len(chunk_list)
            for idx, chunk in enumerate(chunk_list):
                can_use_visual = (
                    chunk.modality == "visual"
                    and bool(chunk.image_path)
                    and "embedding-2" in (self.model_name or "").lower()
                )
                if can_use_visual:
                    out[idx] = self._embed_chunk_via_gemini(chunk, task_type="RETRIEVAL_DOCUMENT")
                    continue
                text_positions.append(idx)
                text_payloads.append((chunk.text or "").strip() or f"{chunk.file_name} page {chunk.page_start}")
            if text_payloads:
                text_embeddings = self._embed_via_gemini(text_payloads, task_type="RETRIEVAL_DOCUMENT")
                for idx, embedding in zip(text_positions, text_embeddings):
                    out[idx] = embedding
            if any(embedding is None for embedding in out):
                raise ValueError("Gemini embedding uretimi tamamlanamadi.")
            return [embedding for embedding in out if embedding is not None]
        return self.embed_texts([chunk.text for chunk in chunk_list])

    def embed_query(self, text: str) -> List[float]:
        # LRU cache for query embeddings — avoids redundant API calls
        # during multi-step retrieval (hybrid search + re-ranking use same query)
        cache_key = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        if cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key]

        if self._uses_gemini():
            result = self._embed_via_gemini([text], task_type="RETRIEVAL_QUERY")[0]
        else:
            result = self.embed_texts([text])[0]

        self._query_cache[cache_key] = result
        if len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)
        return result

