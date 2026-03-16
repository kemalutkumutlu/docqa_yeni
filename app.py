"""
Chainlit UI — Document Q&A with Multimodal Hierarchical RAG.

Run:
    python -m chainlit run app.py -w
"""
from __future__ import annotations

import asyncio
import hmac
import importlib.util
import json
import os
import re
import shutil
import signal
import sqlite3
import tempfile
import time
from queue import Empty, SimpleQueue
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider, Tab, TextInput
from chainlit.types import ThreadDict
from chainlit.user import User
from chainlit.context import ChainlitContextException, context as cl_context

from src.config import load_settings
from src.core.ocr_backend import OCRConfig
from src.core.local_llm import OllamaConfig, ollama_is_available
from src.core.pipeline import RAGPipeline
from src.core.table_structure import TableStructureConfig
from src.core.vlm_extract import VLMConfig
from src.ui_text import (
    build_evidence_panel as _build_evidence_panel,
    build_qa_debug_suffix as _build_qa_debug_suffix,
    embedding_runtime_label as _embedding_runtime_label,
    format_standard_error as _format_standard_error,
    looks_like_chat_mode_request as _looks_like_chat_mode_request,
    looks_like_doc_mode_request as _looks_like_doc_mode_request,
    looks_like_doc_switch as _looks_like_doc_switch_base,
    looks_like_smalltalk as _looks_like_smalltalk,
    render_sidebar_panel as _render_sidebar_panel,
    shorten_for_sidebar as _shorten_for_sidebar,
    smalltalk_style as _smalltalk_style,
)

try:
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
except Exception as _native_history_import_error:  # pragma: no cover - optional dependency path
    SQLAlchemyDataLayer = object
else:
    _native_history_import_error = None

try:
    from engineio.payload import Payload as _EngineIOPayload
except Exception:
    _EngineIOPayload = None
else:
    raw_max_packets = (os.getenv("ENGINEIO_MAX_DECODE_PACKETS", "64") or "64").strip()
    try:
        _engineio_max_packets = max(16, min(512, int(raw_max_packets)))
    except Exception:
        _engineio_max_packets = 64
    _EngineIOPayload.max_decode_packets = _engineio_max_packets


# ── Helpers ──────────────────────────────────────────────────────────────────

# Auto-exit (dev convenience): if enabled, kill the server when the last client disconnects.
# This prevents "port already in use" when you close the browser tab but forget the terminal.
_ACTIVE_CHAT_SESSIONS = 0
_EXIT_TASK: asyncio.Task | None = None
_EXIT_LOCK = asyncio.Lock()
_THREAD_TAG_RE = re.compile(r"^<!--THREAD:([A-Za-z0-9:_\-.]+)-->\s*")
_OPEN_THREAD_CMD_RE = re.compile(r"^/open_thread(?:\s+([A-Za-z0-9:_\-.]+))?\s*$", re.IGNORECASE)
_THREAD_MEMORY: dict[str, list[dict[str, str]]] = {}
_THREAD_PIPELINES: dict[str, RAGPipeline] = {}
_THREAD_LAST_USED: dict[str, float] = {}
_THREAD_MEMORY_MAX_MSGS = 120
_THREAD_CACHE_MAX = 100
_THREAD_CACHE_TTL_SECONDS = 6 * 60 * 60
_SIDEBAR_REV_KEY = "sidebar_render_rev"
_THREAD_STATE_KEY = "docqa_thread_state"
_THREAD_STATE_VERSION = 1
_NATIVE_HISTORY_BOOTSTRAP_DONE = False
_UPLOAD_WORK_SEMAPHORE: asyncio.Semaphore | None = None
_DOC_QA_SEMAPHORE: asyncio.Semaphore | None = None
_CHAT_QA_SEMAPHORE: asyncio.Semaphore | None = None


def _auto_exit_enabled() -> bool:
    v = (os.getenv("AUTO_EXIT_ON_NO_CLIENTS", "") or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _auto_exit_grace_seconds() -> float:
    raw = (os.getenv("AUTO_EXIT_GRACE_SECONDS", "8") or "").strip()
    try:
        sec = float(raw)
    except Exception:
        sec = 8.0
    return max(0.0, min(120.0, sec))


def _env_truthy(name: str, default: str = "") -> bool:
    value = (os.getenv(name, default) or "").strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 32) -> int:
    raw = (os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _upload_work_semaphore() -> asyncio.Semaphore:
    global _UPLOAD_WORK_SEMAPHORE
    if _UPLOAD_WORK_SEMAPHORE is None:
        _UPLOAD_WORK_SEMAPHORE = asyncio.Semaphore(
            _env_int("DOCQA_UPLOAD_CONCURRENCY", 2, minimum=1, maximum=8)
        )
    return _UPLOAD_WORK_SEMAPHORE


def _doc_qa_semaphore() -> asyncio.Semaphore:
    global _DOC_QA_SEMAPHORE
    if _DOC_QA_SEMAPHORE is None:
        _DOC_QA_SEMAPHORE = asyncio.Semaphore(
            _env_int("DOCQA_QA_CONCURRENCY", 3, minimum=1, maximum=12)
        )
    return _DOC_QA_SEMAPHORE


def _chat_qa_semaphore() -> asyncio.Semaphore:
    global _CHAT_QA_SEMAPHORE
    if _CHAT_QA_SEMAPHORE is None:
        _CHAT_QA_SEMAPHORE = asyncio.Semaphore(
            _env_int("DOCQA_CHAT_CONCURRENCY", 6, minimum=1, maximum=16)
        )
    return _CHAT_QA_SEMAPHORE


def _native_history_requested() -> bool:
    return _env_truthy("CHAINLIT_NATIVE_HISTORY", "0")


def _native_history_auth_enabled() -> bool:
    return bool((os.getenv("CHAINLIT_AUTH_USERNAME", "") or "").strip()) and bool(
        (os.getenv("CHAINLIT_AUTH_PASSWORD", "") or "").strip()
    )


def _mark_thread_used(thread_id: str | None) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    _THREAD_LAST_USED[tid] = time.time()


def _drop_thread_cache(thread_id: str | None) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    _THREAD_MEMORY.pop(tid, None)
    _THREAD_PIPELINES.pop(tid, None)
    _THREAD_LAST_USED.pop(tid, None)


def _cleanup_thread_caches() -> None:
    now = time.time()
    stale_ids = [
        tid
        for tid, last_used in list(_THREAD_LAST_USED.items())
        if now - float(last_used or 0.0) > _THREAD_CACHE_TTL_SECONDS
    ]
    for tid in stale_ids:
        _drop_thread_cache(tid)

    overflow = max(0, len(_THREAD_LAST_USED) - _THREAD_CACHE_MAX)
    if overflow <= 0:
        return
    for tid, _ in sorted(_THREAD_LAST_USED.items(), key=lambda item: item[1])[:overflow]:
        _drop_thread_cache(tid)


def _native_history_conninfo() -> str | None:
    explicit = (os.getenv("CHAINLIT_HISTORY_DATABASE_URL", "") or "").strip()
    if explicit:
        return explicit

    database_url = (os.getenv("DATABASE_URL", "") or "").strip()
    if database_url:
        return database_url

    db_path = (os.getenv("CHAINLIT_DB_PATH", "") or "").strip()
    if not db_path:
        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        db_path = str(data_dir / "chainlit_history.db")

    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return f"sqlite+aiosqlite:///{path}"


def _native_history_ready() -> tuple[bool, str | None]:
    if not _native_history_requested():
        return False, "disabled"
    if SQLAlchemyDataLayer is None:
        detail = str(_native_history_import_error or "sqlalchemy data layer unavailable")
        return False, detail
    conninfo = _native_history_conninfo()
    if not conninfo:
        return False, "missing database url"
    if conninfo.startswith("sqlite+aiosqlite:"):
        try:
            import aiosqlite  # noqa: F401,WPS433
        except Exception as exc:  # pragma: no cover - optional dependency path
            return False, str(exc)
    return True, None


def _native_history_sqlite_path(conninfo: str) -> Path | None:
    prefix = "sqlite+aiosqlite:///"
    if not conninfo.startswith(prefix):
        return None
    raw = conninfo[len(prefix):]
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _native_history_sqlite_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _bootstrap_native_history_sqlite(conninfo: str) -> None:
    global _NATIVE_HISTORY_BOOTSTRAP_DONE
    if _NATIVE_HISTORY_BOOTSTRAP_DONE:
        return
    db_path = _native_history_sqlite_path(conninfo)
    if db_path is None:
        _NATIVE_HISTORY_BOOTSTRAP_DONE = True
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            identifier TEXT NOT NULL UNIQUE,
            "createdAt" TEXT NOT NULL,
            metadata TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            "createdAt" TEXT,
            name TEXT,
            "userId" TEXT,
            "userIdentifier" TEXT,
            tags TEXT,
            metadata TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS steps (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            "threadId" TEXT,
            "parentId" TEXT,
            streaming INTEGER,
            "waitForAnswer" INTEGER,
            "isError" INTEGER,
            metadata TEXT,
            tags TEXT,
            input TEXT,
            output TEXT,
            "createdAt" TEXT,
            start TEXT,
            "end" TEXT,
            "defaultOpen" INTEGER,
            generation TEXT,
            "showInput" TEXT,
            language TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feedbacks (
            id TEXT PRIMARY KEY,
            "forId" TEXT,
            value REAL,
            comment TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS elements (
            id TEXT PRIMARY KEY,
            "threadId" TEXT,
            type TEXT,
            "chainlitKey" TEXT,
            url TEXT,
            "objectKey" TEXT,
            name TEXT,
            display TEXT,
            size TEXT,
            language TEXT,
            page INTEGER,
            "forId" TEXT,
            mime TEXT,
            props TEXT,
            "autoPlay" INTEGER,
            "playerConfig" TEXT
        )
        """,
        'CREATE INDEX IF NOT EXISTS idx_users_identifier ON users(identifier)',
        'CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads("userId")',
        'CREATE INDEX IF NOT EXISTS idx_steps_thread_id ON steps("threadId")',
        'CREATE INDEX IF NOT EXISTS idx_feedbacks_for_id ON feedbacks("forId")',
        'CREATE INDEX IF NOT EXISTS idx_elements_thread_id ON elements("threadId")',
    ]

    with _native_history_sqlite_connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        for statement in ddl:
            conn.execute(statement)
        step_columns = {row[1] for row in conn.execute('PRAGMA table_info("steps")')}
        if "defaultOpen" not in step_columns:
            conn.execute('ALTER TABLE steps ADD COLUMN "defaultOpen" INTEGER')
        conn.commit()
    _NATIVE_HISTORY_BOOTSTRAP_DONE = True


def _native_history_backfill_user_binding(conninfo: str, identifier: str) -> None:
    db_path = _native_history_sqlite_path(conninfo)
    if db_path is None or not identifier:
        return

    with _native_history_sqlite_connect(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT id FROM users WHERE identifier = ? ORDER BY rowid DESC LIMIT 1",
            (identifier,),
        ).fetchone()
        if not row:
            return
        user_id = row[0]
        cur.execute(
            'UPDATE threads SET "userId" = ?, "userIdentifier" = ? WHERE ("userId" IS NULL OR "userId" = "")',
            (user_id, identifier),
        )
        unnamed = cur.execute(
            'SELECT id, metadata FROM threads WHERE (name IS NULL OR name = "")'
        ).fetchall()
        for thread_id, raw_metadata in unnamed:
            metadata = {}
            if isinstance(raw_metadata, str) and raw_metadata.strip():
                try:
                    metadata = json.loads(raw_metadata)
                except Exception:
                    metadata = {}
            elif isinstance(raw_metadata, dict):
                metadata = raw_metadata
            title = _thread_name_from_metadata(metadata)
            if title:
                cur.execute('UPDATE threads SET name = ? WHERE id = ?', (title, thread_id))
        conn.commit()


async def _native_history_backfill_user_binding_async(conninfo: str, identifier: str) -> None:
    db_path = _native_history_sqlite_path(conninfo)
    if db_path is None or not identifier:
        return
    try:
        import aiosqlite  # noqa: WPS433
    except Exception:
        await asyncio.to_thread(_native_history_backfill_user_binding, conninfo, identifier)
        return

    async with aiosqlite.connect(str(db_path), timeout=30) as conn:
        await conn.execute("PRAGMA busy_timeout=30000")
        async with conn.execute(
            "SELECT id FROM users WHERE identifier = ? ORDER BY rowid DESC LIMIT 1",
            (identifier,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return
        user_id = row[0]
        await conn.execute(
            'UPDATE threads SET "userId" = ?, "userIdentifier" = ? WHERE ("userId" IS NULL OR "userId" = "")',
            (user_id, identifier),
        )
        async with conn.execute(
            'SELECT id, metadata FROM threads WHERE (name IS NULL OR name = "")'
        ) as cursor:
            unnamed = await cursor.fetchall()
        for thread_id, raw_metadata in unnamed:
            metadata = {}
            if isinstance(raw_metadata, str) and raw_metadata.strip():
                try:
                    metadata = json.loads(raw_metadata)
                except Exception:
                    metadata = {}
            elif isinstance(raw_metadata, dict):
                metadata = raw_metadata
            title = _thread_name_from_metadata(metadata)
            if title:
                await conn.execute('UPDATE threads SET name = ? WHERE id = ?', (title, thread_id))
        await conn.commit()


class NativeHistoryDataLayer(SQLAlchemyDataLayer):  # type: ignore[misc]
    _THREAD_UPSERT_COLUMNS = (
        "id",
        "createdAt",
        "name",
        "userId",
        "userIdentifier",
        "tags",
        "metadata",
    )

    async def _resolve_current_user_binding(self) -> tuple[str | None, str | None]:
        try:
            session_user = cl_context.session.user
        except ChainlitContextException:
            session_user = None
        except Exception:
            session_user = None

        if session_user is None:
            return None, None

        identifier = getattr(session_user, "identifier", None)
        user_id = getattr(session_user, "id", None)
        if user_id:
            return str(user_id), str(identifier or "")

        if identifier:
            persisted = await self.get_user(str(identifier))
            if persisted and getattr(persisted, "id", None):
                return str(persisted.id), str(identifier)
            return None, str(identifier)
        return None, None

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ):
        resolved_name = name or _thread_name_from_metadata(metadata)
        resolved_user_id = user_id
        resolved_identifier = None
        if not resolved_user_id:
            resolved_user_id, resolved_identifier = await self._resolve_current_user_binding()
        merged_metadata = None
        if metadata is not None:
            existing = await self.execute_sql(
                query='SELECT "metadata" FROM threads WHERE "id" = :id',
                parameters={"id": thread_id},
            )
            base_metadata = {}
            if isinstance(existing, list) and existing:
                raw_existing = existing[0].get("metadata") or {}
                if isinstance(raw_existing, str):
                    try:
                        base_metadata = json.loads(raw_existing)
                    except json.JSONDecodeError:
                        base_metadata = {}
                elif isinstance(raw_existing, dict):
                    base_metadata = raw_existing
            incoming_metadata = {k: v for k, v in metadata.items() if v is not None}
            merged_metadata = {**base_metadata, **incoming_metadata}

        effective_name = resolved_name
        if effective_name is None and isinstance(merged_metadata, dict):
            meta_name = merged_metadata.get("name")
            if meta_name is not None:
                effective_name = str(meta_name).strip() or None

        payload = {
            "id": thread_id,
            "createdAt": await self.get_current_timestamp(),
            "name": effective_name,
            "userId": resolved_user_id,
            "userIdentifier": resolved_identifier,
            "tags": json.dumps(tags) if tags is not None else None,
            "metadata": json.dumps(merged_metadata) if merged_metadata is not None else None,
        }
        parameters = {
            key: value
            for key, value in payload.items()
            if key in self._THREAD_UPSERT_COLUMNS and value is not None
        }
        columns = ", ".join(f'"{key}"' for key in parameters.keys())
        values = ", ".join(f":{key}" for key in parameters.keys())
        updates = ", ".join(
            f'"{key}" = :{key}' for key in parameters.keys() if key != "id"
        )
        query = f"""
            INSERT INTO threads ({columns})
            VALUES ({values})
            ON CONFLICT ("id") DO UPDATE
            SET {updates};
        """
        await self.execute_sql(query=query, parameters=parameters)

    async def create_step(self, step_dict):
        step_copy = dict(step_dict)
        if isinstance(step_copy.get("tags"), list):
            step_copy["tags"] = json.dumps(step_copy["tags"])
        await super().create_step(step_copy)


def _native_history_status_note() -> str | None:
    if not _native_history_requested():
        return None
    ready, detail = _native_history_ready()
    if not ready:
        return f"Native history backend hazir degil: {detail or 'unknown'}"
    if not _native_history_auth_enabled():
        return (
            "Native history backend hazir, ancak login tanimli degil. "
            "Gercek thread listeleme/resume icin `CHAINLIT_AUTH_USERNAME` ve "
            "`CHAINLIT_AUTH_PASSWORD` tanimlaman gerekir."
        )
    return None


_NATIVE_HISTORY_READY, _NATIVE_HISTORY_ERROR = _native_history_ready()

if _NATIVE_HISTORY_READY and SQLAlchemyDataLayer is not None:
    @cl.data_layer
    def get_data_layer():
        conninfo = _native_history_conninfo()
        assert conninfo is not None
        _bootstrap_native_history_sqlite(conninfo)
        connect_args = {"timeout": 30} if conninfo.startswith("sqlite+aiosqlite:///") else None
        return NativeHistoryDataLayer(
            conninfo=conninfo,
            connect_args=connect_args,
            storage_provider=None,
            user_thread_limit=5000,
            show_logger=False,
        )


if _native_history_requested() and _native_history_auth_enabled():
    @cl.password_auth_callback
    async def password_auth(username: str, password: str):
        expected_user = (os.getenv("CHAINLIT_AUTH_USERNAME", "") or "").strip()
        expected_pass = os.getenv("CHAINLIT_AUTH_PASSWORD", "") or ""
        display_name = (os.getenv("CHAINLIT_AUTH_DISPLAY_NAME", "") or "").strip() or expected_user
        if not (
            hmac.compare_digest(username or "", expected_user)
            and hmac.compare_digest(password or "", expected_pass)
        ):
            return None
        return User(
            identifier=expected_user,
            display_name=display_name,
            metadata={"provider": "local-password"},
        )


def _extract_thread_marker(text: str) -> tuple[str | None, str]:
    raw = (text or "").strip()
    m = _THREAD_TAG_RE.match(raw)
    if not m:
        return None, raw
    thread_id = (m.group(1) or "").strip()
    rest = raw[m.end():].strip()
    return (thread_id or None), rest


def _thread_history_dir() -> Path:
    configured = (os.getenv("THREAD_HISTORY_DIR", "") or "").strip()
    if configured:
        return Path(configured)
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    return data_dir / "thread_history"


def _thread_history_path(thread_id: str | None) -> Path | None:
    tid = (thread_id or "").strip()
    if not tid:
        return None
    return _thread_history_dir() / f"{tid}.json"


def _thread_history_payload_load(thread_id: str | None) -> dict[str, Any]:
    tid = (thread_id or "").strip()
    if not tid:
        return {}
    path = _thread_history_path(tid)
    if path is None or not path.exists():
        return {"thread_id": tid}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"thread_id": tid}
    return raw if isinstance(raw, dict) else {"thread_id": tid}


def _thread_history_payload_save(thread_id: str | None, payload: dict[str, Any]) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    path = _thread_history_path(tid)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(payload or {})
        data["thread_id"] = tid
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _thread_memory_load(thread_id: str | None) -> list[dict[str, str]]:
    tid = (thread_id or "").strip()
    if not tid:
        return []
    _cleanup_thread_caches()
    cached = _THREAD_MEMORY.get(tid)
    if cached is not None:
        _mark_thread_used(tid)
        return cached

    raw = _thread_history_payload_load(tid)
    items = raw.get("messages") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []

    loaded: list[dict[str, str]] = []
    for item in items[-_THREAD_MEMORY_MAX_MSGS:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip()
        content = re.sub(r"\s+", " ", (item.get("content") or "").strip())
        if role not in ("user", "assistant") or not content:
            continue
        loaded.append({"role": role, "content": content})
    if loaded:
        _THREAD_MEMORY[tid] = loaded
        _mark_thread_used(tid)
    return loaded


def _thread_memory_persist(thread_id: str | None) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    messages = _THREAD_MEMORY.get(tid, [])
    try:
        payload = _thread_history_payload_load(tid)
        payload = {
            **payload,
            "message_count": len(messages),
            "messages": messages[-_THREAD_MEMORY_MAX_MSGS:],
        }
        _thread_history_payload_save(tid, payload)
    except Exception:
        # Thread persistence is best-effort and must not break chat flow.
        return


def _thread_memory_add(thread_id: str | None, role: str, content: str) -> None:
    tid = (thread_id or "").strip()
    msg = re.sub(r"\s+", " ", (content or "").strip())
    if not tid or not msg or role not in ("user", "assistant"):
        return
    buf = list(_thread_memory_load(tid))
    if buf and buf[-1].get("role") == role and buf[-1].get("content") == msg:
        return
    buf.append({"role": role, "content": msg})
    if len(buf) > _THREAD_MEMORY_MAX_MSGS:
        buf = buf[-_THREAD_MEMORY_MAX_MSGS:]
    _THREAD_MEMORY[tid] = buf
    _mark_thread_used(tid)
    _cleanup_thread_caches()
    _thread_memory_persist(tid)


def _thread_pipeline_get(thread_id: str | None) -> RAGPipeline | None:
    tid = (thread_id or "").strip()
    if not tid:
        return None
    _cleanup_thread_caches()
    pipeline = _THREAD_PIPELINES.get(tid)
    if pipeline is not None:
        _mark_thread_used(tid)
    return pipeline


def _thread_pipeline_set(thread_id: str | None, pipeline: RAGPipeline | None) -> None:
    tid = (thread_id or "").strip()
    if not tid or pipeline is None:
        return
    _THREAD_PIPELINES[tid] = pipeline
    _mark_thread_used(tid)
    _cleanup_thread_caches()


def _normalize_thread_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    mode = (raw.get("mode") or "").strip().lower()
    if mode not in ("chat", "doc"):
        mode = "doc"
    runtime_overrides = raw.get("runtime_overrides")
    if not isinstance(runtime_overrides, dict):
        runtime_overrides = {}
    documents = raw.get("documents")
    if not isinstance(documents, list):
        documents = []
    documents = [str(item).strip() for item in documents if str(item or "").strip()]
    return {
        "version": int(raw.get("version") or _THREAD_STATE_VERSION),
        "mode": mode,
        "active_doc": str(raw.get("active_doc") or "").strip(),
        "documents": documents,
        "runtime_overrides": runtime_overrides,
        "profile_warning": str(raw.get("profile_warning") or "").strip(),
    }


def _thread_name_from_metadata(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    recent = metadata.get("recent_user_messages")
    if isinstance(recent, list):
        for item in reversed(recent):
            text = str(item or "").strip()
            if text:
                return _shorten_for_sidebar(text, limit=72)
    state = metadata.get(_THREAD_STATE_KEY)
    if isinstance(state, dict):
        docs = state.get("documents")
        if isinstance(docs, list) and docs:
            text = str(docs[-1] or "").strip()
            if text:
                return _shorten_for_sidebar(text, limit=72)
    return None


def _collect_thread_state(pipeline: RAGPipeline | None = None) -> dict[str, Any]:
    pipeline = _resolve_sidebar_pipeline(pipeline)
    docs = pipeline.list_documents() if pipeline else []
    return _normalize_thread_state(
        {
            "version": _THREAD_STATE_VERSION,
            "mode": cl.user_session.get("mode") or "doc",
            "active_doc": pipeline.active_document_name if pipeline else "",
            "documents": docs,
            "runtime_overrides": _runtime_overrides(),
            "profile_warning": cl.user_session.get(_PROFILE_WARNING_KEY) or "",
        }
    )


def _thread_state_load(thread_id: str | None) -> dict[str, Any]:
    tid = (thread_id or "").strip()
    if not tid:
        return {}
    payload = _thread_history_payload_load(tid)
    return _normalize_thread_state(payload.get("thread_state"))


def _thread_state_persist(thread_id: str | None, pipeline: RAGPipeline | None = None) -> None:
    tid = (thread_id or "").strip()
    state = _collect_thread_state(pipeline)
    cl.user_session.set(_THREAD_STATE_KEY, state)
    if not tid:
        return
    payload = _thread_history_payload_load(tid)
    payload["thread_state"] = state
    _thread_history_payload_save(tid, payload)


def _thread_state_from_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return _normalize_thread_state(metadata.get(_THREAD_STATE_KEY))


def _restore_thread_state(
    thread_id: str | None,
    *,
    metadata: Any = None,
    pipeline: RAGPipeline | None = None,
) -> RAGPipeline | None:
    tid = (thread_id or "").strip()
    state = _thread_state_from_metadata(metadata)
    if not state:
        state = _thread_state_load(tid)

    if not state:
        if tid:
            cl.user_session.set("ui_thread_id", tid)
        return pipeline

    if tid:
        cl.user_session.set("ui_thread_id", tid)
    cl.user_session.set("mode", state.get("mode") or "doc")
    cl.user_session.set(_RUNTIME_OVERRIDES_KEY, state.get("runtime_overrides") or {})
    cl.user_session.set(_THREAD_STATE_KEY, state)
    cl.user_session.set(_PROFILE_WARNING_KEY, state.get("profile_warning") or "")

    if pipeline is None:
        pipeline = _thread_pipeline_get(tid) or cl.user_session.get("pipeline")
    if pipeline is None:
        pipeline = _get_pipeline()
        cl.user_session.set("pipeline", pipeline)
    else:
        # Thread/session state may be restored onto an already-live pipeline object.
        # Re-apply runtime overrides so sidebar and runtime stay aligned after reload/resume.
        _apply_runtime_overrides_to_pipeline(pipeline)
        cl.user_session.set("pipeline", pipeline)

    active_doc = (state.get("active_doc") or "").strip()
    if active_doc and pipeline and pipeline.has_documents:
        pipeline.set_active_document(active_doc)

    if tid and pipeline is not None:
        _thread_pipeline_set(tid, pipeline)
    return pipeline


def _active_thread_id() -> str | None:
    tid = (cl.user_session.get("ui_thread_id") or "").strip()
    return tid or None


def _resolve_sidebar_pipeline(preferred: RAGPipeline | None = None) -> RAGPipeline | None:
    # When an explicit pipeline is passed (e.g. after file upload), prefer it
    # over thread pipeline — the caller has the most up-to-date reference.
    if preferred is not None:
        return preferred

    active_tid = (cl.user_session.get("ui_thread_id") or "").strip()
    if active_tid:
        thread_pipeline = _thread_pipeline_get(active_tid)
        if thread_pipeline is not None:
            return thread_pipeline

    session_pipeline = cl.user_session.get("pipeline")
    if session_pipeline is not None:
        return session_pipeline

    return None


def _next_sidebar_rev() -> int:
    cur = cl.user_session.get(_SIDEBAR_REV_KEY)
    if not isinstance(cur, int):
        cur = 0
    cur += 1
    cl.user_session.set(_SIDEBAR_REV_KEY, cur)
    return cur


async def _cancel_exit_task() -> None:
    global _EXIT_TASK
    if _EXIT_TASK and not _EXIT_TASK.done():
        _EXIT_TASK.cancel()
    _EXIT_TASK = None


async def _schedule_auto_exit_if_idle() -> None:
    """
    If enabled and no active sessions remain, exit the process after a grace period.
    """
    global _EXIT_TASK
    if not _auto_exit_enabled():
        return
    grace = _auto_exit_grace_seconds()

    async def _worker() -> None:
        await asyncio.sleep(grace)
        # Double-check state right before exiting.
        async with _EXIT_LOCK:
            if _ACTIVE_CHAT_SESSIONS <= 0:
                print("[auto-exit] requesting graceful shutdown", flush=True)
                signal.raise_signal(signal.SIGINT)

    await _cancel_exit_task()
    _EXIT_TASK = asyncio.create_task(_worker())


ACCEPTED_MIME = [
    "application/pdf",
    "image/png",
    "image/jpeg",
]

_CHAT_PROFILE_TO_PROVIDER = {
    "Gemini": "gemini",
    "OpenAI": "openai",
    "Local": "local",
    "Extractive": "none",
}
_CHAT_HISTORY_KEY = "recent_user_messages"
_CHAT_HISTORY_MAX = 12
_RUNTIME_OVERRIDES_KEY = "runtime_overrides"
_RUNTIME_PRESET_VALUES = ["custom", "online_best", "hybrid_best", "local_best", "fast"]
_PROCESSING_MODE_VALUES = ["classic", "multimodal", "smart"]
_TABLE_STRUCTURE_MODE_VALUES = ["off", "on", "smart"]
_MULTIMODAL_ANSWER_MODE_VALUES = ["off", "auto", "on"]
_VISUAL_CHUNK_LEVEL_VALUES = ["page", "region"]
_VISUAL_REGION_SOURCE_VALUES = ["heuristic", "detector"]
_VISUAL_DETECTOR_BACKEND_VALUES = ["none", "docai", "docling", "sidecar"]
_TOGGLE_VALUES = ["on", "off"]
_PDF_TEXT_BACKEND_VALUES = ["auto", "pymupdf", "docling", "smart"]
_OCR_BACKEND_VALUES = ["docai", "paddle_vl", "paddle", "tesseract_legacy", "smart"]
_TABLE_STRUCTURE_BACKEND_VALUES = ["off", "auto", "docai", "gemini", "heuristic"]
_LLM_PROVIDER_VALUES = ["gemini", "openai", "local", "none"]
_GENERATION_MODEL_PRESETS = [
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gpt-4o-mini",
    "qwen2.5:7b",
]
_EMBEDDING_MODEL_PRESETS = [
    "gemini-embedding-001",
    "gemini-embedding-2-preview",
    "auto",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
]
_EMBEDDING_DEVICE_VALUES = ["auto", "cpu", "cuda"]
_VLM_MODE_VALUES = ["off", "auto", "smart", "force"]
_VLM_PROVIDER_VALUES = ["gemini", "local"]
_VLM_MAX_PAGES_LIMIT = 200
_VLM_MAX_PAGES_DEFAULT = 25
_PROFILE_WARNING_KEY = "profile_warning"


def _get_cached_settings():
    try:
        settings = cl.user_session.get("app_settings")
    except ChainlitContextException:
        return load_settings()
    if settings is None:
        settings = load_settings()
        cl.user_session.set("app_settings", settings)
    return settings


def _default_profile_name() -> str:
    settings = load_settings()
    provider = (settings.llm_provider or "gemini").strip().lower()
    for profile_name, profile_provider in _CHAT_PROFILE_TO_PROVIDER.items():
        if profile_provider == provider:
            return profile_name
    return "Gemini"


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


def _sanitize_select_value(value, allowed: list[str], fallback: str) -> str:
    text = (str(value or "")).strip().lower()
    if text in allowed:
        return text
    return fallback


def _resolve_embedding_model_choice(choice: str, embedding_device: str) -> str:
    selected = (choice or "").strip().lower()
    if selected.startswith("gemini-embedding-") and importlib.util.find_spec("google.genai") is None:
        selected = "auto"
        choice = "auto"
    if selected == "auto":
        use_cuda = _cuda_available() and embedding_device != "cpu"
        return "intfloat/multilingual-e5-base" if use_cuda else "intfloat/multilingual-e5-small"
    return (choice or "").strip()


def _normalize_toggle_value(value, fallback: str) -> str:
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "y", "on", "enabled"):
        return "on"
    if text in ("0", "false", "no", "n", "off", "disabled"):
        return "off"
    return fallback


def _toggle_to_bool(value, fallback: bool) -> bool:
    normalized = _normalize_toggle_value(value, "on" if fallback else "off")
    return normalized == "on"


def _llm_model_for_provider(pipeline: RAGPipeline, provider: str) -> str:
    resolved = (provider or "").strip().lower()
    if resolved == "openai":
        return (pipeline.openai_model or "").strip() or "gpt-4o-mini"
    if resolved == "local":
        if pipeline.ollama_config:
            return (pipeline.ollama_config.llm_model or "").strip() or "qwen2.5:7b"
        return "qwen2.5:7b"
    if resolved == "none":
        return "extractive"
    return (pipeline.gemini_model or "").strip() or "gemini-3.1-pro-preview"


def _generation_model_values(pipeline: RAGPipeline, settings, provider: str, current_choice: str) -> list[str]:
    resolved = (provider or "").strip().lower()
    values: list[str] = []
    if resolved == "gemini":
        for model_name in (
            getattr(pipeline, "gemini_model", ""),
            getattr(settings, "gemini_model", ""),
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
        ):
            clean = str(model_name or "").strip()
            if clean and clean not in values:
                values.append(clean)
    elif resolved == "openai":
        for model_name in (
            getattr(pipeline, "openai_model", ""),
            getattr(settings, "openai_model", ""),
            "gpt-4o-mini",
        ):
            clean = str(model_name or "").strip()
            if clean and clean not in values:
                values.append(clean)
    elif resolved == "local":
        for model_name in (
            getattr(getattr(pipeline, "ollama_config", None), "llm_model", ""),
            getattr(settings, "ollama_llm_model", ""),
            "qwen2.5:7b",
        ):
            clean = str(model_name or "").strip()
            if clean and clean not in values:
                values.append(clean)
    if current_choice:
        clean = str(current_choice).strip()
        if clean and clean not in values and resolved != "none":
            values.append(clean)
    return values


def _normalize_generation_model_choice(pipeline: RAGPipeline, settings, provider: str, choice: str) -> str:
    values = _generation_model_values(pipeline, settings, provider, choice)
    clean = str(choice or "").strip()
    if clean and clean in values:
        return clean
    return _llm_model_for_provider(pipeline, provider)


def _settings_dependency_summary(
    *,
    processing_mode: str,
    visual_chunk_level: str,
    visual_region_source: str,
    visual_detector_backend: str,
    ocr_enabled: str,
    ocr_backend: str,
    table_enabled: str,
    table_backend: str,
    llm_provider: str,
    generation_model: str,
    vlm_mode: str,
    vlm_provider: str,
    vlm_max_pages: int,
    pdf_text_backend: str = "pymupdf",
) -> str:
    layout_line = "disabled"
    if processing_mode != "multimodal":
        layout_line = "kapali: processing_mode=classic"
    elif visual_chunk_level != "region":
        layout_line = f"kapali: visual_chunk_level={visual_chunk_level}"
    elif visual_region_source != "detector":
        layout_line = f"kapali: visual_region_source={visual_region_source}"
    elif visual_detector_backend == "none":
        layout_line = "heuristic fallback: detector backend=none"
    else:
        layout_line = f"aktif: {visual_detector_backend}"

    multimodal_answer_line = (
        "anlamli"
        if processing_mode == "multimodal"
        else "sinirli: processing_mode=classic"
    )
    table_line = (
        f"aktif: {table_backend}" + (" (smart)" if table_enabled == "smart" else "")
        if table_enabled in ("on", "smart")
        else "kapali"
    )
    generation_line = (
        "extractive (LLM yok)"
        if llm_provider == "none"
        else f"{llm_provider} / {generation_model}"
    )
    vlm_line = (
        "kapali"
        if vlm_mode == "off"
        else f"{vlm_provider} / {vlm_mode} / max_pages={vlm_max_pages}"
    )
    return "\n".join(
        [
            f"PDF text: {pdf_text_backend}",
            f"OCR: {ocr_enabled} / {ocr_backend}",
            f"Multimodal answer: {multimodal_answer_line}",
            f"Layout detector: {layout_line}",
            f"Table structure: {table_line}",
            f"Generation: {generation_line}",
            f"VLM: {vlm_line}",
        ]
    )


def _runtime_dependency_flags(
    *,
    processing_mode: str,
    visual_chunk_level: str,
    visual_region_source: str,
    ocr_enabled: str,
    table_enabled: str,
    llm_provider: str,
    embedding_model_choice: str,
    vlm_mode: str,
) -> dict[str, bool]:
    multimodal_active = processing_mode in ("multimodal", "smart")
    region_mode_active = multimodal_active and visual_chunk_level == "region"
    detector_mode_active = region_mode_active and visual_region_source == "detector"
    ocr_active = ocr_enabled == "on"
    table_stage_available = multimodal_active
    table_active = table_stage_available and table_enabled in ("on", "smart")
    llm_active = llm_provider != "none"
    llm_supports_multimodal_answer = llm_provider == "gemini"
    embedding_device_relevant = not str(embedding_model_choice or "").strip().lower().startswith("gemini-embedding-")
    vlm_active = vlm_mode != "off"
    return {
        "generation_model": not llm_active,
        "ocr_backend": not ocr_active,
        "visual_chunk_level": not multimodal_active,
        "table_structure_enabled": not table_stage_available,
        "multimodal_answer_mode": not (multimodal_active and llm_supports_multimodal_answer),
        "visual_region_source": not region_mode_active,
        "visual_detector_backend": not detector_mode_active,
        "table_structure_backend": not table_active,
        "vlm_provider": not vlm_active,
        "vlm_max_pages": not vlm_active,
        "embedding_device": not embedding_device_relevant,
    }


def _preset_defaults(pipeline: RAGPipeline, settings, preset: str) -> dict[str, str | int]:
    selected = (preset or "custom").strip().lower()
    docai_ocr_ready = bool(getattr(settings, "docai_ocr_processor_id", ""))
    docai_layout_ready = bool(getattr(settings, "docai_layout_processor_id", ""))
    docling_ready = bool(getattr(settings, "docling_python_bin", ""))
    online_ocr_backend = "docai" if docai_ocr_ready else "paddle_vl"
    online_layout_backend = "docai" if docai_layout_ready else ("docling" if docling_ready else "none")
    if selected == "online_best":
        return {
            "ocr_enabled": "on",
            "ocr_backend": online_ocr_backend,
            "processing_mode": "multimodal",
            "multimodal_answer_mode": "auto",
            "visual_chunk_level": "region",
            "visual_region_source": "detector",
            "visual_detector_backend": online_layout_backend,
            "table_structure_enabled": "on",
            "table_structure_backend": "auto",
            "llm_provider": "gemini",
            "generation_model_choice": (settings.gemini_model or pipeline.gemini_model or "gemini-3.1-pro-preview"),
            "embedding_model_choice": "gemini-embedding-2-preview",
            "embedding_device": "cuda" if _cuda_available() else "auto",
            "vlm_mode": "force",
            "vlm_provider": "gemini",
            "vlm_max_pages": 25,
        }
    if selected == "hybrid_best":
        return {
            "ocr_enabled": "on",
            "ocr_backend": "paddle_vl",
            "processing_mode": "multimodal",
            "multimodal_answer_mode": "auto",
            "visual_chunk_level": "region",
            "visual_region_source": "detector",
            "visual_detector_backend": online_layout_backend,
            "table_structure_enabled": "on",
            "table_structure_backend": "auto",
            "llm_provider": "gemini",
            "generation_model_choice": (settings.gemini_model or pipeline.gemini_model or "gemini-3.1-pro-preview"),
            "embedding_model_choice": "gemini-embedding-2-preview",
            "embedding_device": "cuda" if _cuda_available() else "auto",
            "vlm_mode": "force",
            "vlm_provider": "gemini",
            "vlm_max_pages": 25,
        }
    if selected == "local_best":
        return {
            "ocr_enabled": "on",
            "ocr_backend": "paddle_vl",
            "processing_mode": "multimodal",
            "multimodal_answer_mode": "auto",
            "visual_chunk_level": "region",
            "visual_region_source": "detector",
            "visual_detector_backend": "docling",
            "table_structure_enabled": "on",
            "table_structure_backend": "heuristic",
            "llm_provider": "local",
            "generation_model_choice": (settings.ollama_llm_model or getattr(getattr(pipeline, "ollama_config", None), "llm_model", "qwen2.5:7b") or "qwen2.5:7b"),
            "embedding_model_choice": "auto",
            "embedding_device": "cuda" if _cuda_available() else "auto",
            "vlm_mode": "auto",
            "vlm_provider": "local",
            "vlm_max_pages": 10,
        }
    if selected == "fast":
        return {
            "ocr_enabled": "on",
            "ocr_backend": "paddle",
            "processing_mode": "classic",
            "multimodal_answer_mode": "off",
            "visual_chunk_level": "page",
            "visual_region_source": "heuristic",
            "visual_detector_backend": "none",
            "table_structure_enabled": "off",
            "table_structure_backend": "off",
            "llm_provider": "gemini",
            "generation_model_choice": (settings.gemini_model or pipeline.gemini_model or "gemini-3.1-pro-preview"),
            "embedding_model_choice": "auto",
            "embedding_device": "auto",
            "vlm_mode": "off",
            "vlm_provider": "gemini",
            "vlm_max_pages": 5,
        }
    return {}


def _effective_settings_values(pipeline: RAGPipeline) -> dict[str, str | int]:
    current_vlm = pipeline.vlm_config
    current_table = pipeline.table_structure_config
    current_ocr = pipeline.ocr_config
    return {
        "ocr_enabled": "on" if getattr(current_ocr, "enabled", True) else "off",
        "ocr_backend": getattr(current_ocr, "backend", "docai"),
        "processing_mode": getattr(pipeline, "processing_mode", "classic"),
        "multimodal_answer_mode": getattr(pipeline, "multimodal_answer_mode", "auto"),
        "visual_chunk_level": getattr(pipeline, "visual_chunk_level", "page"),
        "visual_region_source": getattr(pipeline, "visual_region_source", "heuristic"),
        "visual_detector_backend": getattr(pipeline, "visual_detector_backend", "none"),
        "table_structure_enabled": ("smart" if bool(getattr(current_table, "smart", False)) else "on") if bool(getattr(current_table, "enabled", False)) else "off",
        "table_structure_backend": getattr(current_table, "backend", "auto"),
        "llm_provider": getattr(pipeline, "llm_provider", "gemini"),
        "generation_model_choice": _llm_model_for_provider(pipeline, getattr(pipeline, "llm_provider", "gemini")),
        "embedding_model_choice": getattr(pipeline, "embedding_model", "auto"),
        "embedding_device": getattr(pipeline, "embedding_device", "auto"),
        "vlm_mode": getattr(current_vlm, "mode", "auto") if current_vlm else "auto",
        "vlm_provider": getattr(current_vlm, "provider", "gemini") if current_vlm else "gemini",
        "vlm_max_pages": int(getattr(current_vlm, "max_pages", _VLM_MAX_PAGES_DEFAULT) if current_vlm else _VLM_MAX_PAGES_DEFAULT),
        "pdf_text_backend": getattr(pipeline, "pdf_text_backend", "pymupdf"),
    }


def _matches_preset(values: dict[str, str | int], preset: str, pipeline: RAGPipeline, settings) -> bool:
    defaults = _preset_defaults(pipeline, settings, preset)
    if not defaults:
        return False
    for key, preset_value in defaults.items():
        actual_value = values.get(key)
        expected_value = preset_value
        if key == "embedding_model_choice":
            expected_value = _resolve_embedding_model_choice(
                str(preset_value),
                str(defaults.get("embedding_device", "auto") or "auto"),
            )
        if actual_value != expected_value:
            return False
    return True


def _settings_fallback_summary(
    *,
    pipeline: RAGPipeline,
    settings,
    ocr_enabled: str,
    ocr_backend: str,
    visual_region_source: str,
    visual_detector_backend: str,
    llm_provider: str,
    vlm_provider: str,
    table_enabled: str,
    table_backend: str,
) -> str:
    lines: list[str] = []
    if ocr_enabled != "on":
        lines.append("OCR fallback: yok, cunku OCR kapali.")
    elif ocr_backend == "docai" and not getattr(pipeline.ocr_config, "docai_processor_id", ""):
        lines.append("OCR fallback: DOCAI_OCR_PROCESSOR_ID yoksa legacy fallback devreye girer.")
    elif ocr_backend == "paddle_vl":
        lines.append("OCR fallback: paddle_vl basarisiz olursa paddle -> legacy zinciri calisir.")
    elif ocr_backend == "paddle":
        lines.append("OCR fallback: paddle basarisiz olursa legacy zinciri calisir.")
    else:
        lines.append("OCR fallback: secili backend dogrudan kullanilir.")

    if visual_region_source != "detector":
        lines.append("Layout fallback: detector zinciri kapali, heuristic region planning kullanilir.")
    elif visual_detector_backend == "none":
        lines.append("Layout fallback: detector backend=none oldugu icin heuristic kullanilir.")
    elif visual_detector_backend == "docai" and not getattr(pipeline, "docai_layout_processor_id", ""):
        lines.append("Layout fallback: DOCAI_LAYOUT_PROCESSOR_ID yoksa heuristic kullanilir.")
    elif visual_detector_backend == "docling" and not getattr(pipeline, "docling_python_bin", ""):
        lines.append("Layout fallback: DOCLING_PYTHON_BIN yoksa heuristic kullanilir.")
    else:
        lines.append(f"Layout path: detector backend `{visual_detector_backend}` ile calisir.")

    resolved_llm_provider, llm_warning = _resolve_llm_provider_choice(
        llm_provider,
        settings,
        pipeline.ollama_config or OllamaConfig(),
    )
    if llm_warning:
        lines.append(f"Generation fallback: {llm_warning}")
    else:
        lines.append(f"Generation path: `{resolved_llm_provider}` aktif.")

    embedding_choice = str(getattr(pipeline, "embedding_model", "") or "")
    if embedding_choice.startswith("gemini-embedding-") and not _gemini_sdk_available():
        lines.append("Embedding fallback: `google-genai` paketi yok, local embedding fallback kullanilir.")

    resolved_vlm_provider, vlm_warning = _resolve_vlm_provider_choice(
        vlm_provider,
        settings,
        pipeline.ollama_config or OllamaConfig(),
    )
    if vlm_warning:
        lines.append(f"VLM fallback: {vlm_warning}")
    else:
        lines.append(f"VLM path: `{resolved_vlm_provider}` aktif.")

    if table_enabled not in ("on", "smart"):
        lines.append("Table stage: kapali.")
    elif table_backend == "off":
        lines.append("Table stage: backend=off oldugu icin kapali.")
    elif table_backend == "docai" and not getattr(getattr(pipeline, "table_structure_config", None), "docai_processor_id", ""):
        lines.append("Table fallback: docai processor yoksa fallback backend gerekir.")
    else:
        lines.append(f"Table path: `{table_backend}` secili.")

    return "\n".join(lines)


def _settings_documents_summary(pipeline: RAGPipeline) -> str:
    docs = pipeline.list_documents() if pipeline else []
    active_doc = pipeline.active_document_name if pipeline else ""
    mode = cl.user_session.get("mode") or "doc"

    lines = [
        f"Mode: {mode}",
        f"Active doc: {active_doc or '-'}",
        f"Loaded docs: {len(docs)}",
    ]
    if docs:
        lines.extend(f"- {doc}" for doc in docs[:6])
        if len(docs) > 6:
            lines.append(f"... +{len(docs) - 6} more")
    else:
        lines.append("- Henuz belge yuklenmedi")
    return "\n".join(lines)


def _runtime_overrides() -> dict:
    try:
        raw = cl.user_session.get(_RUNTIME_OVERRIDES_KEY)
    except ChainlitContextException:
        return {}
    return raw if isinstance(raw, dict) else {}


def _get_runtime_value(key: str, fallback):
    return _runtime_overrides().get(key, fallback)


def _gemini_sdk_available() -> bool:
    return importlib.util.find_spec("google.genai") is not None


def _gemini_auth_stack_available() -> bool:
    return _gemini_sdk_available() and importlib.util.find_spec("google.auth") is not None


def _has_gemini_backend(settings) -> bool:
    try:
        from src.core.gemini_client import use_vertex_ai
    except Exception:
        return False
    if not _gemini_auth_stack_available():
        return False
    if use_vertex_ai():
        return True
    return bool((settings.gemini_api_key or "").strip())


def _resolve_llm_provider_choice(requested_provider: str, settings, ollama_cfg: OllamaConfig) -> tuple[str, str | None]:
    provider = (requested_provider or "").strip().lower()
    gemini_ready = _has_gemini_backend(settings)
    openai_ready = bool((settings.openai_api_key or "").strip())
    ollama_ready, ollama_detail = ollama_is_available(ollama_cfg)

    if provider == "none":
        return "none", None
    if provider == "gemini":
        if gemini_ready:
            return "gemini", None
        if openai_ready:
            return "openai", "Gemini auth hazir degil. Gecici olarak `OpenAI` kullaniliyor."
        if ollama_ready:
            return "local", "Gemini auth hazir degil. Gecici olarak `Local` kullaniliyor."
        return "none", "Gemini auth hazir degil. Gecici olarak `Extractive` moda dusuldu."
    if provider == "openai":
        if openai_ready:
            return "openai", None
        if gemini_ready:
            return "gemini", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Gemini` fallback kullaniliyor."
        if ollama_ready:
            return "local", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Local` fallback kullaniliyor."
        return "none", "Secilen profil `OpenAI`, ancak `OPENAI_API_KEY` tanimli degil. `Extractive` fallback kullaniliyor."
    if provider == "local":
        if ollama_ready:
            return "local", None
        if gemini_ready:
            return "gemini", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `Gemini` fallback kullaniliyor."
        if openai_ready:
            return "openai", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `OpenAI` fallback kullaniliyor."
        return "none", f"Secilen profil `Local`, ancak Ollama erisilemedi (`{ollama_detail}`). `Extractive` fallback kullaniliyor."
    return "gemini", None


def _resolve_vlm_provider_choice(requested_provider: str, settings, ollama_cfg: OllamaConfig) -> tuple[str, str | None]:
    provider = (requested_provider or "").strip().lower()
    gemini_ready = _has_gemini_backend(settings)
    ollama_ready, ollama_detail = ollama_is_available(ollama_cfg)

    if provider == "local":
        if ollama_ready:
            return "local", None
        if gemini_ready:
            return "gemini", f"Secilen `VLM Provider=local`, ancak Ollama erisilemedi (`{ollama_detail}`). `gemini` fallback kullaniliyor."
        return "local", f"Secilen `VLM Provider=local`, ancak Ollama erisilemedi (`{ollama_detail}`). Yukleme sirasinda VLM devre disi kalabilir."

    if provider == "gemini" and not gemini_ready:
        if ollama_ready:
            return "local", "Gemini VLM auth hazir degil. `local` fallback kullaniliyor."
        return "gemini", "Gemini VLM auth hazir degil. Yukleme sirasinda VLM devre disi kalabilir."

    return provider or "gemini", None


def _embedding_choice_values(pipeline: RAGPipeline) -> list[str]:
    embedding_values = list(_EMBEDDING_MODEL_PRESETS)
    current_choice = str(_get_runtime_value("embedding_model_choice", pipeline.embedding_model) or "").strip()
    if pipeline.embedding_model not in embedding_values:
        embedding_values.append(pipeline.embedding_model)
    if current_choice and current_choice not in embedding_values:
        embedding_values.append(current_choice)
    return embedding_values


def _settings_sidebar_payload(pipeline: RAGPipeline) -> dict[str, Any]:
    settings = _get_cached_settings()
    effective = _effective_settings_values(pipeline)
    runtime_preset = _sanitize_select_value(
        _get_runtime_value("runtime_preset", "custom"),
        _RUNTIME_PRESET_VALUES,
        "custom",
    )
    saved = {"runtime_preset": runtime_preset, **effective}
    generation_models = {
        provider: _generation_model_values(
            pipeline,
            settings,
            provider,
            str(saved.get("generation_model_choice", "")),
        )
        for provider in ("gemini", "openai", "local")
    }
    dependency_summary = _settings_dependency_summary(
        processing_mode=str(saved.get("processing_mode", "classic")),
        visual_chunk_level=str(saved.get("visual_chunk_level", "page")),
        visual_region_source=str(saved.get("visual_region_source", "heuristic")),
        visual_detector_backend=str(saved.get("visual_detector_backend", "none")),
        ocr_enabled=str(saved.get("ocr_enabled", "on")),
        ocr_backend=str(saved.get("ocr_backend", "docai")),
        table_enabled=str(saved.get("table_structure_enabled", "off")),
        table_backend=str(saved.get("table_structure_backend", "auto")),
        llm_provider=str(saved.get("llm_provider", "gemini")),
        generation_model=str(saved.get("generation_model_choice", "")),
        vlm_mode=str(saved.get("vlm_mode", "auto")),
        vlm_provider=str(saved.get("vlm_provider", "gemini")),
        vlm_max_pages=int(saved.get("vlm_max_pages", _VLM_MAX_PAGES_DEFAULT)),
        pdf_text_backend=str(saved.get("pdf_text_backend", "pymupdf")),
    )
    fallback_summary = _settings_fallback_summary(
        pipeline=pipeline,
        settings=settings,
        ocr_enabled=str(saved.get("ocr_enabled", "on")),
        ocr_backend=str(saved.get("ocr_backend", "docai")),
        visual_region_source=str(saved.get("visual_region_source", "heuristic")),
        visual_detector_backend=str(saved.get("visual_detector_backend", "none")),
        llm_provider=str(saved.get("llm_provider", "gemini")),
        vlm_provider=str(saved.get("vlm_provider", "gemini")),
        table_enabled=str(saved.get("table_structure_enabled", "off")),
        table_backend=str(saved.get("table_structure_backend", "auto")),
    )
    preset_defaults = {
        preset: _preset_defaults(pipeline, settings, preset)
        for preset in _RUNTIME_PRESET_VALUES
        if preset != "custom"
    }
    return {
        "source": "docqa-settings-server",
        "kind": "state",
        "payload": {
            "saved": saved,
            "options": {
                "runtime_preset": list(_RUNTIME_PRESET_VALUES),
                "processing_mode": list(_PROCESSING_MODE_VALUES),
                "ocr_enabled": list(_TOGGLE_VALUES),
                "ocr_backend": list(_OCR_BACKEND_VALUES),
                "llm_provider": list(_LLM_PROVIDER_VALUES),
                "generation_models": generation_models,
                "embedding_model": _embedding_choice_values(pipeline),
                "embedding_device": list(_EMBEDDING_DEVICE_VALUES),
                "vlm_mode": list(_VLM_MODE_VALUES),
                "vlm_provider": list(_VLM_PROVIDER_VALUES),
                "visual_chunk_level": list(_VISUAL_CHUNK_LEVEL_VALUES),
                "visual_region_source": list(_VISUAL_REGION_SOURCE_VALUES),
                "visual_detector_backend": list(_VISUAL_DETECTOR_BACKEND_VALUES),
                "pdf_text_backend": list(_PDF_TEXT_BACKEND_VALUES),
                "table_structure_enabled": list(_TABLE_STRUCTURE_MODE_VALUES),
                "table_structure_backend": list(_TABLE_STRUCTURE_BACKEND_VALUES),
                "multimodal_answer_mode": list(_MULTIMODAL_ANSWER_MODE_VALUES),
                "vlm_max_pages": {
                    "min": 0,
                    "max": _VLM_MAX_PAGES_LIMIT,
                    "default": _VLM_MAX_PAGES_DEFAULT,
                },
            },
            "preset_defaults": preset_defaults,
            "summaries": {
                "applied": dependency_summary,
                "fallback": fallback_summary,
                "documents": _settings_documents_summary(pipeline),
            },
        },
    }


async def _send_settings_sidebar_state(pipeline: RAGPipeline | None = None) -> None:
    if pipeline is None:
        pipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        return
    try:
        await cl.send_window_message(_settings_sidebar_payload(pipeline))
    except Exception:
        return


def _settings_widgets(pipeline: RAGPipeline) -> list:
    vlm_cfg = pipeline.vlm_config
    table_cfg = pipeline.table_structure_config
    ocr_cfg = pipeline.ocr_config
    processing_mode_current = _sanitize_select_value(
        _get_runtime_value("processing_mode", getattr(pipeline, "processing_mode", "classic")),
        _PROCESSING_MODE_VALUES,
        getattr(pipeline, "processing_mode", "classic"),
    )
    multimodal_answer_mode_current = _sanitize_select_value(
        _get_runtime_value("multimodal_answer_mode", getattr(pipeline, "multimodal_answer_mode", "auto")),
        _MULTIMODAL_ANSWER_MODE_VALUES,
        getattr(pipeline, "multimodal_answer_mode", "auto"),
    )
    embedding_device_current = _sanitize_select_value(
        _get_runtime_value("embedding_device", pipeline.embedding_device),
        _EMBEDDING_DEVICE_VALUES,
        "auto",
    )
    embedding_choice_current = str(
        _get_runtime_value("embedding_model_choice", pipeline.embedding_model)
    ).strip()
    embedding_values = list(_EMBEDDING_MODEL_PRESETS)
    if pipeline.embedding_model not in embedding_values:
        embedding_values.append(pipeline.embedding_model)
    if embedding_choice_current and embedding_choice_current not in embedding_values:
        embedding_values.append(embedding_choice_current)
    if not embedding_choice_current:
        embedding_choice_current = pipeline.embedding_model

    vlm_mode_current = _sanitize_select_value(
        _get_runtime_value("vlm_mode", vlm_cfg.mode if vlm_cfg else "auto"),
        _VLM_MODE_VALUES,
        vlm_cfg.mode if vlm_cfg else "auto",
    )
    vlm_provider_current = _sanitize_select_value(
        _get_runtime_value("vlm_provider", vlm_cfg.provider if vlm_cfg else "gemini"),
        _VLM_PROVIDER_VALUES,
        vlm_cfg.provider if vlm_cfg else "gemini",
    )
    visual_chunk_level_current = _sanitize_select_value(
        _get_runtime_value("visual_chunk_level", getattr(pipeline, "visual_chunk_level", "page")),
        _VISUAL_CHUNK_LEVEL_VALUES,
        getattr(pipeline, "visual_chunk_level", "page"),
    )
    visual_region_source_current = _sanitize_select_value(
        _get_runtime_value("visual_region_source", getattr(pipeline, "visual_region_source", "heuristic")),
        _VISUAL_REGION_SOURCE_VALUES,
        getattr(pipeline, "visual_region_source", "heuristic"),
    )
    visual_detector_backend_current = _sanitize_select_value(
        _get_runtime_value("visual_detector_backend", getattr(pipeline, "visual_detector_backend", "none")),
        _VISUAL_DETECTOR_BACKEND_VALUES,
        getattr(pipeline, "visual_detector_backend", "none"),
    )
    ocr_enabled_current = _normalize_toggle_value(
        _get_runtime_value("ocr_enabled", "on" if getattr(ocr_cfg, "enabled", True) else "off"),
        "on" if getattr(ocr_cfg, "enabled", True) else "off",
    )
    ocr_backend_current = _sanitize_select_value(
        _get_runtime_value("ocr_backend", getattr(ocr_cfg, "backend", "docai")),
        _OCR_BACKEND_VALUES,
        getattr(ocr_cfg, "backend", "docai"),
    )
    _table_default = ("smart" if getattr(table_cfg, "smart", False) else "on") if getattr(table_cfg, "enabled", False) else "off"
    table_enabled_current = _sanitize_select_value(
        _get_runtime_value("table_structure_enabled", _table_default),
        _TABLE_STRUCTURE_MODE_VALUES,
        _table_default,
    )
    table_backend_current = _sanitize_select_value(
        _get_runtime_value("table_structure_backend", getattr(table_cfg, "backend", "auto")),
        _TABLE_STRUCTURE_BACKEND_VALUES,
        getattr(table_cfg, "backend", "auto"),
    )
    llm_provider_current = _sanitize_select_value(
        _get_runtime_value("llm_provider", getattr(pipeline, "llm_provider", "gemini")),
        _LLM_PROVIDER_VALUES,
        getattr(pipeline, "llm_provider", "gemini"),
    )
    settings = _get_cached_settings()
    generation_model_current = _normalize_generation_model_choice(
        pipeline,
        settings,
        llm_provider_current,
        str(_get_runtime_value("generation_model_choice", _llm_model_for_provider(pipeline, llm_provider_current))).strip(),
    )
    generation_model_values = _generation_model_values(
        pipeline,
        settings,
        llm_provider_current,
        generation_model_current,
    )
    vlm_pages_current = _clamp_int(
        _get_runtime_value("vlm_max_pages", vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT),
        vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT,
        0,
        _VLM_MAX_PAGES_LIMIT,
    )
    multimodal_active = processing_mode_current in ("multimodal", "smart")
    region_mode_active = multimodal_active and visual_chunk_level_current == "region"
    detector_mode_active = region_mode_active and visual_region_source_current == "detector"
    ocr_active = ocr_enabled_current == "on"
    table_stage_available = multimodal_active
    table_active = table_enabled_current in ("on", "smart") and table_stage_available
    llm_active = llm_provider_current != "none"
    llm_supports_multimodal_answer = llm_provider_current == "gemini"
    vlm_active = vlm_mode_current != "off"
    embedding_device_relevant = not str(embedding_choice_current or "").strip().lower().startswith("gemini-embedding-")
    dependency_summary = _settings_dependency_summary(
        processing_mode=processing_mode_current,
        visual_chunk_level=visual_chunk_level_current,
        visual_region_source=visual_region_source_current,
        visual_detector_backend=visual_detector_backend_current,
        ocr_enabled=ocr_enabled_current,
        ocr_backend=ocr_backend_current,
        table_enabled=table_enabled_current,
        table_backend=table_backend_current,
        llm_provider=llm_provider_current,
        generation_model=generation_model_current,
        vlm_mode=vlm_mode_current,
        vlm_provider=vlm_provider_current,
        vlm_max_pages=vlm_pages_current,
    )
    runtime_preset_current = _sanitize_select_value(
        _get_runtime_value("runtime_preset", "custom"),
        _RUNTIME_PRESET_VALUES,
        "custom",
    )
    fallback_summary = _settings_fallback_summary(
        pipeline=pipeline,
        settings=settings,
        ocr_enabled=ocr_enabled_current,
        ocr_backend=ocr_backend_current,
        visual_region_source=visual_region_source_current,
        visual_detector_backend=visual_detector_backend_current,
        llm_provider=llm_provider_current,
        vlm_provider=vlm_provider_current,
        table_enabled=table_enabled_current,
        table_backend=table_backend_current,
    )

    basic_widgets = [
        Select(
            id="runtime_preset",
            label="Runtime Preset",
            values=_RUNTIME_PRESET_VALUES,
            initial_value=runtime_preset_current,
            description="Hazir ayar kombinasyonlari. Manual override yaparsan preset otomatik olarak custom'a dusebilir.",
        ),
        Select(
            id="processing_mode",
            label="Processing Mode",
            values=_PROCESSING_MODE_VALUES,
            initial_value=processing_mode_current,
            description="classic: mevcut text-first akış. multimodal: visual page chunk + multimodal retrieval.",
        ),
        Select(
            id="ocr_enabled",
            label="OCR",
            values=_TOGGLE_VALUES,
            initial_value=ocr_enabled_current,
            description="Scanned/image belgelerde OCR katmanini acip kapatir.",
        ),
        Select(
            id="llm_provider",
            label="LLM Provider",
            values=_LLM_PROVIDER_VALUES,
            initial_value=llm_provider_current,
            description="Cevap uretim yolu. none secilirse extractive yanit kullanilir.",
        ),
    ]

    if llm_active:
        basic_widgets.append(
            Select(
                id="generation_model",
                label="Generation Model",
                values=generation_model_values,
                initial_value=generation_model_current,
                description="Secili LLM provider icin kullanilacak model.",
            )
        )
    advanced_widgets = []
    advanced_widgets.append(
        Select(
            id="embedding_model",
            label="Embedding Model",
            values=embedding_values,
            initial_value=embedding_choice_current,
            description="Varsayilan: Gemini embedding. auto: lokal e5-base/e5-small secilir.",
        )
    )
    advanced_widgets.append(
        Select(
            id="embedding_device",
            label="Embedding Device",
            values=_EMBEDDING_DEVICE_VALUES,
            initial_value=embedding_device_current,
            description="Sadece lokal embedding modellerinde anlamli.",
            disabled=not embedding_device_relevant,
        )
    )
    advanced_widgets.append(
        Select(
            id="vlm_mode",
            label="VLM Mode",
            values=_VLM_MODE_VALUES,
            initial_value=vlm_mode_current,
        )
    )
    advanced_widgets.append(
        Select(
            id="ocr_backend",
            label="OCR Backend",
            values=_OCR_BACKEND_VALUES,
            initial_value=ocr_backend_current,
            description="OCR acikken anlamli. paddle_vl en guclu lokal yol, docai online OCR yoludur.",
            disabled=not ocr_active,
        )
    )
    advanced_widgets.append(
        Select(
            id="visual_chunk_level",
            label="Visual Chunk Level",
            values=_VISUAL_CHUNK_LEVEL_VALUES,
            initial_value=visual_chunk_level_current,
            description="Multimodal acikken anlamli. region secilirse region-source ve detector zinciri devreye girer.",
            disabled=not multimodal_active,
        )
    )
    advanced_widgets.append(
        Select(
            id="table_structure_enabled",
            label="Table Structure",
            values=_TABLE_STRUCTURE_MODE_VALUES,
            initial_value=table_enabled_current,
            description="on: her sayfada calisir. smart: sadece taranmis/OCR sayfalarda calisir (native PDF tablolari zaten islenir).",
            disabled=not table_stage_available,
        )
    )
    advanced_widgets.append(
        Select(
            id="multimodal_answer_mode",
            label="Multimodal Answer Generation",
            values=_MULTIMODAL_ANSWER_MODE_VALUES,
            initial_value=multimodal_answer_mode_current,
            description="Gemini + multimodal zincirinde anlamli.",
            disabled=not (multimodal_active and llm_supports_multimodal_answer),
        )
    )
    advanced_widgets.append(
        Select(
            id="visual_region_source",
            label="Visual Region Source",
            values=_VISUAL_REGION_SOURCE_VALUES,
            initial_value=visual_region_source_current,
            description="Region secilince hangi proposal kaynaginin kullanilacagini belirler.",
            disabled=not region_mode_active,
        )
    )
    advanced_widgets.append(
        Select(
            id="visual_detector_backend",
            label="Visual Detector Backend",
            values=_VISUAL_DETECTOR_BACKEND_VALUES,
            initial_value=visual_detector_backend_current,
            description="Secili detector zinciri: docai online, docling local, sidecar harici bbox JSON.",
            disabled=not detector_mode_active,
        )
    )
    advanced_widgets.append(
        Select(
            id="table_structure_backend",
            label="Table Structure Backend",
            values=_TABLE_STRUCTURE_BACKEND_VALUES,
            initial_value=table_backend_current,
            description="Table stage acikken hangi backend'in kullanilacagini belirler.",
            disabled=not table_active,
        )
    )
    advanced_widgets.append(
        Select(
            id="vlm_provider",
            label="VLM Provider",
            values=_VLM_PROVIDER_VALUES,
            initial_value=vlm_provider_current,
            disabled=not vlm_active,
        )
    )
    advanced_widgets.append(
        Slider(
            id="vlm_max_pages",
            label="VLM Max Pages",
            initial=float(vlm_pages_current),
            min=0,
            max=_VLM_MAX_PAGES_LIMIT,
            step=1,
            disabled=not vlm_active,
        )
    )
    advanced_widgets.extend(
        [
            TextInput(
                id="active_pipeline_summary",
                label="Active Pipeline Summary",
                initial=dependency_summary,
                multiline=True,
                description="Ayarlarin birbirini nasil etkiledigini gosteren ozet. Bu alan bilgilendirme amaclidir.",
                disabled=True,
            ),
            TextInput(
                id="fallback_explanation",
                label="Why Fallback Happens",
                initial=fallback_summary,
                multiline=True,
                description="Secili kombinasyonda hangi kosul fallback veya disable davranisi dogurur, onu aciklar.",
                disabled=True,
            ),
        ]
    )
    return [
        Tab(id="settings_basic", label="Basic", inputs=basic_widgets),
        Tab(id="settings_advanced", label="Advanced", inputs=advanced_widgets),
    ]


def _apply_runtime_overrides_to_pipeline(pipeline: RAGPipeline) -> None:
    overrides = _runtime_overrides()
    if not overrides:
        return

    settings = _get_cached_settings()
    ollama_cfg = pipeline.ollama_config or OllamaConfig(
        base_url=settings.ollama_base_url,
        llm_model=settings.ollama_llm_model,
        vlm_model=settings.ollama_vlm_model,
        timeout=settings.ollama_timeout,
    )

    device = _sanitize_select_value(
        overrides.get("embedding_device", pipeline.embedding_device),
        _EMBEDDING_DEVICE_VALUES,
        pipeline.embedding_device,
    )
    model_choice = str(overrides.get("embedding_model_choice", pipeline.embedding_model)).strip()
    model_resolved = _resolve_embedding_model_choice(model_choice, device) or pipeline.embedding_model

    vlm_cfg = pipeline.vlm_config
    vlm_mode_default = vlm_cfg.mode if vlm_cfg else "auto"
    vlm_provider_default = vlm_cfg.provider if vlm_cfg else "gemini"
    vlm_pages_default = vlm_cfg.max_pages if vlm_cfg else _VLM_MAX_PAGES_DEFAULT
    requested_llm_provider = _sanitize_select_value(
        overrides.get("llm_provider"),
        _LLM_PROVIDER_VALUES,
        getattr(pipeline, "llm_provider", "gemini"),
    )
    resolved_llm_provider, _ = _resolve_llm_provider_choice(requested_llm_provider, settings, ollama_cfg)
    generation_model_choice = str(
        overrides.get("generation_model_choice", _llm_model_for_provider(pipeline, requested_llm_provider))
    ).strip()
    generation_model_choice = _normalize_generation_model_choice(
        pipeline,
        settings,
        resolved_llm_provider,
        generation_model_choice,
    )
    gemini_model_next = None
    openai_model_next = None
    local_llm_model_next = None
    if resolved_llm_provider == "gemini" and generation_model_choice:
        gemini_model_next = generation_model_choice
    elif resolved_llm_provider == "openai" and generation_model_choice:
        openai_model_next = generation_model_choice
    elif resolved_llm_provider == "local" and generation_model_choice:
        local_llm_model_next = generation_model_choice

    pipeline.reconfigure_runtime(
        ocr_enabled=_toggle_to_bool(
            overrides.get("ocr_enabled"),
            getattr(pipeline.ocr_config, "enabled", True),
        ),
        ocr_backend=_sanitize_select_value(
            overrides.get("ocr_backend"),
            _OCR_BACKEND_VALUES,
            getattr(pipeline.ocr_config, "backend", "docai"),
        ),
        processing_mode=_sanitize_select_value(
            overrides.get("processing_mode"),
            _PROCESSING_MODE_VALUES,
            getattr(pipeline, "processing_mode", "classic"),
        ),
        multimodal_answer_mode=_sanitize_select_value(
            overrides.get("multimodal_answer_mode"),
            _MULTIMODAL_ANSWER_MODE_VALUES,
            getattr(pipeline, "multimodal_answer_mode", "auto"),
        ),
        visual_chunk_level=_sanitize_select_value(
            overrides.get("visual_chunk_level"),
            _VISUAL_CHUNK_LEVEL_VALUES,
            getattr(pipeline, "visual_chunk_level", "page"),
        ),
        visual_region_source=_sanitize_select_value(
            overrides.get("visual_region_source"),
            _VISUAL_REGION_SOURCE_VALUES,
            getattr(pipeline, "visual_region_source", "heuristic"),
        ),
        visual_detector_backend=_sanitize_select_value(
            overrides.get("visual_detector_backend"),
            _VISUAL_DETECTOR_BACKEND_VALUES,
            getattr(pipeline, "visual_detector_backend", "none"),
        ),
        table_structure_enabled=overrides.get("table_structure_enabled", "off") in ("on", "smart"),
        table_structure_smart=overrides.get("table_structure_enabled") == "smart",
        table_structure_backend=_sanitize_select_value(
            overrides.get("table_structure_backend"),
            _TABLE_STRUCTURE_BACKEND_VALUES,
            getattr(getattr(pipeline, "table_structure_config", None), "backend", "auto"),
        ),
        llm_provider=resolved_llm_provider,
        gemini_model=gemini_model_next,
        openai_model=openai_model_next,
        local_llm_model=local_llm_model_next,
        embedding_model=model_resolved,
        embedding_device=device,
        vlm_mode=_sanitize_select_value(overrides.get("vlm_mode"), _VLM_MODE_VALUES, vlm_mode_default),
        vlm_provider=_sanitize_select_value(overrides.get("vlm_provider"), _VLM_PROVIDER_VALUES, vlm_provider_default),
        vlm_max_pages=_clamp_int(overrides.get("vlm_max_pages"), vlm_pages_default, 0, _VLM_MAX_PAGES_LIMIT),
        pdf_text_backend=_sanitize_select_value(
            overrides.get("pdf_text_backend"),
            _PDF_TEXT_BACKEND_VALUES,
            getattr(pipeline, "pdf_text_backend", "pymupdf"),
        ),
    )


def _active_llm_model(pipeline: RAGPipeline | None) -> str:
    if pipeline is None:
        return "-"
    provider = (pipeline.llm_provider or "gemini").strip().lower()
    if provider == "openai":
        return (pipeline.openai_model or "").strip() or "-"
    if provider == "local":
        if pipeline.ollama_config:
            return (pipeline.ollama_config.llm_model or "").strip() or "-"
        return "-"
    if provider == "none":
        return "extractive"
    return (pipeline.gemini_model or "").strip() or "-"


def _apply_chat_profile_to_pipeline(pipeline: RAGPipeline) -> str | None:
    """
    Apply selected Chainlit chat profile to runtime LLM provider/model.
    Returns an optional user-facing warning.
    """
    settings = _get_cached_settings()
    profile_name = (cl.user_session.get("chat_profile") or "").strip()
    provider = _CHAT_PROFILE_TO_PROVIDER.get(profile_name)
    if not provider:
        return None

    # Keep provider-specific models synced with env defaults.
    pipeline.gemini_model = settings.gemini_model
    pipeline.openai_model = settings.openai_model
    pipeline.ollama_config = OllamaConfig(
        base_url=settings.ollama_base_url,
        llm_model=settings.ollama_llm_model,
        vlm_model=settings.ollama_vlm_model,
        timeout=settings.ollama_timeout,
    )

    resolved_provider, warning = _resolve_llm_provider_choice(provider, settings, pipeline.ollama_config)
    pipeline.llm_provider = resolved_provider
    return warning


async def _sync_profile_to_pipeline(pipeline: RAGPipeline) -> None:
    warning = _apply_chat_profile_to_pipeline(pipeline)
    _apply_runtime_overrides_to_pipeline(pipeline)
    previous = cl.user_session.get(_PROFILE_WARNING_KEY)
    if warning and warning != previous:
        await cl.Message(content=warning).send()
    cl.user_session.set(_PROFILE_WARNING_KEY, warning or "")


def _get_chat_history() -> list[str]:
    raw = cl.user_session.get(_CHAT_HISTORY_KEY)
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _append_chat_history_user_message(message: str) -> None:
    text = _shorten_for_sidebar(message)
    if not text:
        return
    items = _get_chat_history()
    if not items or items[-1] != text:
        items.append(text)
    if len(items) > _CHAT_HISTORY_MAX:
        items = items[-_CHAT_HISTORY_MAX:]
    cl.user_session.set(_CHAT_HISTORY_KEY, items)


def _looks_like_doc_switch(query: str, pipeline: RAGPipeline) -> bool:
    return _looks_like_doc_switch_base(
        query,
        has_documents=bool(pipeline and pipeline.has_documents),
        document_names=(pipeline.list_documents() if pipeline else []),
    )


def _get_pipeline() -> RAGPipeline:
    """Get or lazily create the pipeline stored in the user session."""
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        settings = _get_cached_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Ollama config (only used when LLM_PROVIDER=local or VLM_PROVIDER=local)
        ollama_cfg = OllamaConfig(
            base_url=settings.ollama_base_url,
            llm_model=settings.ollama_llm_model,
            vlm_model=settings.ollama_vlm_model,
            timeout=settings.ollama_timeout,
        )

        vlm_provider = getattr(settings, "vlm_provider", "gemini")

        pipeline = RAGPipeline(
            embedding_model=settings.embedding_model,
            chroma_dir=settings.chroma_dir,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            gemini_fallback_model=getattr(settings, "gemini_fallback_model", ""),
            ocr_config=OCRConfig(
                enabled=getattr(settings, "ocr_enabled", True),
                backend=getattr(settings, "ocr_backend", "docai"),
                lang=getattr(settings, "ocr_lang", "tur+eng"),
                device=getattr(settings, "ocr_device", "auto"),
                paddle_ocr_version=getattr(settings, "paddle_ocr_version", "PP-OCRv5"),
                tesseract_cmd=settings.tesseract_cmd,
                tessdata_prefix=settings.tessdata_prefix,
                tesseract_config=getattr(settings, "tesseract_config", None),
                docai_project_id=getattr(settings, "docai_project_id", ""),
                docai_location=getattr(settings, "docai_location", "us"),
                docai_processor_id=getattr(settings, "docai_ocr_processor_id", ""),
                docai_processor_version=getattr(settings, "docai_ocr_processor_version", ""),
                docai_timeout_seconds=getattr(settings, "docai_timeout_seconds", 120),
            ),
            embedding_device=getattr(settings, "embedding_device", "auto"),
            embedding_dimension=getattr(settings, "embedding_dimension", 3072),
            processing_mode=getattr(settings, "processing_mode", "classic"),
            multimodal_answer_mode=getattr(settings, "multimodal_answer_mode", "auto"),
            section_fetch_max_depth=getattr(settings, "section_fetch_max_depth", 2),
            visual_chunk_level=getattr(settings, "visual_chunk_level", "page"),
            visual_region_source=getattr(settings, "visual_region_source", "heuristic"),
            visual_detector_backend=getattr(settings, "visual_detector_backend", "none"),
            visual_detector_dir=getattr(settings, "visual_detector_dir", None),
            docai_project_id=getattr(settings, "docai_project_id", ""),
            docai_location=getattr(settings, "docai_location", "us"),
            docai_layout_processor_id=getattr(settings, "docai_layout_processor_id", ""),
            docai_layout_processor_version=getattr(settings, "docai_layout_processor_version", "pretrained-layout-parser-v1.6-pro-2025-12-01"),
            docai_timeout_seconds=getattr(settings, "docai_timeout_seconds", 120),
            docling_python_bin=getattr(settings, "docling_python_bin", ""),
            docling_layout_model=getattr(settings, "docling_layout_model", "docling-layout-heron-101"),
            docling_artifacts_path=getattr(settings, "docling_artifacts_path", None),
            docling_device=getattr(settings, "docling_device", "auto"),
            pdf_text_backend=getattr(settings, "pdf_text_backend", "pymupdf"),
            table_structure_config=TableStructureConfig(
                enabled=getattr(settings, "table_structure_enabled", False),
                backend=getattr(settings, "table_structure_backend", "auto"),
                min_confidence=float(getattr(settings, "table_structure_min_confidence", 0.55) or 0.55),
                gemini_api_key=settings.gemini_api_key,
                gemini_model=getattr(settings, "table_structure_gemini_model", settings.gemini_model),
                docai_project_id=getattr(settings, "docai_project_id", ""),
                docai_location=getattr(settings, "docai_location", "us"),
                docai_processor_id=getattr(settings, "docai_table_processor_id", ""),
                docai_processor_version=getattr(settings, "docai_table_processor_version", ""),
                docai_timeout_seconds=getattr(settings, "docai_timeout_seconds", 120),
            ),
            vlm_config=VLMConfig(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                mode=settings.vlm_mode,
                max_pages=settings.vlm_max_pages,
                provider=vlm_provider,
                ollama_base_url=settings.ollama_base_url,
                ollama_vlm_model=settings.ollama_vlm_model,
                ollama_timeout=settings.ollama_timeout,
            ),
            llm_provider=settings.llm_provider,
            ollama_config=ollama_cfg,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            rerank_blend_weight=getattr(settings, "rerank_blend_weight", 0.6),
            relevance_min_score_ratio=getattr(settings, "relevance_min_score_ratio", 0.25),
            relevance_min_keep=getattr(settings, "relevance_min_keep", 3),
            grounding_min_avg_score=getattr(settings, "grounding_min_avg_score", 0.15),
            multi_section_max=getattr(settings, "multi_section_max", 3),
            context_max_tokens=getattr(settings, "context_max_tokens", 100000),
            query_expansion_enabled=getattr(settings, "query_expansion_enabled", False),
        )
        _apply_runtime_overrides_to_pipeline(pipeline)
        cl.user_session.set("pipeline", pipeline)
    return pipeline


async def _process_uploaded_file(file_path: str, file_name: str) -> str:
    """
    Ingest a single uploaded file into the pipeline.
    Returns a status message.
    """
    pipeline = _get_pipeline()
    path = Path(file_path)

    try:
        state = pipeline.add_document(path, display_name=file_name)
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.restored_from_cache:
            lines.append("- Bu belge kalici cache'den geri yuklendi; OCR/VLM/embedding tekrar calismadi.")
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


# ── Lifecycle hooks ──────────────────────────────────────────────────────────


@cl.set_chat_profiles
async def set_chat_profiles(_current_user, _language):
    default_profile = _default_profile_name()
    return [
        cl.ChatProfile(
            name="Gemini",
            display_name="Gemini",
            markdown_description="Google Gemini (RAG + chat).",
            default=(default_profile == "Gemini"),
        ),
        cl.ChatProfile(
            name="OpenAI",
            display_name="OpenAI",
            markdown_description="OpenAI-compatible Chat Completions ile RAG + chat.",
            default=(default_profile == "OpenAI"),
        ),
        cl.ChatProfile(
            name="Local",
            display_name="Local",
            markdown_description="Ollama local model (RAG + chat).",
            default=(default_profile == "Local"),
        ),
        cl.ChatProfile(
            name="Extractive",
            display_name="Extractive",
            markdown_description="LLM yok, sadece extractive cevap.",
            default=(default_profile == "Extractive"),
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", "doc")
    cl.user_session.set(_CHAT_HISTORY_KEY, [])
    pipeline = _get_pipeline()
    profile_warning = _apply_chat_profile_to_pipeline(pipeline)
    _apply_runtime_overrides_to_pipeline(pipeline)
    native_history_note = _native_history_status_note()
    if native_history_note:
        print(f"[ui] {native_history_note}", flush=True)
    if _NATIVE_HISTORY_READY:
        try:
            session_user = cl_context.session.user
            identifier = str(getattr(session_user, "identifier", "") or "").strip()
            conninfo = _native_history_conninfo()
            if conninfo and identifier:
                await _native_history_backfill_user_binding_async(conninfo, identifier)
        except Exception:
            pass

    # Track active sessions for optional auto-exit behavior.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS += 1
        await _cancel_exit_task()
        if _auto_exit_enabled():
            print(
                f"[auto-exit] enabled, active_sessions={_ACTIVE_CHAT_SESSIONS}",
                flush=True,
            )

    # Intentionally no auto welcome message to avoid initial layout jump in UI.
    if profile_warning:
        await cl.Message(content=profile_warning).send()
    cl.user_session.set(_PROFILE_WARNING_KEY, profile_warning or "")
    await cl.ChatSettings(_settings_widgets(pipeline)).send()
    await _update_documents_sidebar(pipeline)
    await _send_settings_sidebar_state(pipeline)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    thread_id = str(thread.get("id") or "").strip()
    metadata = thread.get("metadata") or {}
    if _NATIVE_HISTORY_READY:
        try:
            session_user = cl_context.session.user
            identifier = str(getattr(session_user, "identifier", "") or "").strip()
            conninfo = _native_history_conninfo()
            if conninfo and identifier:
                await _native_history_backfill_user_binding_async(conninfo, identifier)
        except Exception:
            pass
    pipeline = _restore_thread_state(thread_id, metadata=metadata)
    if pipeline is not None:
        await _sync_profile_to_pipeline(pipeline)
        _apply_runtime_overrides_to_pipeline(pipeline)
    await _update_documents_sidebar(pipeline)
    await _send_settings_sidebar_state(pipeline)

async def _apply_runtime_settings(
    pipeline: RAGPipeline,
    values: dict,
    *,
    announce: bool = True,
    resend_chat_settings: bool = True,
    sync_sidebar: bool = True,
):
    current_overrides = _runtime_overrides()
    current_vlm = pipeline.vlm_config
    current_pages = current_vlm.max_pages if current_vlm else _VLM_MAX_PAGES_DEFAULT
    current_ocr = pipeline.ocr_config
    current_table = pipeline.table_structure_config
    selected_device = _sanitize_select_value(
        values.get("embedding_device"),
        _EMBEDDING_DEVICE_VALUES,
        pipeline.embedding_device,
    )
    selected_processing_mode = _sanitize_select_value(
        values.get("processing_mode"),
        _PROCESSING_MODE_VALUES,
        getattr(pipeline, "processing_mode", "classic"),
    )
    selected_multimodal_answer_mode = _sanitize_select_value(
        values.get("multimodal_answer_mode"),
        _MULTIMODAL_ANSWER_MODE_VALUES,
        getattr(pipeline, "multimodal_answer_mode", "auto"),
    )
    selected_visual_chunk_level = _sanitize_select_value(
        values.get("visual_chunk_level"),
        _VISUAL_CHUNK_LEVEL_VALUES,
        getattr(pipeline, "visual_chunk_level", "page"),
    )
    selected_visual_region_source = _sanitize_select_value(
        values.get("visual_region_source"),
        _VISUAL_REGION_SOURCE_VALUES,
        getattr(pipeline, "visual_region_source", "heuristic"),
    )
    selected_visual_detector_backend = _sanitize_select_value(
        values.get("visual_detector_backend"),
        _VISUAL_DETECTOR_BACKEND_VALUES,
        getattr(pipeline, "visual_detector_backend", "none"),
    )
    selected_pdf_text_backend = _sanitize_select_value(
        values.get("pdf_text_backend"),
        _PDF_TEXT_BACKEND_VALUES,
        getattr(pipeline, "pdf_text_backend", "pymupdf"),
    )
    selected_ocr_enabled = _normalize_toggle_value(
        values.get("ocr_enabled"),
        "on" if getattr(current_ocr, "enabled", True) else "off",
    )
    selected_ocr_backend = _sanitize_select_value(
        values.get("ocr_backend"),
        _OCR_BACKEND_VALUES,
        getattr(current_ocr, "backend", "docai"),
    )
    selected_table_enabled = _sanitize_select_value(
        values.get("table_structure_enabled"),
        _TABLE_STRUCTURE_MODE_VALUES,
        ("smart" if getattr(current_table, "smart", False) else "on") if getattr(current_table, "enabled", False) else "off",
    )
    selected_table_backend = _sanitize_select_value(
        values.get("table_structure_backend"),
        _TABLE_STRUCTURE_BACKEND_VALUES,
        getattr(current_table, "backend", "auto"),
    )
    selected_llm_provider = _sanitize_select_value(
        values.get("llm_provider"),
        _LLM_PROVIDER_VALUES,
        getattr(pipeline, "llm_provider", "gemini"),
    )
    selected_generation_model = str(values.get("generation_model") or "").strip()
    selected_model_choice = str(values.get("embedding_model") or pipeline.embedding_model).strip()
    if not selected_model_choice:
        selected_model_choice = pipeline.embedding_model
    resolved_model = _resolve_embedding_model_choice(selected_model_choice, selected_device) or pipeline.embedding_model

    selected_vlm_mode = _sanitize_select_value(
        values.get("vlm_mode"),
        _VLM_MODE_VALUES,
        current_vlm.mode if current_vlm else "auto",
    )
    selected_vlm_provider = _sanitize_select_value(
        values.get("vlm_provider"),
        _VLM_PROVIDER_VALUES,
        current_vlm.provider if current_vlm else "gemini",
    )
    selected_vlm_pages = _clamp_int(
        values.get("vlm_max_pages"),
        current_pages,
        0,
        _VLM_MAX_PAGES_LIMIT,
    )

    settings = _get_cached_settings()
    ollama_cfg = pipeline.ollama_config or OllamaConfig(
        base_url=settings.ollama_base_url,
        llm_model=settings.ollama_llm_model,
        vlm_model=settings.ollama_vlm_model,
        timeout=settings.ollama_timeout,
    )
    previous_preset = _sanitize_select_value(
        current_overrides.get("runtime_preset"),
        _RUNTIME_PRESET_VALUES,
        "custom",
    )
    selected_runtime_preset = _sanitize_select_value(
        values.get("runtime_preset"),
        _RUNTIME_PRESET_VALUES,
        previous_preset,
    )
    if selected_runtime_preset != "custom" and selected_runtime_preset != previous_preset:
        preset_values = _preset_defaults(pipeline, settings, selected_runtime_preset)
        if preset_values:
            selected_ocr_enabled = str(preset_values.get("ocr_enabled", selected_ocr_enabled))
            selected_ocr_backend = str(preset_values.get("ocr_backend", selected_ocr_backend))
            selected_processing_mode = str(preset_values.get("processing_mode", selected_processing_mode))
            selected_multimodal_answer_mode = str(preset_values.get("multimodal_answer_mode", selected_multimodal_answer_mode))
            selected_visual_chunk_level = str(preset_values.get("visual_chunk_level", selected_visual_chunk_level))
            selected_visual_region_source = str(preset_values.get("visual_region_source", selected_visual_region_source))
            selected_visual_detector_backend = str(preset_values.get("visual_detector_backend", selected_visual_detector_backend))
            selected_table_enabled = str(preset_values.get("table_structure_enabled", selected_table_enabled))
            selected_table_backend = str(preset_values.get("table_structure_backend", selected_table_backend))
            selected_llm_provider = str(preset_values.get("llm_provider", selected_llm_provider))
            selected_generation_model = str(preset_values.get("generation_model_choice", selected_generation_model))
            selected_model_choice = str(preset_values.get("embedding_model_choice", selected_model_choice))
            selected_device = str(preset_values.get("embedding_device", selected_device))
            selected_vlm_mode = str(preset_values.get("vlm_mode", selected_vlm_mode))
            selected_vlm_provider = str(preset_values.get("vlm_provider", selected_vlm_provider))
            selected_vlm_pages = _clamp_int(preset_values.get("vlm_max_pages"), selected_vlm_pages, 0, _VLM_MAX_PAGES_LIMIT)
            resolved_model = _resolve_embedding_model_choice(selected_model_choice, selected_device) or pipeline.embedding_model
    current_effective = _effective_settings_values(pipeline)
    dependency_flags = _runtime_dependency_flags(
        processing_mode=selected_processing_mode,
        visual_chunk_level=selected_visual_chunk_level,
        visual_region_source=selected_visual_region_source,
        ocr_enabled=selected_ocr_enabled,
        table_enabled=selected_table_enabled,
        llm_provider=selected_llm_provider,
        embedding_model_choice=selected_model_choice,
        vlm_mode=selected_vlm_mode,
    )
    if dependency_flags["generation_model"]:
        selected_generation_model = str(current_effective.get("generation_model_choice", "") or "")
    if dependency_flags["ocr_backend"]:
        selected_ocr_backend = str(current_effective.get("ocr_backend", selected_ocr_backend) or selected_ocr_backend)
    if dependency_flags["visual_chunk_level"]:
        selected_visual_chunk_level = str(current_effective.get("visual_chunk_level", selected_visual_chunk_level) or selected_visual_chunk_level)
    if dependency_flags["table_structure_enabled"]:
        selected_table_enabled = str(current_effective.get("table_structure_enabled", selected_table_enabled) or selected_table_enabled)
    if dependency_flags["multimodal_answer_mode"]:
        selected_multimodal_answer_mode = str(current_effective.get("multimodal_answer_mode", selected_multimodal_answer_mode) or selected_multimodal_answer_mode)
    if dependency_flags["visual_region_source"]:
        selected_visual_region_source = str(current_effective.get("visual_region_source", selected_visual_region_source) or selected_visual_region_source)
    if dependency_flags["visual_detector_backend"]:
        selected_visual_detector_backend = str(current_effective.get("visual_detector_backend", selected_visual_detector_backend) or selected_visual_detector_backend)
    if dependency_flags["table_structure_backend"]:
        selected_table_backend = str(current_effective.get("table_structure_backend", selected_table_backend) or selected_table_backend)
    if dependency_flags["vlm_provider"]:
        selected_vlm_provider = str(current_effective.get("vlm_provider", selected_vlm_provider) or selected_vlm_provider)
    if dependency_flags["vlm_max_pages"]:
        selected_vlm_pages = int(current_effective.get("vlm_max_pages", selected_vlm_pages) or selected_vlm_pages)
    if dependency_flags["embedding_device"]:
        selected_device = str(current_effective.get("embedding_device", selected_device) or selected_device)
        resolved_model = _resolve_embedding_model_choice(selected_model_choice, selected_device) or pipeline.embedding_model
    resolved_vlm_provider, vlm_warning = _resolve_vlm_provider_choice(
        selected_vlm_provider,
        settings,
        ollama_cfg,
    )
    resolved_llm_provider, llm_warning = _resolve_llm_provider_choice(
        selected_llm_provider,
        settings,
        ollama_cfg,
    )
    selected_generation_model = _normalize_generation_model_choice(
        pipeline,
        settings,
        resolved_llm_provider,
        selected_generation_model,
    )
    gemini_model_next = None
    openai_model_next = None
    local_llm_model_next = None
    if resolved_llm_provider == "gemini" and selected_generation_model:
        gemini_model_next = selected_generation_model
    elif resolved_llm_provider == "openai" and selected_generation_model:
        openai_model_next = selected_generation_model
    elif resolved_llm_provider == "local" and selected_generation_model:
        local_llm_model_next = selected_generation_model
    effective_after_apply = {}

    result = pipeline.reconfigure_runtime(
        ocr_enabled=_toggle_to_bool(selected_ocr_enabled, getattr(current_ocr, "enabled", True)),
        ocr_backend=selected_ocr_backend,
        processing_mode=selected_processing_mode,
        multimodal_answer_mode=selected_multimodal_answer_mode,
        visual_chunk_level=selected_visual_chunk_level,
        visual_region_source=selected_visual_region_source,
        visual_detector_backend=selected_visual_detector_backend,
        table_structure_enabled=selected_table_enabled in ("on", "smart"),
        table_structure_smart=selected_table_enabled == "smart",
        table_structure_backend=selected_table_backend,
        llm_provider=resolved_llm_provider,
        gemini_model=gemini_model_next,
        openai_model=openai_model_next,
        local_llm_model=local_llm_model_next,
        embedding_model=resolved_model,
        embedding_device=selected_device,
        vlm_mode=selected_vlm_mode,
        vlm_provider=resolved_vlm_provider,
        vlm_max_pages=selected_vlm_pages,
        pdf_text_backend=selected_pdf_text_backend,
    )
    effective_after_apply = _effective_settings_values(pipeline)
    stored_runtime_preset = selected_runtime_preset
    if stored_runtime_preset != "custom" and not _matches_preset(effective_after_apply, stored_runtime_preset, pipeline, settings):
        stored_runtime_preset = "custom"
    cl.user_session.set(
        _RUNTIME_OVERRIDES_KEY,
        {
            "runtime_preset": stored_runtime_preset,
            "ocr_enabled": selected_ocr_enabled,
            "ocr_backend": selected_ocr_backend,
            "embedding_model_choice": selected_model_choice,
            "embedding_device": selected_device,
            "processing_mode": selected_processing_mode,
            "multimodal_answer_mode": selected_multimodal_answer_mode,
            "visual_chunk_level": selected_visual_chunk_level,
            "visual_region_source": selected_visual_region_source,
            "visual_detector_backend": selected_visual_detector_backend,
            "table_structure_enabled": selected_table_enabled,
            "table_structure_backend": selected_table_backend,
            "llm_provider": resolved_llm_provider,
            "generation_model_choice": selected_generation_model,
            "vlm_mode": selected_vlm_mode,
            "vlm_provider": resolved_vlm_provider,
            "vlm_max_pages": selected_vlm_pages,
            "pdf_text_backend": selected_pdf_text_backend,
        },
    )
    cl.user_session.set("pipeline", pipeline)
    _thread_pipeline_set(_active_thread_id(), pipeline)

    if (
        result.get("ocr_changed")
        or result.get("llm_changed")
        or result.get("embedding_changed")
        or result.get("vlm_changed")
        or result.get("processing_mode_changed")
        or result.get("multimodal_answer_mode_changed")
        or result.get("visual_chunk_level_changed")
        or result.get("visual_region_source_changed")
        or result.get("visual_detector_backend_changed")
        or result.get("table_structure_changed")
    ):
        messages = [
            "**Runtime ayarlari guncellendi**",
            f"- OCR: `{selected_ocr_enabled}` | backend=`{getattr(pipeline.ocr_config, 'backend', 'docai')}`",
            f"- Preset: `{stored_runtime_preset}`",
            f"- Processing: `{getattr(pipeline, 'processing_mode', 'classic')}`",
            f"- Multimodal Answer Gen: `{getattr(pipeline, 'multimodal_answer_mode', 'auto')}`",
            f"- Visual Chunk Level: `{getattr(pipeline, 'visual_chunk_level', 'page')}`",
            f"- Visual Region Source: `{getattr(pipeline, 'visual_region_source', 'heuristic')}`",
            f"- Visual Detector Backend: `{getattr(pipeline, 'visual_detector_backend', 'none')}`",
            f"- Table Structure: `{'on' if bool(getattr(getattr(pipeline, 'table_structure_config', None), 'enabled', False)) else 'off'}` | backend=`{getattr(getattr(pipeline, 'table_structure_config', None), 'backend', 'auto')}`",
            f"- LLM: `{getattr(pipeline, 'llm_provider', 'gemini')}` | model=`{_active_llm_model(pipeline)}`",
            f"- Embedding: `{pipeline.embedding_model}` | `{_embedding_runtime_label(pipeline.embedding_model, pipeline.embedding_device)}`",
            f"- VLM: `{resolved_vlm_provider}` | `{selected_vlm_mode}` | max_pages=`{selected_vlm_pages}`",
        ]
        if result.get("index_rebuilt"):
            messages.append("- Mevcut dokumanlar icin embedding indeksi yeniden olusturuldu.")
        elif result.get("embedding_changed"):
            messages.append("- Embedding ayari degisti; indeks yeni ayarlarla hazir.")
        if result.get("ocr_changed"):
            messages.append("- OCR ayarlari bir sonraki dosya yuklemelerinde uygulanir.")
        if result.get("llm_changed"):
            messages.append("- LLM provider/model ayari aninda uygulanir; yeni sorulari etkiler.")
        if result.get("vlm_changed"):
            messages.append("- VLM ayarlari bir sonraki dosya yuklemelerinde uygulanir.")
        if result.get("processing_mode_changed"):
            messages.append("- Processing mode bir sonraki belge yuklemelerinde uygulanir. Mevcut belgeleri yeniden yuklemeniz gerekir.")
        if result.get("multimodal_answer_mode_changed"):
            messages.append("- Multimodal answer generation ayari aninda uygulanir; reindex gerekmez.")
        if result.get("visual_chunk_level_changed"):
            messages.append("- Visual chunk level deneysel bir ayardir; mevcut belgeler icin yeniden yukleme gerekir.")
        if result.get("visual_region_source_changed"):
            messages.append("- Region source ayari bir sonraki belge yuklemelerinde uygulanir.")
        if result.get("visual_detector_backend_changed"):
            messages.append("- Visual detector backend bir sonraki belge yuklemelerinde uygulanir.")
        if result.get("table_structure_changed"):
            messages.append("- Table structure ayarlari bir sonraki belge yuklemelerinde uygulanir.")
        if selected_runtime_preset != stored_runtime_preset and stored_runtime_preset == "custom":
            messages.append("- Preset secimi manuel override nedeniyle `custom` moduna gecti.")
        elif selected_runtime_preset != previous_preset:
            messages.append(f"- Preset `{selected_runtime_preset}` uygulandi.")
        if str(selected_model_choice).strip().lower().startswith("gemini-embedding-") and resolved_model != selected_model_choice:
            messages.append("- Not: `google-genai` paketi hazir olmadigi icin embedding local fallback ile calisiyor.")
        if selected_visual_region_source == "detector" and selected_visual_detector_backend == "none":
            messages.append("- Not: `Visual Region Source=detector` iken backend `none`; bu durumda heuristic fallback kullanilir.")
        if llm_warning:
            messages.append(f"- Not: {llm_warning}")
        if vlm_warning:
            messages.append(f"- Not: {vlm_warning}")
        if announce:
            await cl.Message(content="\n".join(messages)).send()

    if resend_chat_settings:
        await cl.ChatSettings(_settings_widgets(pipeline)).send()
    if sync_sidebar:
        await _send_settings_sidebar_state(pipeline)
    return result


@cl.on_settings_update
async def on_settings_update(values: dict):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()
    await _apply_runtime_settings(pipeline, values, announce=True, resend_chat_settings=True, sync_sidebar=True)


@cl.on_window_message
async def on_window_message(data):
    if not isinstance(data, dict):
        return
    if str(data.get("source") or "") != "docqa-settings-ui":
        return

    action = str(data.get("action") or "").strip().lower()
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()

    if action == "request_state":
        await _send_settings_sidebar_state(pipeline)
        return
    if action == "reset":
        await _send_settings_sidebar_state(pipeline)
        return
    if action == "apply":
        payload = data.get("payload")
        if isinstance(payload, dict):
            await _apply_runtime_settings(pipeline, payload, announce=True, resend_chat_settings=True, sync_sidebar=True)
    await _update_documents_sidebar(pipeline)


@cl.on_chat_end
async def on_chat_end() -> None:
    _drop_thread_cache(_active_thread_id())
    # Decrement active sessions and auto-exit if enabled.
    async with _EXIT_LOCK:
        global _ACTIVE_CHAT_SESSIONS
        _ACTIVE_CHAT_SESSIONS = max(0, _ACTIVE_CHAT_SESSIONS - 1)
        if _ACTIVE_CHAT_SESSIONS == 0:
            if _auto_exit_enabled():
                print(
                    f"[auto-exit] last client disconnected, exiting in {_auto_exit_grace_seconds()}s",
                    flush=True,
                )
            await _schedule_auto_exit_if_idle()


def _process_uploaded_file_sync(file_path: str, file_name: str, pipeline: RAGPipeline) -> str:
    """Sync wrapper for file processing (called via make_async)."""
    return _process_uploaded_file_sync_with_progress(file_path, file_name, pipeline, None)


def _process_uploaded_file_sync_with_progress(
    file_path: str,
    file_name: str,
    pipeline: RAGPipeline,
    progress_callback=None,
) -> str:
    """Sync file processing with optional progress callback."""
    path = Path(file_path)

    try:
        state = pipeline.add_document(
            path,
            display_name=file_name,
            progress_callback=progress_callback,
        )
        lines = [
            f"**{file_name}** basariyla yuklendi ve indekslendi.",
            f"- Sayfa sayisi: {state.page_count}",
            f"- Chunk sayisi: {len(state.chunks)}",
            f"- Toplam indekslenen chunk: {pipeline.total_chunks}",
        ]
        if state.restored_from_cache:
            lines.append("- Bu belge kalici cache'den geri yuklendi; OCR/VLM/embedding tekrar calismadi.")
        if state.warnings:
            lines.append(f"- Uyarilar: {'; '.join(state.warnings)}")
        return "\n".join(lines)
    except Exception as e:
        return f"**{file_name}** yuklenirken hata olustu: {e}"


def _extract_uploaded_file_info(elem) -> tuple[str | None, str]:
    """
    Be defensive across Chainlit versions: uploaded elements may come as
    Element objects or plain dicts.
    """
    path = getattr(elem, "path", None)
    name = getattr(elem, "name", None)
    if isinstance(elem, dict):
        path = path or elem.get("path")
        name = name or elem.get("name")
    return path, (name or "dosya")


async def _update_documents_sidebar(pipeline: RAGPipeline | None = None) -> None:
    """
    Close the Chainlit element sidebar; document context now lives in the
    reactive runtime settings panel to avoid right-rail overlap.
    """
    try:
        pipeline = _resolve_sidebar_pipeline(pipeline)
        _thread_state_persist(_active_thread_id(), pipeline)
        await cl.ElementSidebar.set_elements([], key=f"belge-durumu-hidden-{_next_sidebar_rev()}")
    except Exception as err:
        print(f"[ui] sidebar close failed: {err}", flush=True)
        return


def _render_upload_progress(
    file_name: str,
    steps: list[str],
    in_progress: str | None,
    final_state: str | None = None,
    summary: str | None = None,
) -> str:
    if final_state == "done":
        title = f"**{file_name}** - ✅ TAMAMLANDI"
    elif final_state == "error":
        title = f"**{file_name}** - ❌ HATA"
    else:
        title = f"**{file_name}** isleniyor..."
    lines = [title, ""]
    for s in steps:
        if s == in_progress:
            lines.append(f"- [..] {s}")
        else:
            lines.append(f"- [x] {s}")
    if not steps and in_progress:
        lines.append(f"- [..] {in_progress}")
    if summary:
        lines.extend(["", "---", summary])
    return "\n".join(lines)


async def _stream_text_response(text: str, chunk_size: int = 24) -> None:
    """
    UI-only streaming: emit an already-generated response in small chunks.
    Does not change retrieval/generation logic.
    """
    msg = cl.Message(content="")
    await msg.send()
    if not text:
        await msg.update()
        return
    for i in range(0, len(text), chunk_size):
        await msg.stream_token(text[i:i + chunk_size])
    await msg.update()


async def _run_chat_response(pipeline: RAGPipeline, query: str, chat_style: str) -> str:
    async with _chat_qa_semaphore():
        return await cl.make_async(pipeline.chat)(query, chat_style)


def _build_qa_response(result, mode: str) -> str:
    answer = result.answer
    return f"{answer}{_build_qa_debug_suffix(result, mode)}"


async def _send_standard_error(
    title: str,
    err: Exception | str,
    retry_payload: dict | None = None,
) -> None:
    actions = None
    if retry_payload:
        actions = [
            cl.Action(
                name="retry_last",
                payload=retry_payload,
                label="Tekrar dene",
                tooltip="Ayni istegi tekrar calistir",
                icon="refresh-cw",
            )
        ]
    await cl.Message(content=_format_standard_error(title, err), actions=actions).send()


async def _stream_doc_answer_live(
    pipeline: RAGPipeline,
    query: str,
    mode: str,
    thinking_msg: cl.Message | None = None,
):
    token_queue: SimpleQueue[str] = SimpleQueue()
    stream_msg: cl.Message | None = None

    def _on_token(token: str) -> None:
        token_queue.put(token)

    async with _doc_qa_semaphore():
        worker = asyncio.create_task(
            cl.make_async(pipeline.ask_stream)(query, _on_token)
        )

        streamed_chars = 0
        thinking_removed = False
        while not worker.done():
            while True:
                try:
                    token = token_queue.get_nowait()
                except Empty:
                    break
                if token:
                    if thinking_msg and not thinking_removed:
                        try:
                            await thinking_msg.remove()
                        except Exception:
                            pass
                        thinking_removed = True
                    if stream_msg is None:
                        stream_msg = cl.Message(content="")
                        await stream_msg.send()
                    streamed_chars += len(token)
                    await stream_msg.stream_token(token)
            await asyncio.sleep(0.03)

        while True:
            try:
                token = token_queue.get_nowait()
            except Empty:
                break
            if token:
                if thinking_msg and not thinking_removed:
                    try:
                        await thinking_msg.remove()
                    except Exception:
                        pass
                    thinking_removed = True
                if stream_msg is None:
                    stream_msg = cl.Message(content="")
                    await stream_msg.send()
                streamed_chars += len(token)
                await stream_msg.stream_token(token)

        result = await worker
    if thinking_msg and not thinking_removed:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        thinking_removed = True
    if stream_msg is None:
        stream_msg = cl.Message(content="")
        await stream_msg.send()
    if streamed_chars == 0 and result.answer:
        await stream_msg.stream_token(result.answer)
    await stream_msg.stream_token(_build_qa_debug_suffix(result, mode))
    await stream_msg.update()
    evidence_panel = _build_evidence_panel(result)
    if evidence_panel:
        await cl.Message(content=evidence_panel).send()
    return result


async def _process_uploaded_file_with_progress(file_path: str, file_name: str) -> str:
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        pipeline = _get_pipeline()

    progress_queue: SimpleQueue[str] = SimpleQueue()
    seen_steps: list[str] = []
    in_progress = "Dosya alindi, islem baslatiliyor..."

    def _on_progress(step: str) -> None:
        progress_queue.put(step)

    progress_msg = cl.Message(content=_render_upload_progress(file_name, seen_steps, in_progress))
    await progress_msg.send()

    async with _upload_work_semaphore():
        worker = asyncio.create_task(
            cl.make_async(_process_uploaded_file_sync_with_progress)(
                file_path,
                file_name,
                pipeline,
                _on_progress,
            )
        )

        while not worker.done():
            updated = False
            while True:
                try:
                    step = progress_queue.get_nowait()
                except Empty:
                    break
                if not seen_steps or seen_steps[-1] != step:
                    seen_steps.append(step)
                    in_progress = step
                    updated = True
            if updated:
                progress_msg.content = _render_upload_progress(file_name, seen_steps, in_progress)
                await progress_msg.update()
            await asyncio.sleep(0.25)

        while True:
            try:
                step = progress_queue.get_nowait()
            except Empty:
                break
            if not seen_steps or seen_steps[-1] != step:
                seen_steps.append(step)
                in_progress = step

        status = await worker
    status_lower = status.lower()
    is_error = "hata olustu" in status_lower or status_lower.startswith("hata:")
    final_state = "error" if is_error else "done"
    progress_msg.content = _render_upload_progress(
        file_name,
        seen_steps,
        None,
        final_state=final_state,
        summary=status,
    )
    await progress_msg.update()
    cl.user_session.set("pipeline", pipeline)
    return status


@cl.action_callback("retry_last")
async def on_retry_last(action: cl.Action):
    payload = action.payload or {}
    query = (payload.get("query") or "").strip()
    kind = (payload.get("kind") or "ask").strip().lower()
    mode = payload.get("mode") or (cl.user_session.get("mode") or "doc")
    chat_style = payload.get("chat_style") or _smalltalk_style(query)

    if not query:
        await cl.Message(content="Tekrar deneme icin sorgu bulunamadi. Lutfen tekrar yaz.").send()
        return

    try:
        await action.remove()
    except Exception:
        pass

    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if not pipeline:
        pipeline = _get_pipeline()
    await _sync_profile_to_pipeline(pipeline)

    thinking_msg = cl.Message(content="Tekrar deneniyor...")
    await thinking_msg.send()
    try:
        if kind == "chat":
            answer = await _run_chat_response(pipeline, query, chat_style)
            await thinking_msg.remove()
            await _stream_text_response(answer)
            await _update_documents_sidebar(pipeline)
            return

        await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
        await _update_documents_sidebar(pipeline)
    except Exception as e:
        await thinking_msg.remove()
        await _send_standard_error(
            "Tekrar deneme sirasinda hata",
            e,
            retry_payload=payload,
        )


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    mode: str = cl.user_session.get("mode") or "doc"
    raw_query = (message.content or "").strip()
    hinted_thread_id, _ = _extract_thread_marker(raw_query) if raw_query else (None, "")
    if hinted_thread_id:
        cl.user_session.set("ui_thread_id", hinted_thread_id)
    active_ui_thread_id = (cl.user_session.get("ui_thread_id") or "").strip() or None
    if active_ui_thread_id:
        pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
        if pipeline_for_thread is not None and pipeline_for_thread is not pipeline:
            pipeline = pipeline_for_thread
            cl.user_session.set("pipeline", pipeline)

    # Check for file attachments in the message
    if message.elements:
        if not pipeline:
            pipeline = _get_pipeline()
            cl.user_session.set("pipeline", pipeline)
        _thread_pipeline_set(active_ui_thread_id, pipeline)
        handled_any = False
        for elem in message.elements:
            file_path, file_name = _extract_uploaded_file_info(elem)
            if file_path:
                handled_any = True
                await _process_uploaded_file_with_progress(file_path, file_name)
                pipeline = cl.user_session.get("pipeline")
                _thread_pipeline_set(active_ui_thread_id, pipeline)
                await _update_documents_sidebar(pipeline)
        if not handled_any:
            await cl.Message(
                content=(
                    "Yuklenen dosya algılandı ancak dosya yolu okunamadı. "
                    "Lütfen dosyayı tekrar yükleyip bir kısa mesajla birlikte gönderin."
                )
            ).send()

    if not raw_query:
        return
    thread_id, query = _extract_thread_marker(raw_query)
    if thread_id:
        cl.user_session.set("ui_thread_id", thread_id)
    active_ui_thread_id = cl.user_session.get("ui_thread_id")
    if active_ui_thread_id:
        pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
        if pipeline_for_thread is not None and pipeline_for_thread is not pipeline:
            pipeline = pipeline_for_thread
            cl.user_session.set("pipeline", pipeline)
    open_m = _OPEN_THREAD_CMD_RE.match(query)
    if open_m:
        forced_thread_id = (open_m.group(1) or "").strip()
        if forced_thread_id:
            cl.user_session.set("ui_thread_id", forced_thread_id)
            active_ui_thread_id = forced_thread_id
        if active_ui_thread_id:
            pipeline = _restore_thread_state(active_ui_thread_id, pipeline=pipeline)
            pipeline_for_thread = _thread_pipeline_get(active_ui_thread_id)
            if pipeline_for_thread is not None:
                pipeline = pipeline_for_thread
                cl.user_session.set("pipeline", pipeline)
        entries = _thread_memory_load((active_ui_thread_id or "").strip())
        if entries:
            for item in entries:
                role = item.get("role", "assistant")
                author = "Kullanici" if role == "user" else "Asistan"
                await cl.Message(content=item.get("content", ""), author=author).send()
        else:
            await cl.Message(content="Bu sohbete ait kayit bulunamadi.").send()
        await _update_documents_sidebar(pipeline)
        return
    if not query:
        return
    chat_style = _smalltalk_style(query)

    # Natural-language mode switches (work in any mode).
    if _looks_like_chat_mode_request(query) or query.strip().lower() in ("/chat", "/sohbet"):
        cl.user_session.set("mode", "chat")
        await cl.Message(content="Sohbet moduna geçildi. Belge soruları için `/doc` yazabilirsin.").send()
        await _update_documents_sidebar(pipeline)
        return
    if _looks_like_doc_mode_request(query) or query.strip().lower() in ("/doc", "/belge"):
        cl.user_session.set("mode", "doc")
        if pipeline and pipeline.has_documents:
            await cl.Message(content="Belge moduna geçildi. Belge sorunu sorabilirsin. (Sohbet için `/chat` yazabilirsin.)").send()
        else:
            await cl.Message(content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)").send()
        await _update_documents_sidebar(pipeline)
        return

    if not query.startswith("/"):
        _append_chat_history_user_message(query)
        _thread_memory_add(active_ui_thread_id, "user", query)

    # Auto small-talk: answer conversational messages even in doc mode.
    if _looks_like_smalltalk(query):
        if not pipeline:
            pipeline = _get_pipeline()
        thinking_msg = cl.Message(content="Dusunuyorum...")
        await thinking_msg.send()
        try:
            answer = await _run_chat_response(pipeline, query, chat_style)
        except Exception as e:
            await thinking_msg.remove()
            await _send_standard_error(
                "Sohbet cevabi olusturulamadi",
                e,
                retry_payload={
                    "kind": "chat",
                    "query": query,
                    "chat_style": chat_style,
                    "mode": cl.user_session.get("mode") or mode,
                },
            )
            return
        await thinking_msg.remove()
        await _stream_text_response(answer)
        _thread_memory_add(active_ui_thread_id, "assistant", answer)
        await _update_documents_sidebar(pipeline)
        return

    # Commands (document-agnostic)
    qlow = query.lower()
    if qlow.startswith("/use "):
        if not pipeline:
            pipeline = _get_pipeline()
        _thread_pipeline_set(active_ui_thread_id, pipeline)
        name = query[5:].strip()
        ok = pipeline.set_active_document(name)
        if ok:
            resolved = pipeline.active_document_name or name
            await cl.Message(content=f"Aktif belge ayarlandı: **{resolved}**").send()
        else:
            docs = pipeline.list_documents() if pipeline else []
            await cl.Message(content=f"Belge bulunamadı: **{name}**\nMevcut belgeler: {', '.join(docs) if docs else '(yok)'}").send()
        await _update_documents_sidebar(pipeline)
        return

    # Ensure pipeline exists
    if not pipeline:
        pipeline = _get_pipeline()
    await _sync_profile_to_pipeline(pipeline)
    _thread_pipeline_set(active_ui_thread_id, pipeline)

    # Chat mode does not require documents
    mode = cl.user_session.get("mode") or mode
    if mode == "chat":
        # If user is asking how to return to doc mode, switch and guide.
        if _looks_like_doc_mode_request(query):
            cl.user_session.set("mode", "doc")
            if pipeline.has_documents:
                await cl.Message(
                    content="Belge moduna geçildi. Belge sorunu sorabilirsin. (İstersen `/chat` ile tekrar sohbet moduna dönebilirsin.)"
                ).send()
            else:
                await cl.Message(
                    content="Belge moduna geçildi. Devam etmek için lütfen bir PDF/PNG/JPG yükle. (Sohbet için `/chat` yazabilirsin.)"
                ).send()
            await _update_documents_sidebar(pipeline)
            return

        # Auto-switch to doc if message clearly refers to a loaded document.
        if _looks_like_doc_switch(query, pipeline):
            cl.user_session.set("mode", "doc")
            mode = "doc"
            await _update_documents_sidebar(pipeline)
        else:
            thinking_msg = cl.Message(content="Dusunuyorum...")
            await thinking_msg.send()
            try:
                answer = await _run_chat_response(pipeline, query, chat_style)
            except Exception as e:
                await thinking_msg.remove()
                await _send_standard_error(
                    "Sohbet cevabi olusturulamadi",
                    e,
                    retry_payload={
                        "kind": "chat",
                        "query": query,
                        "chat_style": chat_style,
                        "mode": cl.user_session.get("mode") or mode,
                    },
                )
                return
            await thinking_msg.remove()
            await _stream_text_response(answer)
            _thread_memory_add(active_ui_thread_id, "assistant", answer)
            await _update_documents_sidebar(pipeline)
            return

    mode = cl.user_session.get("mode") or mode
    # Doc mode requires documents
    if not pipeline.has_documents:
        no_doc_msg = "Henuz belge yuklenmedi. Lutfen once bir belge yukleyin. (Sohbet için `/chat` yazabilirsin.)"
        await cl.Message(content=no_doc_msg).send()
        _thread_memory_add(active_ui_thread_id, "assistant", no_doc_msg)
        await _update_documents_sidebar(pipeline)
        return
    if not pipeline.has_index:
        no_index_msg = (
            "Bu oturumda yuklenen belgelerden indeks olusturulamadi (metin cikarimi bos olabilir veya OCR/VLM gerekir).\n\n"
            "- PDF/PNG/JPG’yi tekrar yuklemeyi dene\n"
            "- Taranmis (image-only) PDF ise OCR kurulu oldugundan emin ol (README → OCR)\n"
            "- (Opsiyonel) VLM aciksa VLM_MAX_PAGES limitini kontrol et\n\n"
            "Sohbet için `/chat` yazabilirsin."
        )
        await cl.Message(content=no_index_msg).send()
        _thread_memory_add(active_ui_thread_id, "assistant", no_index_msg)
        await _update_documents_sidebar(pipeline)
        return

    # Show thinking indicator
    thinking_msg = cl.Message(content="Dusunuyorum...")
    await thinking_msg.send()

    # Generate and stream answer (real token stream when provider supports it)
    try:
        result = await _stream_doc_answer_live(pipeline, query, mode, thinking_msg=thinking_msg)
        _thread_memory_add(active_ui_thread_id, "assistant", result.answer)
        await _update_documents_sidebar(pipeline)
    except Exception as e:
        try:
            await thinking_msg.remove()
        except Exception:
            pass
        await _send_standard_error(
            "Belge cevabi olusturulamadi",
            e,
            retry_payload={
                "kind": "ask",
                "query": query,
                "mode": cl.user_session.get("mode") or mode,
            },
        )
        return
