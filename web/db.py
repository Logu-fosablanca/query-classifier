"""
SQLite session store for the REIC demo UI.

Schema
------
sessions : one row per browser session
turns    : one row per user/assistant turn, linked to session

Auto-expiry
-----------
Sessions inactive for more than SESSION_TTL seconds are deleted.
cleanup_loop() runs in the background every CLEANUP_INTERVAL seconds.
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "sessions.db"
SESSION_TTL = 3600        # 1 hour in seconds
CLEANUP_INTERVAL = 300    # run cleanup every 5 minutes


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                created_at  REAL NOT NULL,
                last_active REAL NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT,
                ts          REAL NOT NULL
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)"
        )
        await db.commit()
    logger.info("DB initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

async def create_session() -> str:
    sid = str(uuid.uuid4())
    now = time.time()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, created_at, last_active) VALUES (?, ?, ?)",
            (sid, now, now),
        )
        await db.commit()
    return sid


async def touch_session(db: aiosqlite.Connection, session_id: str) -> None:
    await db.execute(
        "UPDATE sessions SET last_active = ? WHERE id = ?",
        (time.time(), session_id),
    )


async def session_exists(session_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT last_active FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return False
        if time.time() - row[0] > SESSION_TTL:
            await _delete_session(db, session_id)
            await db.commit()
            return False
        return True


# ---------------------------------------------------------------------------
# Turn storage & retrieval
# ---------------------------------------------------------------------------

async def append_turn(
    session_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> None:
    """Add a user or assistant turn and refresh session TTL."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO turns (session_id, role, content, metadata, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(metadata) if metadata else None, time.time()),
        )
        await touch_session(db, session_id)
        await db.commit()


async def get_history(session_id: str) -> list[dict] | None:
    """
    Return the turn list for a session, or None if the session is gone/expired.

    Each element is a dict with at least {role, content}.
    Assistant turns also include intent_classified (for multi-turn pinning)
    and display fields (confidence, rag_examples, matched_categories, mode, turn_mode).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT last_active FROM sessions WHERE id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()

        if not row:
            return None
        if time.time() - row[0] > SESSION_TTL:
            await _delete_session(db, session_id)
            await db.commit()
            return None

        async with db.execute(
            "SELECT role, content, metadata FROM turns WHERE session_id = ? ORDER BY ts",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()

    turns = []
    for role, content, meta_str in rows:
        turn: dict = {"role": role, "content": content}
        if meta_str:
            turn.update(json.loads(meta_str))
        turns.append(turn)
    return turns


async def clear_session_turns(session_id: str) -> None:
    """Delete all turns for a session (keeps the session row, resets TTL)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        await touch_session(db, session_id)
        await db.commit()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def _delete_session(db: aiosqlite.Connection, session_id: str) -> None:
    await db.execute("DELETE FROM turns    WHERE session_id = ?", (session_id,))
    await db.execute("DELETE FROM sessions WHERE id = ?",         (session_id,))


async def cleanup_expired() -> int:
    """Delete all sessions inactive for more than SESSION_TTL. Returns count deleted."""
    cutoff = time.time() - SESSION_TTL
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT id FROM sessions WHERE last_active < ?", (cutoff,)
        ) as cur:
            expired = [r[0] for r in await cur.fetchall()]
        for sid in expired:
            await _delete_session(db, sid)
        await db.commit()
    if expired:
        logger.info("Cleaned up %d expired session(s).", len(expired))
    return len(expired)


async def cleanup_loop() -> None:
    """Background coroutine: run cleanup_expired() every CLEANUP_INTERVAL seconds."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            await cleanup_expired()
        except Exception:
            logger.exception("Error during session cleanup")
