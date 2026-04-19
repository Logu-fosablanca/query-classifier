"""
FastAPI backend for the Intent Classifier demo UI.

Exposes a single POST /api/classify endpoint that runs the full REIC pipeline:
  SemanticRouter (or HierarchicalRouter) → ExampleStore RAG → LLM call

Supports any OpenAI-compatible external LLM endpoint (OpenAI, Groq, Together AI, etc.)
as well as local Ollama.

Run:
    uvicorn web.app:app --reload --port 8000
    # then open http://localhost:8000
"""

import sys
import os

# Make the library importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from query_classifier.config import RAG_CONFIDENCE_BLEND
from web.db import (
    SESSION_TTL,
    append_turn,
    cleanup_loop,
    create_session,
    get_history,
    init_db,
    session_exists,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="Intent Classifier — REIC Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IntentDef(BaseModel):
    name: str
    description: str


class HierarchyCategoryDef(BaseModel):
    description: str
    intents: List[str]


class ExampleDef(BaseModel):
    text: str
    intent: str
    history_context: Optional[str] = None


class ClassifyRequest(BaseModel):
    text: str
    mode: str = "flat"                             # flat | flat_rag | hierarchical_rag
    turn_mode: str = "single"                      # single | multi
    intents: List[IntentDef]
    hierarchy: Optional[Dict[str, HierarchyCategoryDef]] = None
    examples: List[ExampleDef] = []
    conversation_history: List[Dict[str, Any]] = []  # fallback when no session_id
    session_id: Optional[str] = None               # DB-backed session
    llm_base_url: str = "https://api.openai.com"
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_provider: str = "openai"                   # openai | openrouter | ollama


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    reasoning: str = ""
    language: str = "unknown"
    rag_examples: List[dict] = []
    matched_categories: List[str] = []
    session_id: str = ""


# ---------------------------------------------------------------------------
# Component cache (SemanticRouter + HierarchicalRouter + ExampleStore)
# Keyed by hash of intents/hierarchy/examples/mode — rebuilt only when those change.
# ---------------------------------------------------------------------------

_component_cache: Dict[str, Any] = {}
_MAX_CACHE_SIZE = 5


def _config_hash(req: ClassifyRequest) -> str:
    key = json.dumps(
        {
            "intents": [i.dict() for i in req.intents],
            "hierarchy": (
                {k: v.dict() for k, v in req.hierarchy.items()}
                if req.hierarchy
                else None
            ),
            "examples": sorted([e.dict() for e in req.examples], key=lambda x: x["text"]),
            "mode": req.mode,
        },
        sort_keys=True,
    )
    return hashlib.md5(key.encode()).hexdigest()


def _build_components(req: ClassifyRequest) -> dict:
    from query_classifier.semantic_router import SemanticRouter
    from query_classifier.example_store import ExampleStore

    intents_raw = [i.dict() for i in req.intents]

    # Semantic router — also the shared encoder
    router = SemanticRouter(intents=intents_raw)

    # Example store (RAG modes only)
    store = None
    if req.examples and req.mode in ("flat_rag", "hierarchical_rag"):
        store = ExampleStore(encoder=router.model)
        examples_raw = [
            {**e.dict(), "history_context": e.history_context or None}
            for e in req.examples
            if e.text.strip()
        ]
        store.add_examples_bulk(examples_raw)
        logger.info(f"ExampleStore: {len(store)} examples loaded.")

    # Hierarchical router (HIERARCHICAL_RAG only)
    hier_router = None
    hierarchy_raw: dict = {}
    if req.hierarchy:
        hierarchy_raw = {k: v.dict() for k, v in req.hierarchy.items()}

    if req.mode == "hierarchical_rag" and hierarchy_raw and router.model:
        from query_classifier.hierarchy import HierarchicalRouter
        hier_router = HierarchicalRouter(
            intents=intents_raw,
            hierarchy=hierarchy_raw,
            encoder=router.model,
            top_n_categories=2,
        )

    return {
        "router": router,
        "store": store,
        "hier_router": hier_router,
        "hierarchy_raw": hierarchy_raw,
    }


def _get_components(req: ClassifyRequest) -> dict:
    h = _config_hash(req)
    if h not in _component_cache:
        if len(_component_cache) >= _MAX_CACHE_SIZE:
            oldest_key = next(iter(_component_cache))
            del _component_cache[oldest_key]
            logger.info("Cache evicted oldest config.")
        logger.info(f"Building components for config {h[:8]}…")
        _component_cache[h] = _build_components(req)
    return _component_cache[h]


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

async def _call_openai(prompt: str, base_url: str, model: str, api_key: str) -> str:
    """OpenAI-compatible endpoint (OpenAI, Groq, Together AI, etc.)."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _call_openrouter(prompt: str, model: str, api_key: str) -> str:
    """
    OpenRouter (https://openrouter.ai) — OpenAI-compatible with two extra headers
    that OpenRouter uses for rate-limit attribution and dashboard labelling.
    """
    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Logu-fosablanca/Intent-Classification-with-prompts",
        "X-Title": "REIC Intent Classifier Demo",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
        )
        r.raise_for_status()
        data = r.json()
        # OpenRouter surfaces upstream errors inside the JSON body
        if "error" in data:
            raise HTTPException(status_code=502, detail=str(data["error"]))
        return data["choices"][0]["message"]["content"]


async def _call_ollama(prompt: str, base_url: str, model: str, api_key: str) -> str:
    """Local Ollama instance."""
    import ollama

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    client = ollama.AsyncClient(host=base_url, headers=headers)
    r = await client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return r["message"]["content"]


# ---------------------------------------------------------------------------
# /api/classify
# ---------------------------------------------------------------------------

@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    try:
        c = _get_components(req)
        router      = c["router"]
        store       = c["store"]
        hier_router = c["hier_router"]
        hierarchy   = c["hierarchy_raw"]

        is_multi = req.turn_mode == "multi"

        # ── Session / history resolution ────────────────────────────────────
        sid = req.session_id
        if sid and not await session_exists(sid):
            sid = None          # expired — will create fresh below

        if sid is None:
            sid = await create_session()

        # Prefer DB history over what the frontend sent (authoritative source)
        if is_multi:
            db_history = await get_history(sid) or []
            history = db_history
        else:
            history = []

        # ── Store user turn ─────────────────────────────────────────────────
        await append_turn(sid, "user", req.text)

        # 1. Encode raw query for routing
        query_emb = router.encode_query(req.text)

        # 2. Context-enriched RAG query (MULTI mode: prepend last 2 user turns)
        rag_emb = query_emb
        if is_multi and history and store and query_emb is not None:
            prior = []
            for turn in reversed(history):
                if turn.get("role") == "user":
                    content = turn.get("content", "").strip()
                    if content and content != req.text:
                        prior.append(content)
                    if len(prior) >= 2:
                        break
            if prior:
                rag_query = " ".join(reversed(prior)) + " " + req.text
                logger.info(f"RAG query (enriched): {rag_query!r}")
                rag_emb = router.encode_query(rag_query)

        # 3. Prior categories from history (MULTI + HIERARCHICAL_RAG)
        prior_categories: List[str] = []
        if is_multi and history and hier_router:
            for turn in reversed(history):
                if turn.get("role") == "assistant":
                    intent_name = turn.get("intent_classified")
                    if intent_name:
                        cat = hier_router.get_category_for_intent(intent_name)
                        if cat:
                            prior_categories = [cat]
                    break

        # 4. Route
        matched_categories: List[str] = []
        if req.mode == "hierarchical_rag" and hier_router and query_emb is not None:
            matched_categories, top_matches = hier_router.route(
                query_emb, top_k_intents=5, prior_categories=prior_categories
            )
        elif query_emb is not None:
            top_matches = router.find_top_k_from_embedding(query_emb, k=5)
        else:
            top_matches = router.find_top_k(req.text, k=5)

        # 5. RAG retrieval
        rag_examples: List[dict] = []
        if store and req.mode in ("flat_rag", "hierarchical_rag") and rag_emb is not None:
            all_ex = store.retrieve_from_embedding(rag_emb, k=6)

            if matched_categories and req.mode == "hierarchical_rag":
                allowed: set = set()
                for cat in matched_categories:
                    allowed.update(hierarchy.get(cat, {}).get("intents", []))
                filtered = [e for e in all_ex if e["intent"] in allowed]
                if len(filtered) < 3:
                    seen = {e["text"] for e in filtered}
                    for ex in all_ex:
                        if ex["text"] not in seen:
                            filtered.append(ex)
                        if len(filtered) >= 6:
                            break
                rag_examples = filtered
            else:
                rag_examples = all_ex

        # 6. Build prompt (reuse library's prompt builder)
        from query_classifier.nlp_engine import (
            IntentClassifier,
            ClassificationMode,
            TurnMode,
        )

        clf = IntentClassifier.__new__(IntentClassifier)
        clf.mode = ClassificationMode(req.mode)
        clf.turn_mode = TurnMode(req.turn_mode)
        clf._hierarchy = hierarchy

        prompt = clf._build_prompt(
            req.text,
            top_matches,
            rag_examples,
            matched_categories,
            history if is_multi else None,
        )

        # 7. LLM call
        if req.llm_provider == "ollama":
            content = await _call_ollama(
                prompt, req.llm_base_url, req.llm_model_name, req.llm_api_key
            )
        elif req.llm_provider == "openrouter":
            content = await _call_openrouter(
                prompt, req.llm_model_name, req.llm_api_key
            )
        else:
            content = await _call_openai(
                prompt, req.llm_base_url, req.llm_model_name, req.llm_api_key
            )

        # 8. Parse JSON response — fall back to top semantic match on failure
        fallback_intent = top_matches[0]["intent"]["name"] if top_matches else "unknown"
        fallback_confidence = float(top_matches[0]["score"]) if top_matches else 0.0
        reasoning = ""
        try:
            logger.info(f"LLM raw response: {content[:300]}")
            result = clf._extract_json(content)
            intent_name = result.get("name", fallback_intent)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
        except ValueError:
            logger.warning(f"JSON parse failed — using top semantic match: {fallback_intent}")
            intent_name = fallback_intent
            confidence = fallback_confidence

        # 9. Confidence blend (REIC: ground LLM confidence with retrieval evidence)
        if rag_examples and RAG_CONFIDENCE_BLEND > 0:
            top_ret = float(rag_examples[0]["score"])
            confidence = (1.0 - RAG_CONFIDENCE_BLEND) * confidence + RAG_CONFIDENCE_BLEND * top_ret

        # 10. Confidence gate
        if rag_examples and rag_examples[0]["score"] < 0.35:
            confidence = min(confidence, 0.55)

        # ── Store assistant turn ─────────────────────────────────────────────
        await append_turn(sid, "assistant", intent_name, metadata={
            "intent_classified":  intent_name,
            "confidence":         confidence,
            "reasoning":          reasoning,
            "matched_categories": matched_categories,
            "mode":               req.mode,
            "turn_mode":          req.turn_mode,
            "rag_examples":       rag_examples,
        })

        return ClassifyResponse(
            intent=intent_name,
            confidence=confidence,
            reasoning=reasoning,
            rag_examples=rag_examples,
            matched_categories=matched_categories,
            session_id=sid,
        )

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API returned {e.response.status_code}: {e.response.text[:300]}",
        )
    except Exception as e:
        logger.error("Classification failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def new_session():
    """Create a fresh session. Returns session_id + TTL in seconds."""
    sid = await create_session()
    return {"session_id": sid, "ttl_seconds": SESSION_TTL}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Return the turn history for a session (404 if expired/missing)."""
    turns = await get_history(session_id)
    if turns is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {"session_id": session_id, "turns": turns, "ttl_seconds": SESSION_TTL}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a session's turns (used by 'New conversation')."""
    from web.db import clear_session_turns
    await clear_session_turns(session_id)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Model discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ollama/models")
async def ollama_models(base_url: str = "http://localhost:11434"):
    """Proxy to Ollama /api/tags — avoids browser CORS issues."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(base_url.rstrip("/") + "/api/tags")
            r.raise_for_status()
            data = r.json()
            names = [m["name"] for m in data.get("models", [])]
            return {"models": names}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama at " + base_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/openrouter/models")
async def openrouter_models(api_key: str = ""):
    """
    Fetch the list of models available on OpenRouter.
    Returns them sorted: free models first, then by name.
    Proxied server-side to avoid CORS and keep the API key out of browser logs.
    """
    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            r.raise_for_status()
            data = r.json()
        models_raw = data.get("data", [])
        # Build concise list: id + pricing label
        models = []
        for m in models_raw:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", 1) or 1)
            is_free = prompt_cost == 0
            models.append({"id": mid, "free": is_free})
        # Free models first, then alphabetical
        models.sort(key=lambda x: (not x["free"], x["id"]))
        return {"models": [m["id"] for m in models], "free_count": sum(1 for m in models if m["free"])}
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach OpenRouter")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.delete("/api/cache")
def clear_cache():
    _component_cache.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Static files + SPA root
# ---------------------------------------------------------------------------

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(_static_dir, "index.html"))
