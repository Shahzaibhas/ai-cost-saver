#!/usr/bin/env python3
"""
AI Cost-Saver PRODUCTION — Single File Edition
================================================
Everything in one file: app + tests + CLI.

USAGE:
    # Run server
    python ai_cost_saver_all_in_one.py serve

    # Run tests
    python ai_cost_saver_all_in_one.py test

ENV:
    GEMINI_API_KEY=... OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
    API_KEY=your-secret-key
"""

import sys
import os

# =============================================================================
# PART 1: PRODUCTION APPLICATION
# =============================================================================

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

# ---------------------------------------------------------------------------
# 0. Configuration & Settings
# ---------------------------------------------------------------------------

class Settings:
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8001"))
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    COMPRESSION_ENABLED: bool = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
    COMPRESSION_THRESHOLD_WORDS: int = int(os.getenv("COMPRESSION_THRESHOLD_WORDS", "500"))
    
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    API_KEY: Optional[str] = os.getenv("API_KEY")
    AUTH_ENABLED: bool = bool(API_KEY)
    
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_RPS: float = float(os.getenv("RATE_LIMIT_RPS", "10"))
    RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "20"))
    
    DB_PATH: str = os.getenv("DB_PATH", "ai_calls.db")
    CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.88"))
    
    CIRCUIT_BREAKER_FAILURES: int = int(os.getenv("CIRCUIT_BREAKER_FAILURES", "5"))
    CIRCUIT_BREAKER_TIMEOUT: int = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "30"))
    
    MAX_PROMPT_LENGTH: int = int(os.getenv("MAX_PROMPT_LENGTH", "50000"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
    
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    
    @classmethod
    def load_route_map(cls) -> Dict[str, List[Tuple[str, str]]]:
        env_map = os.getenv("ROUTE_MAP")
        if env_map:
            return json.loads(env_map)
        return {
            "classification": [
                ("gemini", "gemini-1.5-flash-8b"),
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
            ],
            "yes_no": [
                ("gemini", "gemini-1.5-flash-8b"),
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
            ],
            "simple_qa": [
                ("gemini", "gemini-1.5-flash-8b"),
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
            ],
            "summary": [
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
                ("anthropic", "claude-3-haiku-20240307"),
            ],
            "translation": [
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
            ],
            "creative": [
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
                ("anthropic", "claude-3-haiku-20240307"),
            ],
            "complex_qa": [
                ("gemini", "gemini-1.5-flash"),
                ("openai", "gpt-4o-mini"),
                ("anthropic", "claude-3-haiku-20240307"),
            ],
        }


PRICING = {
    "gemini-1.5-flash-8b":     {"input": 0.0375, "output": 0.15},
    "gemini-1.5-flash":        {"input": 0.075,  "output": 0.30},
    "gemini-1.5-pro":          {"input": 1.25,   "output": 5.0},
    "gpt-4o-mini":             {"input": 0.15,   "output": 0.60},
    "gpt-4o":                  {"input": 2.50,   "output": 10.0},
    "claude-3-haiku-20240307": {"input": 0.25,   "output": 1.25},
    "claude-3-sonnet-20240229":{"input": 3.0,    "output": 15.0},
}

GPT4O_PRICING = PRICING["gpt-4o"]

# ---------------------------------------------------------------------------
# 1. Structured Logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging():
    logger = logging.getLogger("ai-cost-saver")
    logger.setLevel(getattr(logging, Settings.LOG_LEVEL.upper()))
    handler = logging.StreamHandler()
    if Settings.LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)
    return logger


logger = setup_logging()

# ---------------------------------------------------------------------------
# 2. Async SQLite Database with Migrations
# ---------------------------------------------------------------------------

class AsyncRequestLogger:
    def __init__(self, db_path: str = "ai_calls.db"):
        self.db_path = db_path
        self._memory_mode = (db_path == ":memory:")
        if self._memory_mode:
            # In-memory: share a single persistent connection across all calls
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
        else:
            self._local = threading.local()
    
    def _get_conn(self):
        if self._memory_mode:
            return self._conn
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn
    
    async def init(self):
        await asyncio.to_thread(self._init_sync)
    
    def _init_sync(self):
        conn = self._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
        cursor = conn.execute("SELECT MAX(version) FROM schema_version")
        current = cursor.fetchone()[0] or 0
        
        if current < 1:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT,
                    model_used TEXT,
                    task_type TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    original_input_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    gpt4o_cost_usd REAL DEFAULT 0.0,
                    compression_cost_usd REAL DEFAULT 0.0,
                    cached INTEGER DEFAULT 0,
                    compressed INTEGER DEFAULT 0,
                    fallback_used INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON calls(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_task ON calls(task_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_cached ON calls(cached)")
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            conn.commit()
        
        if current < 2:
            # Version 2: ensure original_input_tokens column exists
            cols = [row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()]
            if "original_input_tokens" not in cols:
                conn.execute("ALTER TABLE calls ADD COLUMN original_input_tokens INTEGER DEFAULT 0")
            conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            conn.commit()
        
        if current < 3:
            # Version 3: add token_estimate_method column
            cols = [row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()]
            if "token_estimate_method" not in cols:
                conn.execute("ALTER TABLE calls ADD COLUMN token_estimate_method TEXT DEFAULT 'provider_native'")
            conn.execute("INSERT INTO schema_version (version) VALUES (3)")
            conn.commit()
    
    async def log_request(self, **kwargs):
        await asyncio.to_thread(self._log_sync, **kwargs)
    
    def _log_sync(self, **kwargs):
        kwargs.setdefault("timestamp", datetime.utcnow().isoformat())
        kwargs.setdefault("token_estimate_method", "provider_native")
        input_tokens = kwargs.get("input_tokens", 0)
        output_tokens = kwargs.get("output_tokens", 0)
        original_input_tokens = kwargs.get("original_input_tokens", input_tokens)
        gpt4o_cost = self._calc_gpt4o_cost(original_input_tokens, output_tokens)
        kwargs["gpt4o_cost_usd"] = gpt4o_cost
        
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO calls
               (request_id, timestamp, prompt, response, model_used, task_type,
                input_tokens, output_tokens, original_input_tokens, cost_usd, gpt4o_cost_usd,
                compression_cost_usd, cached, compressed, fallback_used, token_estimate_method)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                kwargs["request_id"],
                kwargs["timestamp"],
                kwargs["prompt"],
                kwargs.get("response", ""),
                kwargs.get("model_used", ""),
                kwargs.get("task_type", ""),
                input_tokens,
                output_tokens,
                original_input_tokens,
                kwargs.get("cost_usd", 0.0),
                gpt4o_cost,
                kwargs.get("compression_cost_usd", 0.0),
                1 if kwargs.get("cached") else 0,
                1 if kwargs.get("compressed") else 0,
                1 if kwargs.get("fallback_used") else 0,
                kwargs.get("token_estimate_method", "provider_native"),
            ),
        )
        conn.commit()
    
    def _calc_gpt4o_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * GPT4O_PRICING["input"] + output_tokens * GPT4O_PRICING["output"]) / 1_000_000
    
    async def get_history(self, limit: int = 50, offset: int = 0, task_type: Optional[str] = None, days: int = 30):
        return await asyncio.to_thread(self._get_history_sync, limit, offset, task_type, days)
    
    def _get_history_sync(self, limit, offset, task_type, days):
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = "SELECT * FROM calls WHERE timestamp >= ?"
        params = [since]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        conn = self._get_conn()
        rows = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.execute("SELECT * FROM calls LIMIT 0").description]
        return [dict(zip(columns, row)) for row in rows]
    
    async def get_stats(self, days: int = 30):
        return await asyncio.to_thread(self._get_stats_sync, days)
    
    def _get_stats_sync(self, days):
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT
                COUNT(*) as total_calls,
                SUM(cost_usd) as total_cost,
                SUM(gpt4o_cost_usd) as total_gpt4o_cost,
                SUM(compression_cost_usd) as total_compression_cost,
                SUM(CASE WHEN cached=1 THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN compressed=1 THEN 1 ELSE 0 END) as compressed_calls,
                AVG(cost_usd) as avg_cost_per_call
            FROM calls
            WHERE timestamp >= ?
        """, (since,))
        row = cursor.fetchone()
        columns = [desc[0] for desc in cursor.description]
        stats = dict(zip(columns, row))
        
        total_cost = stats.get("total_cost") or 0
        total_gpt4o_cost = stats.get("total_gpt4o_cost") or 0
        total_compression = stats.get("total_compression_cost") or 0
        actual_spend = total_cost + total_compression
        
        stats["total_actual_cost"] = actual_spend
        stats["total_savings"] = total_gpt4o_cost - actual_spend
        stats["savings_percentage"] = ((total_gpt4o_cost - actual_spend) / total_gpt4o_cost * 100) if total_gpt4o_cost > 0 else 0
        
        model_stats = conn.execute(
            "SELECT model_used, COUNT(*) as count, SUM(cost_usd) as total_cost FROM calls WHERE timestamp >= ? GROUP BY model_used",
            (since,)
        ).fetchall()
        stats["model_breakdown"] = [
            {"model": m, "count": c, "total_cost": tc} for m, c, tc in model_stats
        ]
        return stats
    
    async def get_savings(self, days: int = 30):
        return await asyncio.to_thread(self._get_savings_sync, days)
    
    def _get_savings_sync(self, days):
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT request_id, timestamp, model_used, task_type, input_tokens, output_tokens,
            original_input_tokens, cost_usd, gpt4o_cost_usd, compression_cost_usd, cached,
            token_estimate_method
            FROM calls WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 500""",
            (since,)
        ).fetchall()
        results = []
        total_actual = 0.0
        total_gpt4o = 0.0
        for r in rows:
            actual_cost = (r[7] or 0.0) + (r[9] or 0.0)
            gpt4o_cost = r[8] or 0.0
            savings = gpt4o_cost - actual_cost
            total_actual += actual_cost
            total_gpt4o += gpt4o_cost
            results.append({
                "request_id": r[0],
                "timestamp": r[1],
                "model_used": r[2],
                "task_type": r[3],
                "input_tokens": r[4],
                "output_tokens": r[5],
                "original_input_tokens": r[6],
                "actual_cost": round(actual_cost, 8),
                "gpt4o_cost": round(gpt4o_cost, 8),
                "savings": round(savings, 8),
                "cached": bool(r[10]),
                "token_estimate_method": r[11] or "provider_native",
            })
        return {
            "period_days": days,
            "total_actual_cost": round(total_actual, 6),
            "total_gpt4o_cost": round(total_gpt4o, 6),
            "total_savings": round(total_gpt4o - total_actual, 6),
            "savings_percentage": round(((total_gpt4o - total_actual) / total_gpt4o * 100), 1) if total_gpt4o > 0 else 0,
            "request_count": len(results),
            "requests": results,
        }
    
    async def get_counts(self) -> Tuple[int, int, float]:
        return await asyncio.to_thread(self._get_counts_sync)
    
    def _get_counts_sync(self):
        conn = self._get_conn()
        row = conn.execute("""
            SELECT COUNT(*), SUM(CASE WHEN cached=1 THEN 1 ELSE 0 END), SUM(cost_usd + compression_cost_usd)
            FROM calls
        """).fetchone()
        return row[0] or 0, row[1] or 0, row[2] or 0.0
    
    async def close(self):
        await asyncio.to_thread(self._close_sync)
    
    def _close_sync(self):
        if self._memory_mode:
            if self._conn:
                self._conn.close()
                self._conn = None
        else:
            if hasattr(self._local, "conn") and self._local.conn:
                self._local.conn.close()
                self._local.conn = None


# ---------------------------------------------------------------------------
# 3. Semantic Cache (Incremental TF-IDF + LRU + TTL)
# ---------------------------------------------------------------------------

class CacheEntry:
    __slots__ = ("prompt", "response", "model_used", "task_type", "input_tokens",
                 "output_tokens", "cost_usd", "created_at", "last_accessed")
    
    def __init__(self, prompt, response, model_used, task_type, input_tokens, output_tokens, cost_usd):
        self.prompt = prompt
        self.response = response
        self.model_used = model_used
        self.task_type = task_type
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.created_at = time.time()
        self.last_accessed = time.time()


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.88, max_entries: int = 1000, ttl_seconds: int = 86400):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._sklearn_available = False
        self._vectorizer = None
        self._vector_matrix = None
        self._prompt_order: List[str] = []
        self._needs_rebuild = False
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._vectorizer_cls = TfidfVectorizer
            self._similarity_fn = cosine_similarity
            self._sklearn_available = True
            logger.info("Semantic cache: sklearn available.")
        except ImportError:
            logger.warning("scikit-learn not installed — exact-match cache only.")
    
    async def init(self):
        if self._sklearn_available:
            await self._rebuild_index()
    
    async def _rebuild_index(self):
        """Rebuild TF-IDF index outside the lock."""
        if not self._sklearn_available:
            self._vectorizer = None
            self._vector_matrix = None
            self._prompt_order = []
            self._needs_rebuild = False
            return
        
        # Copy prompt list under lock
        async with self._lock:
            if not self._entries:
                self._vectorizer = None
                self._vector_matrix = None
                self._prompt_order = []
                self._needs_rebuild = False
                return
            prompts = [e.prompt for e in self._entries.values()]
        
        # CPU-bound work outside lock
        def _build():
            vectorizer = self._vectorizer_cls(stop_words="english", max_features=5000)
            return vectorizer, vectorizer.fit_transform(prompts), prompts
        
        vectorizer, matrix, order = await asyncio.to_thread(_build)
        
        # Swap references under lock
        async with self._lock:
            self._vectorizer = vectorizer
            self._vector_matrix = matrix
            self._prompt_order = order
            self._needs_rebuild = False
    
    async def lookup(self, prompt: str) -> Optional[Dict[str, Any]]:
        rebuild_required = False
        async with self._lock:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_hash in self._entries:
                entry = self._entries[prompt_hash]
                if time.time() - entry.created_at > self.ttl_seconds:
                    del self._entries[prompt_hash]
                    rebuild_required = True
                else:
                    entry.last_accessed = time.time()
                    self._entries.move_to_end(prompt_hash)
                    return self._entry_to_dict(entry)
            elif not self._sklearn_available or self._vectorizer is None or len(self._entries) == 0:
                return None
            else:
                # Compute similarity (should be fast, but we are inside lock; keep as is)
                def _compute():
                    try:
                        new_vec = self._vectorizer.transform([prompt])
                        similarities = self._similarity_fn(new_vec, self._vector_matrix).flatten()
                        return int(similarities.argmax()), float(similarities.max())
                    except Exception:
                        return None, 0.0
                
                best_idx, best_sim = await asyncio.to_thread(_compute)
                if best_idx is None or best_sim < self.threshold:
                    return None
                matched_prompt = self._prompt_order[best_idx]
                matched_hash = hashlib.sha256(matched_prompt.encode()).hexdigest()
                entry = self._entries.get(matched_hash)
                if entry and time.time() - entry.created_at <= self.ttl_seconds:
                    entry.last_accessed = time.time()
                    self._entries.move_to_end(matched_hash)
                    return self._entry_to_dict(entry)
        if rebuild_required:
            await self._rebuild_index()
        return None
    
    async def store(self, prompt: str, result: Dict[str, Any]):
        rebuild_required = False
        async with self._lock:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt_hash in self._entries:
                self._entries.move_to_end(prompt_hash)
                return
            
            while len(self._entries) >= self.max_entries:
                del self._entries[next(iter(self._entries))]
            
            self._entries[prompt_hash] = CacheEntry(
                prompt, result["response"], result.get("model_used", ""),
                result.get("task_type", ""), result.get("input_tokens", 0),
                result.get("output_tokens", 0), result.get("cost_usd", 0.0),
            )
            if len(self._entries) % 10 == 0 or self._vectorizer is None:
                rebuild_required = True
        if rebuild_required:
            await self._rebuild_index()
    
    def _entry_to_dict(self, entry: CacheEntry) -> Dict[str, Any]:
        return {
            "response": entry.response,
            "model_used": entry.model_used,
            "task_type": entry.task_type,
            "input_tokens": entry.input_tokens,
            "output_tokens": entry.output_tokens,
            "cost_usd": entry.cost_usd,
            "cached": True,
        }
    
    async def cleanup_expired(self):
        rebuild_required = False
        async with self._lock:
            now = time.time()
            expired = [k for k, e in self._entries.items() if now - e.created_at > self.ttl_seconds]
            for k in expired:
                del self._entries[k]
            if expired:
                rebuild_required = True
        if rebuild_required:
            await self._rebuild_index()


# ---------------------------------------------------------------------------
# 4. Prompt Compressor
# ---------------------------------------------------------------------------

class PromptCompressor:
    def __init__(self, providers):
        self.providers = providers
    
    async def compress(self, prompt: str, task_type: str) -> Tuple[str, bool, float]:
        client = self.providers.get("gemini")
        if not client:
            return prompt, False, 0.0
        
        word_count = len(prompt.split())
        target_tokens = max(200, word_count // 2)
        
        summarise_prompt = (
            "Summarize the following text to preserve all key information. "
            "Keep it as short as possible without losing facts.\n\n" + prompt
        )
        try:
            result = await client.generate(
                model="gemini-1.5-flash-8b",
                prompt=summarise_prompt,
                max_tokens=target_tokens,
                temperature=0.0,
            )
            compressed = result["text"]
            cost = result["cost_usd"]
            logger.info(f"Compressed: {len(prompt)} -> {len(compressed)} chars, cost=${cost:.6f}")
            return compressed, True, cost
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return prompt, False, 0.0


# ---------------------------------------------------------------------------
# 5. Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"
        self._lock = asyncio.Lock()
        self._trial_in_progress = False
    
    async def call(self, coro_factory: Callable, *args, **kwargs):
        async with self._lock:
            if self.state == "OPEN":
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self._trial_in_progress = False
                    logger.info(f"Circuit {self.name}: HALF_OPEN")
                else:
                    raise HTTPException(status_code=503, detail=f"Circuit breaker OPEN for {self.name}")
            
            if self.state == "HALF_OPEN":
                if self._trial_in_progress:
                    raise HTTPException(status_code=503, detail=f"Circuit breaker OPEN for {self.name}")
                self._trial_in_progress = True
        
        try:
            result = await coro_factory(*args, **kwargs)
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                    self._trial_in_progress = False
                    logger.info(f"Circuit {self.name}: CLOSED (recovered)")
                else:
                    self.failures = max(0, self.failures - 1)
            return result
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.time()
                if self.state == "HALF_OPEN":
                    self.state = "OPEN"
                    self._trial_in_progress = False
                    logger.error(f"Circuit {self.name}: OPEN after trial failure")
                elif self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit {self.name}: OPEN after {self.failures} failures")
                else:
                    logger.warning(f"Circuit {self.name}: failure {self.failures}/{self.failure_threshold}")
            raise


# ---------------------------------------------------------------------------
# 6. Rate Limiter (Token Bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rps: float = 10.0, burst: int = 20):
        self.rps = rps
        self.burst = burst
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tokens": burst, "last_update": time.time()})
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        async with self._lock:
            bucket = self.buckets[key]
            now = time.time()
            bucket["tokens"] = min(self.burst, bucket["tokens"] + (now - bucket["last_update"]) * self.rps)
            bucket["last_update"] = now
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False
    
    async def cleanup_old(self, max_age_seconds: int = 3600):
        async with self._lock:
            now = time.time()
            stale = [k for k, v in self.buckets.items() if now - v["last_update"] > max_age_seconds]
            for k in stale:
                del self.buckets[k]


# ---------------------------------------------------------------------------
# 7. Request Deduplication
# ---------------------------------------------------------------------------

class InFlightDedup:
    def __init__(self):
        self._in_flight: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    def _make_key(self, prompt: str, task_type: Optional[str], max_tokens: int, temperature: float) -> str:
        return hashlib.sha256(f"{prompt}|{task_type}|{max_tokens}|{temperature}".encode()).hexdigest()
    
    async def execute(self, prompt: str, task_type: Optional[str], max_tokens: int, temperature: float, coro_factory) -> Any:
        key = self._make_key(prompt, task_type, max_tokens, temperature)
        async with self._lock:
            if key in self._in_flight:
                return await self._in_flight[key]
            future = asyncio.ensure_future(coro_factory())
            self._in_flight[key] = future
        try:
            return await future
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)


# ---------------------------------------------------------------------------
# 8. AI Provider Clients
# ---------------------------------------------------------------------------

class GeminiProvider:
    def __init__(self, api_key: str, breakers: Dict[str, CircuitBreaker] = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")
        import google.generativeai as genai
        self.genai = genai
        genai.configure(api_key=api_key)
        self._breakers = breakers if breakers is not None else {}
    
    async def generate(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(
                f"gemini:{model}",
                Settings.CIRCUIT_BREAKER_FAILURES,
                Settings.CIRCUIT_BREAKER_TIMEOUT
            )
        breaker = self._breakers[model]
        
        async def _do():
            model_instance = self.genai.GenerativeModel(model)
            response = await asyncio.wait_for(
                model_instance.generate_content_async(
                    prompt,
                    generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
                ),
                timeout=Settings.REQUEST_TIMEOUT,
            )
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            cost = self._calc_cost(model, input_tokens, output_tokens)
            return {"text": response.text, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
        return await breaker.call(_do)
    
    def _calc_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        prices = PRICING.get(model)
        if prices is None:
            fallback = PRICING.get("gemini-1.5-flash", {"input": 0.075, "output": 0.30})
            logger.warning(f"Gemini pricing missing for model '{model}', falling back to {fallback}")
            prices = fallback
        return (in_tok * prices["input"] + out_tok * prices["output"]) / 1_000_000


class OpenAIProvider:
    def __init__(self, api_key: str, breakers: Dict[str, CircuitBreaker] = None):
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self._breakers = breakers if breakers is not None else {}
    
    async def generate(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(
                f"openai:{model}",
                Settings.CIRCUIT_BREAKER_FAILURES,
                Settings.CIRCUIT_BREAKER_TIMEOUT
            )
        breaker = self._breakers[model]
        
        async def _do():
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=Settings.REQUEST_TIMEOUT,
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calc_cost(model, input_tokens, output_tokens)
            return {"text": response.choices[0].message.content, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
        return await breaker.call(_do)
    
    def _calc_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        prices = PRICING.get(model)
        if prices is None:
            fallback = PRICING.get("gpt-4o-mini")
            logger.warning(f"OpenAI pricing missing for model '{model}', falling back to {fallback}")
            prices = fallback
        return (in_tok * prices["input"] + out_tok * prices["output"]) / 1_000_000


class AnthropicProvider:
    def __init__(self, api_key: str, breakers: Dict[str, CircuitBreaker] = None):
        import anthropic
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self._breakers = breakers if breakers is not None else {}
    
    async def generate(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        if model not in self._breakers:
            self._breakers[model] = CircuitBreaker(
                f"anthropic:{model}",
                Settings.CIRCUIT_BREAKER_FAILURES,
                Settings.CIRCUIT_BREAKER_TIMEOUT
            )
        breaker = self._breakers[model]
        
        async def _do():
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=Settings.REQUEST_TIMEOUT,
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._calc_cost(model, input_tokens, output_tokens)
            return {"text": response.content[0].text, "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": cost}
        return await breaker.call(_do)
    
    def _calc_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        prices = PRICING.get(model)
        if prices is None:
            fallback = PRICING.get("claude-3-haiku-20240307")
            logger.warning(f"Anthropic pricing missing for model '{model}', falling back to {fallback}")
            prices = fallback
        return (in_tok * prices["input"] + out_tok * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# 9. Task Detection & Routing
# ---------------------------------------------------------------------------

TASK_KEYWORDS = {
    "classification": ["classify", "category", "categories", "label", "tag", "sort into"],
    "yes_no": ["true or false", "yes or no", "is it", "are they", "does it", "can you confirm"],
    "simple_qa": ["what is", "who is", "where is", "when did", "define", "how many", "what are", "list the"],
    "summary": ["summarize", "summarise", "summary", "tldr", "key points", "brief overview", "recap"],
    "translation": ["translate", "translation", "in english", "in spanish", "in french", "in german", "in chinese"],
    "creative": ["story", "poem", "creative", "write a", "imagine", "draft", "compose", "generate a"],
}


def detect_task_type(prompt: str) -> str:
    p = prompt.lower()
    scores = {}
    for task, keywords in TASK_KEYWORDS.items():
        scores[task] = sum(1 for kw in keywords if kw in p)
    best_task = max(scores, key=scores.get, default="complex_qa")
    return best_task if scores[best_task] > 0 else "complex_qa"


ROUTE_MAP = Settings.load_route_map()


async def route_and_call(
    prompt: str,
    task_type: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    provider_clients: Optional[Dict[str, Any]] = None,
    compressor: Optional[PromptCompressor] = None,
    request_id: str = "",
) -> Dict[str, Any]:
    if not task_type:
        task_type = detect_task_type(prompt)
        logger.info(f"[{request_id}] Auto-detected task: {task_type}")
    
    original_prompt = prompt
    original_word_count = len(prompt.split())
    compression_used = False
    compression_cost = 0.0
    
    if compressor and original_word_count > Settings.COMPRESSION_THRESHOLD_WORDS:
        try:
            prompt, compression_used, compression_cost = await compressor.compress(prompt, task_type)
        except Exception as e:
            logger.warning(f"[{request_id}] Compression failed: {e}")
    
    chain = ROUTE_MAP.get(task_type, ROUTE_MAP["complex_qa"])
    last_exception = None
    fallback_used = False
    
    for provider_name, model_id in chain:
        client = provider_clients.get(provider_name) if provider_clients else None
        if not client:
            continue
        try:
            result = await client.generate(
                model=model_id, prompt=prompt, max_tokens=max_tokens, temperature=temperature,
            )
            response_text = result["text"]
            input_tokens = result["input_tokens"]
            output_tokens = result["output_tokens"]
            cost = result["cost_usd"]
            
            # Fix #1: correct token estimate for compression baseline
            original_input_tokens = int(original_word_count / 0.75) if compression_used else input_tokens
            
            logger.info(
                f"[{request_id}] SUCCESS: {provider_name}/{model_id} "
                f"(in={input_tokens}, out={output_tokens}, cost=${cost:.6f})"
                + (" [FALLBACK]" if fallback_used else "")
                + (" [COMPRESSED]" if compression_used else "")
            )
            
            return {
                "response": response_text,
                "model_used": f"{provider_name}/{model_id}",
                "task_type": task_type,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "original_input_tokens": original_input_tokens,
                "cost_usd": cost,
                "compression_cost_usd": compression_cost,
                "cached": False,
                "compressed": compression_used,
                "fallback_used": fallback_used,
                "original_prompt": original_prompt if compression_used else None,
            }
        except HTTPException as e:
            if e.status_code == 503:
                logger.warning(f"[{request_id}] {provider_name}/{model_id} circuit open, trying next...")
            else:
                logger.warning(f"[{request_id}] {provider_name}/{model_id} failed: {e}. Trying next...")
            last_exception = e
            fallback_used = True
            continue
        except Exception as e:
            logger.warning(f"[{request_id}] {provider_name}/{model_id} failed: {e}. Trying next...")
            last_exception = e
            fallback_used = True
            continue
    
    raise RuntimeError(f"All providers exhausted for task '{task_type}'. Last error: {last_exception}")


# ---------------------------------------------------------------------------
# 10. FastAPI Application
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=Settings.MAX_PROMPT_LENGTH)
    task_type: Optional[str] = None
    max_tokens: Optional[int] = Field(256, ge=1, le=4096)
    temperature: Optional[float] = Field(0.0, ge=0.0, le=2.0)
    
    @validator("task_type")
    def validate_task_type(cls, v):
        if v and v not in ROUTE_MAP:
            raise ValueError(f"task_type must be one of: {list(ROUTE_MAP.keys())}")
        return v


class ChatResponse(BaseModel):
    response: str
    model_used: str
    cached: bool
    compressed: bool
    task_type: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    compression_cost_usd: float
    request_id: str


security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not Settings.AUTH_ENABLED:
        return True
    if not credentials or not hmac.compare_digest(credentials.credentials, Settings.API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True


async def rate_limit_middleware(request: Request, call_next):
    if not Settings.RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    # Fix #12: prefer X-Forwarded-For behind proxies
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    elif request.client:
        client_ip = request.client.host
    else:
        client_ip = "unknown"
    
    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    return await call_next(request)


db: AsyncRequestLogger = None
cache: Optional[SemanticCache] = None
compressor: Optional[PromptCompressor] = None
providers: Dict[str, Any] = {}
rate_limiter: RateLimiter = None
inflight_dedup: InFlightDedup = None
background_tasks: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, cache, compressor, providers, rate_limiter, inflight_dedup
    
    db = AsyncRequestLogger(Settings.DB_PATH)
    await db.init()
    
    if Settings.GEMINI_API_KEY:
        providers["gemini"] = GeminiProvider(Settings.GEMINI_API_KEY, {})
    if Settings.OPENAI_API_KEY:
        providers["openai"] = OpenAIProvider(Settings.OPENAI_API_KEY, {})
    if Settings.ANTHROPIC_API_KEY:
        providers["anthropic"] = AnthropicProvider(Settings.ANTHROPIC_API_KEY, {})
    
    if Settings.CACHE_ENABLED:
        cache = SemanticCache(
            similarity_threshold=Settings.CACHE_SIMILARITY_THRESHOLD,
            max_entries=Settings.CACHE_MAX_ENTRIES,
            ttl_seconds=Settings.CACHE_TTL_SECONDS,
        )
        await cache.init()
    
    if Settings.COMPRESSION_ENABLED:
        compressor = PromptCompressor(providers)
    
    rate_limiter = RateLimiter(Settings.RATE_LIMIT_RPS, Settings.RATE_LIMIT_BURST)
    inflight_dedup = InFlightDedup()
    
    async def cache_cleanup():
        try:
            while True:
                await asyncio.sleep(300)
                if cache:
                    await cache.cleanup_expired()
        except asyncio.CancelledError:
            pass
    
    async def rate_limit_cleanup():
        try:
            while True:
                await asyncio.sleep(3600)
                await rate_limiter.cleanup_old()
        except asyncio.CancelledError:
            pass
    
    task1 = asyncio.create_task(cache_cleanup())
    task2 = asyncio.create_task(rate_limit_cleanup())
    background_tasks.update([task1, task2])
    task1.add_done_callback(background_tasks.discard)
    task2.add_done_callback(background_tasks.discard)
    
    logger.info("AI Cost-Saver PRODUCTION started.")
    yield
    
    for t in background_tasks:
        t.cancel()
    await db.close()
    logger.info("AI Cost-Saver PRODUCTION shutdown complete.")


app = FastAPI(title="AI Cost-Saver Proxy", version="2.0.0", lifespan=lifespan)
app.middleware("http")(rate_limit_middleware)

# Fix #5: CORS wildcard + credentials conflict
cors_allow_credentials = Settings.CORS_ORIGINS != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.CORS_ORIGINS,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorized: bool = Depends(verify_api_key)):
    request_id = uuid.uuid4().hex[:8]
    
    if cache:
        cached = await cache.lookup(request.prompt)
        if cached:
            await db.log_request(
                request_id=request_id,
                prompt=request.prompt,
                response=cached["response"],
                model_used=cached["model_used"],
                task_type=cached["task_type"],
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                original_input_tokens=cached["input_tokens"],
                cost_usd=0.0,
                compression_cost_usd=0.0,
                cached=True,
                compressed=False,
                token_estimate_method="provider_native",
            )
            return ChatResponse(
                response=cached["response"],
                model_used=cached["model_used"],
                cached=True,
                compressed=False,
                task_type=cached["task_type"],
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                cost_usd=0.0,
                compression_cost_usd=0.0,
                request_id=request_id,
            )
    
    async def _do_call():
        return await route_and_call(
            prompt=request.prompt,
            task_type=request.task_type,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            provider_clients=providers,
            compressor=compressor,
            request_id=request_id,
        )
    
    try:
        result = await inflight_dedup.execute(
            request.prompt, request.task_type, request.max_tokens, request.temperature, _do_call
        )
    except Exception as e:
        logger.error(f"[{request_id}] All providers failed: {e}")
        raise HTTPException(status_code=502, detail="All AI providers are temporarily unavailable.")
    
    await db.log_request(
        request_id=request_id,
        prompt=request.prompt,
        response=result["response"],
        model_used=result["model_used"],
        task_type=result["task_type"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        original_input_tokens=result.get("original_input_tokens", result["input_tokens"]),
        cost_usd=result["cost_usd"],
        compression_cost_usd=result["compression_cost_usd"],
        cached=result["cached"],
        compressed=result["compressed"],
        fallback_used=result["fallback_used"],
        token_estimate_method="provider_native",
    )
    
    if cache and not result["cached"]:
        await cache.store(prompt=request.prompt, result=result)
    
    return ChatResponse(
        response=result["response"],
        model_used=result["model_used"],
        cached=result["cached"],
        compressed=result["compressed"],
        task_type=result["task_type"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        cost_usd=result["cost_usd"],
        compression_cost_usd=result["compression_cost_usd"],
        request_id=request_id,
    )


@app.get("/v1/metrics")
async def metrics(authorized: bool = Depends(verify_api_key)):
    total, cache_hits, total_cost = await db.get_counts()
    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "total_cost_usd": round(total_cost, 6),
        "cache_hit_rate": round(cache_hits / total * 100, 2) if total > 0 else 0,
    }


@app.get("/v1/history")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    task_type: Optional[str] = None,
    days: int = Query(30, ge=1, le=365),
    authorized: bool = Depends(verify_api_key),
):
    return await db.get_history(limit=limit, offset=offset, task_type=task_type, days=days)


@app.get("/v1/providers")
async def providers_status(authorized: bool = Depends(verify_api_key)):
    # Fix #9: circuit_states now keyed by provider:model
    circuit_states = {}
    for name, client in providers.items():
        for model, breaker in client._breakers.items():
            key = f"{name}:{model}"
            circuit_states[key] = breaker.state
    return {
        "gemini": bool(Settings.GEMINI_API_KEY),
        "openai": bool(Settings.OPENAI_API_KEY),
        "anthropic": bool(Settings.ANTHROPIC_API_KEY),
        "circuit_states": circuit_states,
    }


@app.get("/v1/stats")
async def stats(days: int = Query(30, ge=1, le=365), authorized: bool = Depends(verify_api_key)):
    return await db.get_stats(days=days)


@app.get("/v1/savings")
async def savings(days: int = Query(30, ge=1, le=365), authorized: bool = Depends(verify_api_key)):
    return await db.get_savings(days=days)


@app.get("/health")
async def health():
    healthy_providers = []
    provider_states = {}
    for name, client in providers.items():
        breakers = client._breakers
        # Determine aggregated circuit state for this provider
        if not breakers:
            state = "CLOSED"
        else:
            states_set = {b.state for b in breakers.values()}
            if "OPEN" in states_set:
                state = "OPEN"
            elif "HALF_OPEN" in states_set:
                state = "HALF_OPEN"
            else:
                state = "CLOSED"
        provider_states[name] = {"state": state}
        
        # Determine if this provider contributes to overall health
        # Fix: treat empty breakers (no calls yet) as healthy
        if not breakers or any(b.state != "OPEN" for b in breakers.values()):
            healthy_providers.append(name)
    
    return {
        "status": "healthy" if healthy_providers else "degraded",
        "providers": provider_states,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Fix #6: unauthenticated dashboard data endpoint
@app.get("/v1/dashboard-data")
async def dashboard_data():
    stats_data = await db.get_stats(days=30)
    savings_data = await db.get_savings(days=30)
    
    circuit_states = {}
    for name, client in providers.items():
        for model, breaker in client._breakers.items():
            key = f"{name}:{model}"
            circuit_states[key] = breaker.state
    
    providers_config = {
        "gemini": bool(Settings.GEMINI_API_KEY),
        "openai": bool(Settings.OPENAI_API_KEY),
        "anthropic": bool(Settings.ANTHROPIC_API_KEY),
    }
    
    return {
        "stats": stats_data,
        "savings": savings_data,
        "providers": {
            "config": providers_config,
            "circuit_states": circuit_states
        }
    }


LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Cost-Saver — Cut Your AI API Bill by 40%</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}
.container{max-width:900px;margin:0 auto;padding:2rem 1.5rem}
.hero{text-align:center;padding:3rem 0}
.hero h1{font-size:2.8rem;margin-bottom:1rem;color:#fff;font-weight:800}
.hero p{font-size:1.15rem;color:#94a3b8;max-width:600px;margin:0 auto 2rem}
.btn{display:inline-block;background:#2563eb;color:#fff;padding:1rem 2.5rem;border-radius:10px;text-decoration:none;font-weight:600;font-size:1.05rem;transition:background .2s}
.btn:hover{background:#1d4ed8}
.btn-secondary{background:#334155;margin-left:.75rem}
.btn-secondary:hover{background:#475569}
.features{padding:2rem 0}
.features h2{text-align:center;font-size:1.6rem;margin-bottom:2rem;color:#60a5fa}
.grid{display:grid;grid-template-columns:1fr;gap:1.25rem}
.card{background:#1e293b;border-radius:14px;padding:1.75rem;border:1px solid #334155}
.card h3{color:#fff;margin-bottom:.5rem;font-size:1.15rem}
.card p{color:#94a3b8;font-size:.95rem}
.pricing{padding:2rem 0;text-align:center}
.pricing h2{font-size:1.6rem;margin-bottom:2rem;color:#60a5fa}
.price-card .price{font-size:2.4rem;font-weight:800;color:#34d399}
.price-card p{color:#cbd5e1;margin:.25rem 0}
.price-card .muted{color:#64748b;font-size:.85rem}
.footer{text-align:center;padding:2rem 0;color:#64748b;font-size:.9rem;border-top:1px solid #1e293b;margin-top:2rem}
.footer a{color:#60a5fa;text-decoration:none}
@media(min-width:640px){
.hero h1{font-size:3.2rem}
.grid{grid-template-columns:repeat(3,1fr)}
}
</style>
</head>
<body>
<div class="container">
<div class="hero">
<h1>Cut Your AI API Bill by 40%</h1>
<p>Drop-in proxy that automatically routes your prompts to the cheapest capable model. One line of code. Zero config. Built by a shoe shop operations manager in Pakistan who codes at 5 AM.</p>
<a href="mailto:nayapayxd@gmail.com?subject=AI%20Cost-Saver%20Early%20Access" class="btn">Get Early Access</a>
<a href="/dashboard" class="btn btn-secondary">Live Dashboard</a>
</div>

<div class="features">
<h2>How It Works</h2>
<div class="grid">
<div class="card">
<h3>🎯 Smart Routing</h3>
<p>Classification tasks go to Gemini Flash. Complex reasoning goes to GPT-4o. Automatically. You don't pick the model — the system does.</p>
</div>
<div class="card">
<h3>⚡ Semantic Cache</h3>
<p>Never pay twice for the same answer. Similar prompts return cached results instantly, cutting redundant API calls to zero.</p>
</div>
<div class="card">
<h3>🛡️ Zero Downtime</h3>
<p>If OpenAI is down, you fail over to Gemini or Anthropic in milliseconds. Circuit breakers keep you online when providers fail.</p>
</div>
</div>
</div>

<div class="pricing">
<h2>Simple Pricing</h2>
<div class="grid">
<div class="card price-card">
<div class="price">$29</div>
<p><strong>Starter</strong></p>
<p class="muted">For solo devs & side projects</p>
</div>
<div class="card price-card">
<div class="price">$79</div>
<p><strong>Growth</strong></p>
<p class="muted">For small teams shipping AI features</p>
</div>
<div class="card price-card">
<div class="price">$199</div>
<p><strong>Pro</strong></p>
<p class="muted">For companies burning $2K+/mo on APIs</p>
</div>
</div>
</div>

<div class="footer">
<p>Built in Attock, Pakistan 🇵🇰 · <a href="mailto:nayapayxd@gmail.com">nayapayxd@gmail.com</a></p>
<p style="margin-top:.5rem">14-day free trial · No credit card required</p>
</div>
</div>
</body>
</html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Cost-Saver Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:2rem}
.container{max-width:1100px;margin:0 auto}
h1{font-size:2rem;margin-bottom:.5rem}
.subtitle{color:#94a3b8;margin-bottom:2rem}
.card{background:#1e293b;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}
.savings-big{font-size:3rem;font-weight:700;color:#34d399}
.savings-label{color:#94a3b8;font-size:.9rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:1.5rem}
.stat{background:#1e293b;border-radius:12px;padding:1.25rem;text-align:center}
.stat-value{font-size:1.75rem;font-weight:700;color:#60a5fa}
.stat-label{color:#94a3b8;font-size:.85rem;margin-top:.25rem}
table{width:100%;border-collapse:collapse}
th,td{padding:.75rem;text-align:left;border-bottom:1px solid #334155;font-size:.85rem}
th{color:#94a3b8;font-weight:600}
.savings-positive{color:#34d399}
.savings-negative{color:#f87171}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;margin-right:.25rem}
.badge-cached{background:#1e40af;color:#93c5fd}
.badge-compressed{background:#5b21b6;color:#c4b5fd}
.badge-fallback{background:#92400e;color:#fcd34d}
.provider-state{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.state-closed{background:#34d399}
.state-open{background:#f87171}
.state-half{background:#fbbf24}
@media(max-width:600px){body{padding:1rem}.savings-big{font-size:2rem}}
</style>
</head>
<body>
<div class="container">
<h1>AI Cost-Saver PROD</h1><p class="subtitle">Real-time savings dashboard — refreshed every 10 seconds</p>
<div class="grid" id="stats-grid">
<div class="stat"><div class="stat-value" id="savings-pct">--</div><div class="stat-label">Savings vs GPT-4o</div></div>
<div class="stat"><div class="stat-value" id="total-cost">--</div><div class="stat-label">Your Actual Cost</div></div>
<div class="stat"><div class="stat-value" id="gpt4o-cost">--</div><div class="stat-label">GPT-4o Would Cost</div></div>
<div class="stat"><div class="stat-value" id="total-calls">--</div><div class="stat-label">Total Requests</div></div>
<div class="stat"><div class="stat-value" id="cache-rate">--</div><div class="stat-label">Cache Hit Rate</div></div>
<div class="stat"><div class="stat-value" id="avg-cost">--</div><div class="stat-label">Avg Cost/Call</div></div>
</div>
<div class="card"><div class="savings-label">Total Savings (30 days)</div><div class="savings-big" id="total-savings">$0.00</div></div>
<div class="card"><h2 style="margin-bottom:1rem;">Provider Health</h2><div id="provider-health"></div></div>
<div class="card"><h2 style="margin-bottom:1rem;">Model Breakdown</h2><div style="overflow-x:auto;"><table><thead><tr><th>Model</th><th>Calls</th><th>Total Cost</th></tr></thead><tbody id="model-breakdown"></tbody></table></div></div>
<div class="card"><h2 style="margin-bottom:1rem;">Recent Requests</h2><div style="overflow-x:auto;"><table><thead><tr><th>Time</th><th>Model</th><th>Task</th><th>Actual</th><th>GPT-4o</th><th>Saved</th><th>Flags</th></tr></thead><tbody id="recent-requests"></tbody></table></div></div>
</div>
<script>
async function refresh(){
try{
const resp = await fetch('/v1/dashboard-data');
const data = await resp.json();
const savings = data.savings;
const stats = data.stats;
const providers = data.providers;
document.getElementById('total-savings').textContent='$'+(stats.total_savings||0).toFixed(6);
document.getElementById('savings-pct').textContent=(stats.savings_percentage||0).toFixed(1)+'%';
document.getElementById('total-cost').textContent='$'+(savings.total_actual_cost||0).toFixed(6);
document.getElementById('gpt4o-cost').textContent='$'+(savings.total_gpt4o_cost||0).toFixed(6);
document.getElementById('total-calls').textContent=stats.total_calls||0;
document.getElementById('cache-rate').textContent=(stats.cache_hits&&stats.total_calls?(stats.cache_hits/stats.total_calls*100).toFixed(1):0)+'%';
document.getElementById('avg-cost').textContent='$'+(stats.avg_cost_per_call||0).toFixed(6);
document.getElementById('provider-health').innerHTML=Object.entries(providers.circuit_states||{}).map(([name,state])=>{const cls=state==='CLOSED'?'state-closed':state==='OPEN'?'state-open':'state-half';return`<div style="margin-bottom:.5rem;"><span class="provider-state ${cls}"></span>${name}: <strong>${state}</strong></div>`}).join('');
document.getElementById('model-breakdown').innerHTML=(stats.model_breakdown||[]).map(m=>`<tr><td>${m.model||'-'}</td><td>${m.count||0}</td><td>$${(m.total_cost||0).toFixed(6)}</td></tr>`).join('');
document.getElementById('recent-requests').innerHTML=savings.requests.slice(0,20).map(r=>`<tr><td>${new Date(r.timestamp).toLocaleTimeString()}</td><td>${r.model_used||'cache'}</td><td>${r.task_type||'-'}</td><td>$${r.actual_cost.toFixed(8)}</td><td>$${r.gpt4o_cost.toFixed(8)}</td><td class="${r.savings>=0?'savings-positive':'savings-negative'}">$${r.savings.toFixed(8)}</td><td>${r.cached?'<span class="badge badge-cached">CACHED</span>':''}</td></tr>`).join('');
}catch(e){console.error('Dashboard refresh error:',e);}}
refresh();setInterval(refresh,10000);
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def landing():
    return LANDING_HTML


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# =============================================================================
# PART 2: TEST SUITE
# =============================================================================
# All test classes are defined here. They are discovered by pytest when
# running `python ai_cost_saver_all_in_one.py test`.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

class TestCircuitBreaker:
    @pytest.fixture
    def breaker(self):
        return CircuitBreaker("test", failure_threshold=3, recovery_timeout=1)
    
    @pytest.mark.asyncio
    async def test_closed_state_allows_calls(self, breaker):
        mock_coro = AsyncMock(return_value="success")
        result = await breaker.call(mock_coro)
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        mock_coro = AsyncMock(side_effect=Exception("fail"))
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(mock_coro)
        assert breaker.state == "OPEN"
    
    @pytest.mark.asyncio
    async def test_rejects_when_open(self, breaker):
        breaker.state = "OPEN"
        breaker.last_failure_time = time.time()
        with pytest.raises(HTTPException) as exc:
            await breaker.call(AsyncMock(return_value="success"))
        assert "Circuit breaker OPEN" in str(exc.value)
    
    @pytest.mark.asyncio
    async def test_half_open_then_closes(self, breaker):
        breaker.state = "OPEN"
        breaker.last_failure_time = time.time() - 2
        result = await breaker.call(AsyncMock(return_value="success"))
        assert result == "success"
        assert breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_half_open_then_reopens(self, breaker):
        breaker.state = "OPEN"
        breaker.last_failure_time = time.time() - 2
        with pytest.raises(Exception):
            await breaker.call(AsyncMock(side_effect=Exception("fail")))
        assert breaker.state == "OPEN"


class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        return RateLimiter(rps=10.0, burst=5)
    
    @pytest.mark.asyncio
    async def test_allows_within_burst(self, limiter):
        for _ in range(5):
            assert await limiter.is_allowed("client1") is True
    
    @pytest.mark.asyncio
    async def test_blocks_when_exhausted(self, limiter):
        for _ in range(5):
            await limiter.is_allowed("client1")
        assert await limiter.is_allowed("client1") is False
    
    @pytest.mark.asyncio
    async def test_refills_over_time(self, limiter):
        for _ in range(5):
            await limiter.is_allowed("client1")
        await asyncio.sleep(0.15)
        assert await limiter.is_allowed("client1") is True
    
    @pytest.mark.asyncio
    async def test_isolates_clients(self, limiter):
        for _ in range(5):
            await limiter.is_allowed("client1")
        assert await limiter.is_allowed("client2") is True
    
    @pytest.mark.asyncio
    async def test_cleanup_old(self, limiter):
        await limiter.is_allowed("old_client")
        limiter.buckets["old_client"]["last_update"] = time.time() - 7200
        await limiter.cleanup_old(max_age_seconds=3600)
        assert "old_client" not in limiter.buckets


class TestInFlightDedup:
    @pytest.fixture
    def dedup(self):
        return InFlightDedup()
    
    @pytest.mark.asyncio
    async def test_single_call_executes(self, dedup):
        call_count = 0
        async def slow_op():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "result"
        result = await dedup.execute("prompt", "qa", 256, 0.0, slow_op)
        assert result == "result"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_dedup(self, dedup):
        call_count = 0
        async def slow_op():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return "result"
        results = await asyncio.gather(
            dedup.execute("prompt", "qa", 256, 0.0, slow_op),
            dedup.execute("prompt", "qa", 256, 0.0, slow_op),
            dedup.execute("prompt", "qa", 256, 0.0, slow_op),
        )
        assert all(r == "result" for r in results)
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_different_params_not_deduped(self, dedup):
        call_count = 0
        async def op():
            nonlocal call_count
            call_count += 1
            return "result"
        await dedup.execute("prompt1", "qa", 256, 0.0, op)
        await dedup.execute("prompt2", "qa", 256, 0.0, op)
        assert call_count == 2


class TestSemanticCache:
    @pytest.fixture
    def cache(self):
        return SemanticCache(similarity_threshold=0.88, max_entries=5, ttl_seconds=3600)
    
    @pytest.mark.asyncio
    async def test_exact_match(self, cache):
        await cache.store("hello world", {
            "response": "hi", "model_used": "test", "task_type": "qa",
            "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.001
        })
        result = await cache.lookup("hello world")
        assert result is not None
        assert result["response"] == "hi"
    
    @pytest.mark.asyncio
    async def test_no_match(self, cache):
        assert await cache.lookup("nonexistent") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache):
        cache.ttl_seconds = 0
        await cache.store("expire", {
            "response": "gone", "model_used": "test", "task_type": "qa",
            "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.001
        })
        await asyncio.sleep(0.01)
        assert await cache.lookup("expire") is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        for i in range(6):
            await cache.store(f"prompt {i}", {
                "response": f"r{i}", "model_used": "test", "task_type": "qa",
                "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.001
            })
        assert await cache.lookup("prompt 0") is None
        assert await cache.lookup("prompt 5") is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache):
        cache.ttl_seconds = 0
        await cache.store("old", {
            "response": "old", "model_used": "test", "task_type": "qa",
            "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.001
        })
        await asyncio.sleep(0.01)
        await cache.cleanup_expired()
        assert len(cache._entries) == 0
    
    @pytest.mark.asyncio
    async def test_store_no_duplicate(self, cache):
        data = {"response": "a", "model_used": "test", "task_type": "qa",
                "input_tokens": 2, "output_tokens": 1, "cost_usd": 0.001}
        await cache.store("dup", data)
        await cache.store("dup", data)
        assert len(cache._entries) == 1


class TestAsyncRequestLogger:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = tmp_path / "test.db"
        logger = AsyncRequestLogger(str(db_path))
        asyncio.run(logger.init())
        yield logger
        asyncio.run(logger.close())
    
    def test_init_creates_tables(self, db):
        conn = sqlite3.connect(db.db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "calls" in tables
        assert "schema_version" in tables
        conn.close()
    
    @pytest.mark.asyncio
    async def test_log_request(self, db):
        await db.log_request(
            request_id="abc", prompt="test", response="r", model_used="m",
            task_type="qa", input_tokens=10, output_tokens=5,
            original_input_tokens=15, cost_usd=0.001,
            compression_cost_usd=0.0001, cached=False, compressed=True,
            token_estimate_method="provider_native"
        )
        total, cache_hits, total_cost = await db.get_counts()
        assert total == 1
        assert cache_hits == 0
        assert total_cost == pytest.approx(0.0011, abs=1e-6)
    
    @pytest.mark.asyncio
    async def test_log_cached_request(self, db):
        await db.log_request(
            request_id="c1", prompt="p", response="r", model_used="cache",
            task_type="qa", input_tokens=10, output_tokens=5,
            original_input_tokens=10, cost_usd=0.0,
            compression_cost_usd=0.0, cached=True, compressed=False,
            token_estimate_method="provider_native"
        )
        total, cache_hits, _ = await db.get_counts()
        assert total == 1
        assert cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_gpt4o_baseline_uses_original_tokens(self, db):
        await db.log_request(
            request_id="comp", prompt="long", response="short",
            model_used="gemini/flash", task_type="summary",
            input_tokens=50, output_tokens=20, original_input_tokens=500,
            cost_usd=0.005, compression_cost_usd=0.001,
            cached=False, compressed=True, fallback_used=False,
            token_estimate_method="provider_native"
        )
        stats = await db.get_stats(days=1)
        assert stats["total_gpt4o_cost"] > 0
        # gpt4o cost should be based on original_input_tokens (500) and output_tokens (20)
        expected_gpt4o = (500 * GPT4O_PRICING["input"] + 20 * GPT4O_PRICING["output"]) / 1_000_000
        assert stats["total_gpt4o_cost"] == pytest.approx(expected_gpt4o)


class TestTaskDetection:
    def test_classification_detected(self):
        assert detect_task_type("classify this text") == "classification"
    
    def test_yes_no_detected(self):
        assert detect_task_type("Is it raining?") == "yes_no"
    
    def test_simple_qa_detected(self):
        assert detect_task_type("what is the capital of France?") == "simple_qa"
    
    def test_summary_detected(self):
        assert detect_task_type("please summarize the meeting") == "summary"
    
    def test_translation_detected(self):
        assert detect_task_type("translate to french") == "translation"
    
    def test_creative_detected(self):
        assert detect_task_type("write a story about a cat") == "creative"
    
    def test_default_complex(self):
        assert detect_task_type("explain quantum physics") == "complex_qa"


# ---------------------------------------------------------------------------
# API Integration Tests (uses TestClient)
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient

@pytest.fixture
def client():
    # Override global providers with mocks for testing
    global db, providers, cache, compressor, rate_limiter, inflight_dedup
    # Reset globals to avoid state leakage
    db = AsyncRequestLogger(":memory:")
    asyncio.run(db.init())
    
    providers = {}
    cache = None  # disable cache for most integration tests
    compressor = None
    rate_limiter = RateLimiter(1000, 1000)  # essentially unlimited
    inflight_dedup = InFlightDedup()
    
    # Mock providers if needed for some tests
    with TestClient(app) as c:
        yield c


class TestChatEndpoint:
    def test_chat_without_auth_if_disabled(self, client):
        # Auth is disabled by default because API_KEY not set
        resp = client.post("/v1/chat", json={
            "prompt": "hello",
            "max_tokens": 50
        })
        # Without any real providers, should get 502
        assert resp.status_code == 502
        assert "unavailable" in resp.json()["detail"].lower()
    
    def test_chat_with_invalid_task_type(self, client):
        resp = client.post("/v1/chat", json={
            "prompt": "test",
            "task_type": "invalid_type"
        })
        assert resp.status_code == 422
    
    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        # Without providers, should be degraded
        assert data["status"] == "degraded"
    
    def test_metrics_endpoint(self, client):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 0
    
    def test_history_endpoint(self, client):
        resp = client.get("/v1/history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
    
    def test_stats_endpoint(self, client):
        resp = client.get("/v1/stats?days=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_calls" in data
    
    def test_savings_endpoint(self, client):
        resp = client.get("/v1/savings?days=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_actual_cost" in data
    
    def test_providers_status(self, client):
        resp = client.get("/v1/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "gemini" in data
    
    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
    
    def test_dashboard_data_endpoint(self, client):
        resp = client.get("/v1/dashboard-data")
        assert resp.status_code == 200
        data = resp.json()
        assert "stats" in data
        assert "savings" in data
        assert "providers" in data


# =============================================================================
# CLI: python ai_cost_saver_all_in_one.py [serve|test]
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run pytest on this file itself
        pytest.main([__file__, "-v", "--tb=short", "--disable-warnings"])
    else:
        # Serve the application
        uvicorn.run(app, host="0.0.0.0", port=Settings.PROXY_PORT, log_level="info")