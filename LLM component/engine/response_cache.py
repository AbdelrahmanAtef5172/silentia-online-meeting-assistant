"""
cache/response_cache.py
────────────────────────
Stores correction results to avoid redundant API calls for identical or
near-identical inputs.

Two-level cache:
  L1 — Exact cache: SHA-256 hash of normalized input → CorrectionResult.
       In-memory LRU with configurable max size.
  L2 — Semantic cache: Not implemented in v1.0 (future optimization).
"""

import hashlib
import time
import re
from typing import Optional

from cachetools import LRUCache

from engine.schemas import CorrectionResult, CorrectionStatus


class ResponseCache:
    """
    In-memory LRU cache for correction results.

    Cache key normalization (applied before hashing):
      - Strip leading/trailing whitespace
      - Collapse multiple spaces to single space
      - Lowercase the entire string

    What is NOT cached:
      - Inputs with status="failed"
      - Results with quality="fallback"  (not yet known at cache level)
      - Results when the provider timed out
    """

    def __init__(self, max_size: int = 512, ttl_seconds: int = 3600):
        self._cache: LRUCache = LRUCache(maxsize=max_size)
        self._ttl = ttl_seconds
        self._timestamps: dict[str, float] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, raw_text: str) -> Optional[CorrectionResult]:
        """Return cached result or None."""
        key = self._normalize_key(raw_text)
        if key not in self._cache:
            self._misses += 1
            return None

        if self._is_expired(key):
            self._evictions += 1
            del self._cache[key]
            del self._timestamps[key]
            self._misses += 1
            return None

        self._hits += 1
        return self._cache[key]

    def set(self, raw_text: str, result: CorrectionResult) -> None:
        """Store result. Evicts LRU entry if at capacity."""
        if result.status == CorrectionStatus.FAILED:
            return

        key = self._normalize_key(raw_text)
        self._cache[key] = result
        self._timestamps[key] = time.time()

    def invalidate(self, raw_text: str) -> None:
        """Remove a specific entry (for testing/debugging)."""
        key = self._normalize_key(raw_text)
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]

    def clear(self) -> None:
        """Wipe the entire cache. Use between sessions if needed."""
        self._cache.clear()
        self._timestamps.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> dict:
        """Return hit_rate, size, evictions for monitoring."""
        total = self._hits + self._misses
        return {
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._cache.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    @staticmethod
    def _normalize_key(text: str) -> str:
        normalized = text.strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _is_expired(self, key: str) -> bool:
        if self._ttl <= 0:
            return False
        age = time.time() - self._timestamps.get(key, 0)
        return age > self._ttl
