"""Unit tests for ResponseCache — LRU eviction, key normalization, TTL."""

import time
import pytest
from engine.response_cache import ResponseCache
from engine.schemas import CorrectionResult, CorrectionStatus


@pytest.fixture
def cache():
    return ResponseCache(max_size=10, ttl_seconds=3600)


def make_result(text: str, status=CorrectionStatus.SUCCESS) -> CorrectionResult:
    return CorrectionResult(
        corrected_text=text,
        original_text=text,
        status=status,
    )


def test_get_and_set(cache):
    result = make_result("She went to the store.")
    cache.set("she go store", result)
    cached = cache.get("she go store")
    assert cached is not None
    assert cached.corrected_text == "She went to the store."


def test_cache_miss_returns_none(cache):
    result = cache.get("nonexistent input")
    assert result is None


def test_key_normalization(cache):
    result = make_result("She went to the store.")
    cache.set("she go store", result)
    assert cache.get("SHE GO STORE") is not None
    assert cache.get("  she go store  ") is not None
    assert cache.get("she  go  store") is not None


def test_lru_eviction(cache):
    for i in range(15):
        cache.set(f"input {i}", make_result(f"Output {i}"))
    stats = cache.stats()
    assert stats["size"] <= 10


def test_ttl_expiry():
    short_cache = ResponseCache(max_size=10, ttl_seconds=0.1)
    result = make_result("She went to the store.")
    short_cache.set("she go store", result)
    assert short_cache.get("she go store") is not None
    time.sleep(0.15)
    assert short_cache.get("she go store") is None


def test_no_ttl_expiry():
    no_ttl_cache = ResponseCache(max_size=10, ttl_seconds=0)
    result = make_result("She went to the store.")
    no_ttl_cache.set("she go store", result)
    assert no_ttl_cache.get("she go store") is not None


def test_does_not_cache_failed_results(cache):
    result = make_result("", status=CorrectionStatus.FAILED)
    cache.set("she go store", result)
    assert cache.get("she go store") is None


def test_invalidate(cache):
    result = make_result("She went to the store.")
    cache.set("she go store", result)
    cache.invalidate("she go store")
    assert cache.get("she go store") is None


def test_clear(cache):
    cache.set("input 1", make_result("Output 1"))
    cache.set("input 2", make_result("Output 2"))
    cache.clear()
    stats = cache.stats()
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_stats_tracking(cache):
    assert cache.stats()["hit_rate"] == 0.0
    result = make_result("She went to the store.")
    cache.set("she go store", result)

    cache.get("she go store")
    cache.get("nonexistent")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_normalization_different_whitespace(cache):
    result = make_result("Output")
    cache.set("hello   world", result)
    assert cache.get("hello world") is not None
    assert cache.get("  hello  world  ") is not None


def test_set_updates_existing_key(cache):
    result1 = make_result("First correction")
    result2 = make_result("Updated correction")
    cache.set("same input", result1)
    cache.set("same input", result2)
    cached = cache.get("same input")
    assert cached.corrected_text == "Updated correction"
