"""
Simple cache test for CorrectionService.

Run:
    python test_cache.py
"""

import time
from engine.service import CorrectionService


def print_result(title, result):
    print(f"\n{title}")
    print("-" * 40)
    print(f"Corrected : {result.text_for_tts}")
    print(f"Status    : {result.status.value}")
    print(f"Provider  : {result.provider_used}")
    print(f"Cache     : {'HIT' if result.from_cache else 'MISS'}")
    print(f"Latency   : {result.latency_ms:.2f} ms")


def main():
    service = CorrectionService.from_config()

    text = "she go store yesterday"

    print(f"Input: {text}")

    # First request
    start = time.perf_counter()
    result1 = service.correct(text)
    elapsed1 = (time.perf_counter() - start) * 1000
    print_result("FIRST REQUEST", result1)
    print(f"Actual execution time: {elapsed1:.2f} ms")

    # Second request
    start = time.perf_counter()
    result2 = service.correct(text)
    elapsed2 = (time.perf_counter() - start) * 1000
    print_result("SECOND REQUEST", result2)
    print(f"Actual execution time: {elapsed2:.2f} ms")

    print("\nCache Statistics")
    print("-" * 40)
    print(service._cache.stats())


if __name__ == "__main__":
    main()