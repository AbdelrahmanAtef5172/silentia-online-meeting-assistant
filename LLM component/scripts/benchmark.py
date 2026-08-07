"""
scripts/benchmark.py
────────────────────
Latency, quality, and throughput benchmarks for the LLM correction component.

Usage:
    python scripts/benchmark.py --quality-eval --fixtures tests/fixtures/slr_samples.json
    python scripts/benchmark.py --latency --num-samples 20
    python scripts/benchmark.py --throughput --batch-size 10
"""

import argparse
import json
import time
import sys
import os

from dotenv import load_dotenv

from engine.service import CorrectionService
from engine.schemas import CorrectionContext


def main():
    parser = argparse.ArgumentParser(description="LLM component benchmark")
    parser.add_argument("--quality-eval", action="store_true", help="Run quality evaluation on fixtures")
    parser.add_argument("--fixtures", type=str, default="tests/fixtures/slr_samples.json",
                        help="Path to fixture JSON file")
    parser.add_argument("--latency", action="store_true", help="Run latency benchmark")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples for latency test")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for throughput test")

    args = parser.parse_args()
    load_dotenv()

    if not any([args.quality_eval, args.latency, args.throughput]):
        parser.print_help()
        sys.exit(1)

    env = os.environ.get("ENV", "development")
    service = CorrectionService.from_config(env=env)

    if args.quality_eval:
        run_quality_eval(service, args.fixtures)

    if args.latency:
        run_latency_benchmark(service, args.num_samples)

    if args.throughput:
        run_throughput_benchmark(service, args.batch_size)


def run_quality_eval(service, fixtures_path: str):
    print("Running quality evaluation...")

    with open(fixtures_path) as f:
        samples = json.load(f)

    pass_count = 0
    total = len(samples)
    latencies = []

    for sample in samples:
        context = CorrectionContext(topic_domain=sample.get("domain")) if sample.get("domain") else None
        start = time.time()
        result = service.correct(sample["input"], context=context)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        if result.status.value in ("success", "cached") and result.corrected_text:
            pass_count += 1
        elif result.status.value == "passthrough":
            pass_count += 1
        else:
            print(f"  FAIL: [{sample['id']}] {sample['input'][:50]}... -> {result.status.value}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    score = (pass_count / total * 100) if total > 0 else 0

    print(f"\nQuality score:   {score:.1f}% ({pass_count}/{total} pass)")
    print(f"Avg latency:     {avg_latency:.0f}ms")
    print(f"Cache hit rate:  {service._cache.stats()['hit_rate'] * 100:.0f}%")


def run_latency_benchmark(service, num_samples: int):
    print(f"Running latency benchmark ({num_samples} samples)...")

    test_inputs = [
        "she go store yesterday buy apple",
        "patient show symptom high fever three day",
        "i want eat pizza my friend",
        "he go school every day",
        "they meet yesterday morning discuss project",
    ]

    latencies = []
    for i in range(num_samples):
        text = test_inputs[i % len(test_inputs)]
        start = time.time()
        result = service.correct(text)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        status = "OK" if result.status.value in ("success", "cached", "passthrough") else "FAIL"
        print(f"  [{status}] sample {i + 1}: {elapsed:.0f}ms (provider={result.provider_used})")

    avg = sum(latencies) / len(latencies)
    print(f"\nAverage latency:  {avg:.0f}ms")
    print(f"Min latency:      {min(latencies):.0f}ms")
    print(f"Max latency:      {max(latencies):.0f}ms")
    print(f"P95 latency:      {sorted(latencies)[int(len(latencies) * 0.95)]:.0f}ms")


def run_throughput_benchmark(service, batch_size: int):
    print(f"Running throughput benchmark (batch size = {batch_size})...")

    test_inputs = [
        "she go store yesterday buy apple",
        "patient show symptom high fever three day",
        "i want eat pizza my friend",
        "he go school every day",
        "they meet yesterday morning discuss project",
        "doctor prescribe medicine patient",
        "i need go hospital",
        "she make cake yesterday",
        "they play football tomorrow",
        "he drink coffee every morning",
    ]

    texts = [test_inputs[i % len(test_inputs)] for i in range(batch_size)]

    start = time.time()
    results = service.correct_batch(texts, max_concurrency=4)
    elapsed = (time.time() - start) * 1000

    success_count = sum(1 for r in results if r.status.value in ("success", "cached", "passthrough"))
    throughput = batch_size / (elapsed / 1000)

    print(f"Completed:        {success_count}/{batch_size} successful")
    print(f"Total time:       {elapsed:.0f}ms")
    print(f"Throughput:       {throughput:.1f} corrections/second")


if __name__ == "__main__":
    main()
