import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.service import TTSService


def main():
    parser = argparse.ArgumentParser(description="TTS Latency and Audio Quality Benchmark")
    parser.add_argument("--text", type=str, default="Hello, this is a benchmark test.",
                        help="Text to synthesize during benchmark")
    parser.add_argument("--gender", type=str, default="female",
                        choices=["male", "female", "unknown"],
                        help="Gender for voice selection")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of synthesis iterations to run")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["coqui", "edge", "pyttsx3"],
                        help="Force a specific provider")
    args = parser.parse_args()

    service = TTSService.from_config(force_provider=args.provider)

    latencies = []
    durations = []

    print(f"Benchmark: {args.iterations} iterations")
    print(f"  Text:   {args.text[:60]}...")
    print(f"  Gender: {args.gender}")
    print()

    for i in range(args.iterations):
        start = time.perf_counter()
        result = service.synthesize(
            text=args.text,
            gender=args.gender,
            output_mode="file",
            session_id=f"bench_{i}",
        )
        elapsed = time.perf_counter() - start

        latencies.append(result.latency_ms)
        durations.append(result.duration_ms)

        print(f"  [{i+1}/{args.iterations}] "
              f"status={result.status} "
              f"provider={result.provider_used} "
              f"voice={result.voice_id} "
              f"latency={result.latency_ms:.0f}ms "
              f"audio_duration={result.duration_ms:.0f}ms "
              f"wall_time={elapsed*1000:.0f}ms")

    if latencies:
        print(f"\n  Summary:")
        print(f"    Avg latency:     {sum(latencies)/len(latencies):.0f}ms")
        print(f"    Min latency:     {min(latencies):.0f}ms")
        print(f"    Max latency:     {max(latencies):.0f}ms")
        print(f"    Avg audio dur:   {sum(durations)/len(durations):.0f}ms")


if __name__ == "__main__":
    main()
