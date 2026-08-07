"""
scripts/run_standalone.py
─────────────────────────
CLI entry point for standalone mode.

Usage:
    echo "she go store yesterday" | python scripts/run_standalone.py
    python scripts/run_standalone.py --text "she go store yesterday"
    python scripts/run_standalone.py --input path/to/slr_output.txt --output corrected.txt
    python scripts/run_standalone.py --text "patient show symptom high fever" --domain medical --verbose
    python scripts/run_standalone.py --batch path/to/sentences.json --output corrections.json
    ENV=production python scripts/run_standalone.py --text "i want go home"
"""

import argparse
import json
import sys
import os

from dotenv import load_dotenv

from engine.service import CorrectionService
from engine.schemas import CorrectionContext


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="LLM grammar correction for SLR pipeline output"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, help="Single text to correct")
    input_group.add_argument("--input", type=str, help="Input file path (one sentence per line)")
    input_group.add_argument("--batch", type=str, help="JSON file with list of sentences")

    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--domain", type=str, default=None, help="Topic domain (medical, legal, casual)")
    parser.add_argument("--verbose", action="store_true", help="Show provider, latency, cache status")

    args = parser.parse_args()

    env = os.environ.get("ENV", "development")
    service = CorrectionService.from_config(env=env)

    if args.text:
        results = [_process_single(service, args.text, args.domain, args.verbose)]
        _output_results(results, args.output, args.verbose)
    elif args.input:
        with open(args.input) as f:
            texts = [line.strip() for line in f if line.strip()]
        results = [_process_single(service, t, args.domain, args.verbose) for t in texts]
        _output_results(results, args.output, args.verbose)
    elif args.batch:
        with open(args.batch) as f:
            data = json.load(f)
        texts = data if isinstance(data, list) else data.get("sentences", [])
        results = service.correct_batch(texts)
        _output_batch(results, args.output)
    else:
        texts = [line.strip() for line in sys.stdin if line.strip()]
        if not texts:
            parser.print_help()
            sys.exit(1)
        results = [_process_single(service, t, args.domain, args.verbose) for t in texts]
        _output_results(results, args.output, args.verbose)


def _process_single(service, text: str, domain: str = None, verbose: bool = False):
    context = CorrectionContext(topic_domain=domain) if domain else None
    result = service.correct(text, context=context)
    if verbose:
        status_icon = "✓" if result.status.value in ("success", "cached", "passthrough") else "✗"
        print(
            f"[{status_icon}] status={result.status.value} "
            f"provider={result.provider_used or 'none'} "
            f"latency={result.latency_ms:.0f}ms "
            f"cache={'hit' if result.from_cache else 'miss'}",
            file=sys.stderr,
        )
        if result.warning:
            print(f"[!] warning: {result.warning}", file=sys.stderr)
    return result


def _output_results(results, output_path: str = None, verbose: bool = False):
    lines = [r.text_for_tts for r in results]
    output = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(output + "\n")
        if verbose:
            print(f"Written to {output_path}", file=sys.stderr)
    else:
        print(output)


def _output_batch(results, output_path: str = None):
    data = [r.to_dict() for r in results]
    output = json.dumps(data, indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
