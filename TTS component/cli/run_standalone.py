import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.service import TTSService


def main():
    parser = argparse.ArgumentParser(description="TTS Component — Standalone CLI")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--gender", type=str, default="unknown",
                        choices=["male", "female", "unknown"],
                        help="Speaker gender for voice selection")
    parser.add_argument("--output", type=str, default=None, help="Output file path (mode=file)")
    parser.add_argument("--mode", type=str, default="play", choices=["play", "file"],
                        help="Output mode: play audio or save to file")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["coqui", "edge"],
                        help="Force a specific TTS provider")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (0.5–2.0)")
    parser.add_argument("--session", type=str, default=None, help="Session ID for file naming")
    parser.add_argument("--repl", action="store_true",
                        help="Interactive mode — text prompt stays alive, model loads once")
    args = parser.parse_args()

    if args.repl:
        _run_repl(args)
    else:
        _run_once(args)


def _run_once(args):
    if not args.text:
        print("Error: --text is required (use --repl for interactive mode)")
        sys.exit(1)

    service = TTSService.from_config(force_provider=args.provider)

    result = service.synthesize(
        text=args.text,
        gender=args.gender,
        output_mode=args.mode,
        output_path=args.output,
        session_id=args.session,
        speed=args.speed,
    )

    print(result.to_dict())


def _run_repl(args):
    service = TTSService.from_config(force_provider=args.provider)

    print("TTS REPL — type text and press Enter. Empty line or Ctrl+C to exit.")

    if args.text:
        result = service.synthesize(
            text=args.text,
            gender=args.gender,
            output_mode=args.mode,
            output_path=args.output,
            session_id=args.session,
            speed=args.speed,
        )
        print(result.to_dict())

    while True:
        try:
            text = input("Text> ").strip()
            if not text:
                break
        except (EOFError, KeyboardInterrupt):
            print()
            break

        result = service.synthesize(
            text=text,
            gender=args.gender,
            output_mode=args.mode,
            session_id=args.session,
            speed=args.speed,
        )
        print(result.to_dict())


if __name__ == "__main__":
    main()
