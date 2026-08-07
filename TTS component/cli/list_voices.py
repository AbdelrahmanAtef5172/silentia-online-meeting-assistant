import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config_loader import load_config
from engines.coqui_provider import CoquiProvider
from engines.edge_provider import EdgeProvider
from engines.pyttsx3_provider import Pyttsx3Provider


def main():
    parser = argparse.ArgumentParser(description="List available TTS voices by provider")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["coqui", "edge", "pyttsx3"],
                        help="Provider to list voices for (default: all)")
    args = parser.parse_args()

    config = load_config()
    providers = {
        "coqui": CoquiProvider(config["providers"]["coqui"]),
        "edge": EdgeProvider(config["providers"]["edge"]),
        "pyttsx3": Pyttsx3Provider(config["providers"]["pyttsx3"]),
    }

    if args.provider:
        targets = [args.provider]
    else:
        targets = ["coqui", "edge", "pyttsx3"]

    for name in targets:
        p = providers[name]
        print(f"\n{'='*60}")
        print(f"  Provider: {name}")
        print(f"{'='*60}")
        try:
            voices = p.list_voices()
            for gender, voice_list in voices.items():
                print(f"\n  {gender.upper()} ({len(voice_list)} voices):")
                for vid in voice_list:
                    print(f"    - {vid}")
        except Exception as e:
            print(f"  Error listing voices: {e}")


if __name__ == "__main__":
    main()
