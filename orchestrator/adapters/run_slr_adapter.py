"""
Adapter script for the SLR component (the only component whose CLI
lacks structured JSON output).  Called by the orchestrator via
subprocess with the SLR component's own .venv Python.

Supports both single-clip and continuous recognition modes.

Usage:
    <slr_venv>/bin/python run_slr_adapter.py \
        --video path/to/video.mp4 \
        --checkpoint path/to/best_model.pt \
        --label-map path/to/label_map.json \
        --models-dir path/to/models \
        --device cpu \
        --mode single \
        --output /tmp/slr_result.json
"""

import json
import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "SLR component"))

parser = argparse.ArgumentParser(description="SLR adapter — structured JSON output")
parser.add_argument("--video", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label-map", default=None)
parser.add_argument("--models-dir", default="models")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--mode", default="single", choices=["single", "continuous"])
parser.add_argument("--output", required=True)
parser.add_argument("--component-root", default=None)
args = parser.parse_args()

component_root = args.component_root or os.environ.get("SLR_ROOT") or DEFAULT_ROOT
component_root = os.path.abspath(component_root)
sys.path.insert(0, component_root)
sys.path.insert(0, os.path.join(component_root, "src"))
os.chdir(component_root)

# ---------------------------------------------------------------------------
# Single-clip mode
# ---------------------------------------------------------------------------
def _run_single() -> dict:
    from src.sign_language_model import SignLanguageRecognizer

    recognizer = SignLanguageRecognizer(
        checkpoint_path=os.path.abspath(args.checkpoint),
        label_map_path=os.path.abspath(args.label_map) if args.label_map else None,
        models_dir=os.path.abspath(args.models_dir),
        device=args.device,
    )

    if not recognizer.ready:
        return {"success": False, "error": recognizer.init_error}

    result = recognizer.predict(args.video, top_k=1)
    recognizer.close()
    return result


# ---------------------------------------------------------------------------
# Continuous / sentence mode
# ---------------------------------------------------------------------------
def _run_continuous() -> dict:
    import cv2
    from src.inference import ContinuousSignRecognizer

    output = {"success": False, "sentence": "", "words": []}

    def on_word(word: str, confidence: float) -> None:
        output["words"].append({"word": word, "confidence": confidence})

    def on_sentence(sentence: str) -> None:
        output["sentence"] = sentence
        output["success"] = True

    recognizer = ContinuousSignRecognizer(
        checkpoint_path=os.path.abspath(args.checkpoint),
        label_map_path=os.path.abspath(args.label_map) if args.label_map else None,
        models_dir=os.path.abspath(args.models_dir),
        device=args.device,
        on_word_recognized=on_word,
        on_sentence_complete=on_sentence,
    )

    if not recognizer.ready:
        return {"success": False, "error": recognizer.init_error}

    cap = cv2.VideoCapture(args.video)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            recognizer.process_frame(frame)
    finally:
        cap.release()

    recognizer.flush()
    recognizer.close()

    # If no sentence was detected via end-of-utterance pause, use whatever
    # words were accumulated as the sentence.
    if not output["success"] and output["words"]:
        output["sentence"] = " ".join(w["word"] for w in output["words"])
        output["success"] = bool(output["words"])

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
result = _run_continuous() if args.mode == "continuous" else _run_single()

with open(os.path.abspath(args.output), "w") as f:
    json.dump(result, f, indent=2)

if not result.get("success"):
    print(f"SLR prediction failed: {result.get('error', 'unknown error')}", file=sys.stderr)
    sys.exit(1)

print(f"SLR result saved to {args.output}", file=sys.stderr)
