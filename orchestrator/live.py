"""
live.py — Live webcam pipeline for Silentia

Captures frames from webcam, processes them through Vision and SLR in real-time,
and when a complete sentence is recognized, runs LLM correction + TTS speech.

Usage:
    ../SLR\ component/.venv/bin/python live.py [--webcam 0] [--no-preview]
"""

import os

# Force CPU — must be set before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VISION_ROOT = os.path.join(PROJECT_ROOT, "Vision component")
SLR_ROOT = os.path.join(PROJECT_ROOT, "SLR component")
LLM_ROOT = os.path.join(PROJECT_ROOT, "LLM component")
TTS_ROOT = os.path.join(PROJECT_ROOT, "TTS component")

sys.path.insert(0, VISION_ROOT)
sys.path.insert(0, SLR_ROOT)
sys.path.insert(0, os.path.join(SLR_ROOT, "src"))

import cv2
import numpy as np

from engine.component import GenderDetectionComponent
from engine.schemas import GenderLabel
from src.inference import ContinuousSignRecognizer, ContinuousConfig


class LiveSilentiaPipeline:
    def __init__(self, webcam_id: int = 0, show_preview: bool = True):
        self.webcam_id = webcam_id
        self.show_preview = show_preview

        # ---- Vision ----
        print("[live] Initializing Vision...", flush=True)
        self.vision = GenderDetectionComponent.from_config(
            path=os.path.join(VISION_ROOT, "configs", "config.yaml"),
            env="development",
        )

        # ---- SLR (continuous) ----
        print("[live] Initializing SLR...", flush=True)

        def on_word(word: str, confidence: float) -> None:
            print(f"  + word: {word}  ({confidence:.2f})", flush=True)

        def on_sentence(sentence: str) -> None:
            print(f"\n>>> SENTENCE: \"{sentence}\"", flush=True)
            self._run_llm_tts(sentence)

        self.slr = ContinuousSignRecognizer(
            checkpoint_path=os.path.join(SLR_ROOT, "models", "best_model.pt"),
            label_map_path=os.path.join(SLR_ROOT, "config", "label_map.json"),
            models_dir=os.path.join(SLR_ROOT, "models"),
            device="cpu",
            on_word_recognized=on_word,
            on_sentence_complete=on_sentence,
        )

        self.last_gender: str = "unknown"
        self.last_confidence: float = 0.0
        self.frame_idx = 0
        self._stop = False

    # ------------------------------------------------------------------
    def _run_llm_tts(self, slr_text: str) -> None:
        llm_py = os.path.join(LLM_ROOT, ".venv", "bin", "python")
        llm_script = os.path.join(LLM_ROOT, "scripts", "run_standalone.py")
        tts_py = os.path.join(TTS_ROOT, ".venv", "bin", "python")
        tts_script = os.path.join(TTS_ROOT, "cli", "run_standalone.py")

        # --- LLM ---
        print("  -> LLM correcting...", flush=True)
        try:
            proc = subprocess.run(
                [llm_py, llm_script, "--text", slr_text],
                cwd=LLM_ROOT,
                capture_output=True,
                text=True,
                timeout=60,
            )
            corrected = proc.stdout.strip()
            if not corrected:
                corrected = slr_text
            print(f"  -> Corrected: \"{corrected}\"", flush=True)
        except Exception as e:
            print(f"  -> LLM error: {e}", flush=True)
            corrected = slr_text

        # --- TTS ---
        gender = self.last_gender if self.last_gender != "no_face" else "unknown"
        print(f"  -> TTS speaking (gender={gender})...", flush=True)
        try:
            subprocess.run(
                [tts_py, tts_script, "--text", corrected, "--gender", gender, "--mode", "play"],
                cwd=TTS_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as e:
            print(f"  -> TTS error: {e}", flush=True)
        print("  -> Done\n", flush=True)

    # ------------------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self.webcam_id)
        if not cap.isOpened():
            print(f"[live] ERROR: cannot open webcam {self.webcam_id}", flush=True)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"[live] Webcam {self.webcam_id} opened.  Press 'q' to quit.\n", flush=True)

        while not self._stop:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- Vision ----
            vis_result = self.vision.process_frame(frame, self.frame_idx, timestamp=self.frame_idx / fps)
            label = vis_result.label.value if hasattr(vis_result.label, 'value') else vis_result.label
            if label != "no_face":
                self.last_gender = label
                self.last_confidence = vis_result.confidence

            # ---- SLR ----
            slr_status = self.slr.process_frame(frame)

            # ---- Preview ----
            if self.show_preview:
                self._draw_overlay(frame, slr_status)
                cv2.imshow("Silentia Live — press 'q' to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._stop = True

            self.frame_idx += 1

        # ---- Cleanup ----
        self.slr.flush()
        cap.release()
        cv2.destroyAllWindows()
        self.slr.close()
        print("[live] Stopped.", flush=True)

    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, slr_status: dict) -> None:
        h, w = frame.shape[:2]

        def put_text(img, text, pos, scale, color, thickness=2):
            x, y = pos
            cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), thickness + 1)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness)

        # Gender
        gender_color = (0, 255, 0) if self.last_gender == "male" else (0, 200, 255) if self.last_gender == "female" else (100, 100, 100)
        put_text(frame, f"Gender: {self.last_gender.upper()}  ({self.last_confidence:.2f})", (14, 30), 0.55, gender_color)

        # Recognized words / sentence
        words = " ".join(self.slr.current_words) if self.slr.current_words else "(waiting for signs...)"
        put_text(frame, "Signs:", (14, 58), 0.50, (180, 180, 180))
        put_text(frame, words, (92, 58), 0.50, (0, 255, 255))

        # Status
        state = slr_status.get("state", "?")
        state_colors = {
            "collecting": (255, 200, 0), "word_recognized": (0, 255, 0),
            "sentence_complete": (0, 200, 255), "idle": (100, 100, 100),
        }
        sc = state_colors.get(state, (200, 200, 200))
        put_text(frame, f"Status: {state}", (14, 86), 0.45, sc)

        # Frame count in corner
        put_text(frame, f"frame: {self.frame_idx}", (w - 120, h - 14), 0.40, (120, 120, 120))


# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Silentia Live Webcam Pipeline")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    args = parser.parse_args()

    pipe = LiveSilentiaPipeline(webcam_id=args.webcam, show_preview=not args.no_preview)
    pipe.run()


if __name__ == "__main__":
    main()
