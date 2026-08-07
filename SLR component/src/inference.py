"""
inference.py
=============

Standalone inference entry point for the WLASL-100 Sign Language Recognition component.

Two usage modes:

1. Single-clip prediction (a pre-trimmed video of one sign):

    python inference.py --checkpoint models/best_model.pt --video path/to/clip.mp4

2. Continuous mode (a longer video, or a live webcam feed, containing a sequence of signs
   separated by short pauses) -- builds up a sentence word-by-word and flushes it to the
   downstream LLM refinement component when a longer pause (end-of-utterance) is detected:

    python inference.py --checkpoint models/best_model.pt --video path/to/sentence.mp4 --continuous
    python inference.py --checkpoint models/best_model.pt --webcam --continuous

Both modes share the same underlying `SignLanguageRecognizer` from `sign_language_model.py`
and follow the same never-raises contract: errors are reported in the returned dict / printed
as a clear message, the process does not crash on bad input.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from sign_language_model import SignLanguageRecognizer, KeypointExtractor, ensure_mediapipe_models


# ========================================================================================
# CONTINUOUS / STREAMING SENTENCE-BUILDING RECOGNIZER
# ========================================================================================

@dataclass
class ContinuousConfig:
    pause_frames_threshold: int = 8       # consecutive no-hand frames -> end current sign segment
    sentence_end_frames_threshold: int = 45  # consecutive no-hand frames -> end whole utterance
    min_segment_frames: int = 10          # segments shorter than this are discarded as noise
    max_segment_frames: int = 90          # force-finalize a segment even without a pause
    confidence_threshold: float = 0.30    # predictions below this are dropped, not appended
    suppress_repeated_words: bool = True  # avoid appending the same word twice in a row
    max_sentence_words: int = 50          # auto-finalize sentence when this many words accumulate
    target_fps: int = 30                  # assumed fps for MediaPipe timestamp calculation


class ContinuousSignRecognizer:
    """Consumes a video/live stream frame-by-frame, segments it into individual signs using a
    hand-presence pause heuristic, classifies each segment with `SignLanguageRecognizer`, and
    assembles recognized glosses into a sentence buffer. On detecting a longer pause
    (end-of-utterance), the accumulated sentence is flushed via `on_sentence_complete`.

    Important limitation (by design): WLASL100 is an *isolated-sign* dataset -- the trained
    model has only ever seen single, separately-performed signs, not natural continuously
    co-articulated signing (where adjacent signs blend into each other with no pause). This
    pause-based segmentation is therefore a practical heuristic for "signer pauses briefly
    between words," not a solution to full continuous sign language recognition (CSLR), which
    is a substantially harder research problem typically addressed with CTC-style sequence
    models trained on continuous data (e.g. RWTH-PHOENIX). Expect this to work well for
    deliberate, slightly-paced signing and to degrade on fast, fluent, blended signing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        label_map_path: Optional[str] = None,
        models_dir: str = "models",
        device: str = "cpu",
        config: Optional[ContinuousConfig] = None,
        frame_step: int = 1,
        on_word_recognized: Optional[Callable[[str, float], None]] = None,
        on_sentence_complete: Optional[Callable[[str], None]] = None,
    ):
        self.recognizer = SignLanguageRecognizer(
            checkpoint_path=checkpoint_path,
            label_map_path=label_map_path,
            models_dir=models_dir,
            device=device,
            create_extractor=False,
        )
        self.cfg = config or ContinuousConfig()
        self.on_word_recognized = on_word_recognized
        self.on_sentence_complete = on_sentence_complete

        self.ready = self.recognizer.ready
        self.init_error = self.recognizer.init_error

        # streaming state
        self._segment_features: List[np.ndarray] = []
        self._no_hand_run = 0
        self._sentence_words: List[str] = []
        self._extractor: Optional[KeypointExtractor] = None
        self._frame_idx = 0
        self._frame_step = max(1, frame_step)
        self._skip_counter = 0
        self._fps = max(1, self.cfg.target_fps)

        if self.ready:
            try:
                pose_path, hand_path = ensure_mediapipe_models(models_dir)
                self._extractor = KeypointExtractor(pose_path, hand_path)
            except Exception as e:
                self.ready = False
                self.init_error = f"{type(e).__name__}: {e}"

    # ---- internal --------------------------------------------------------------------

    def _safe_callback(self, fn: Optional[Callable], *args) -> None:
        """Callbacks are invoked defensively: a bug in the caller's LLM/TTS glue code must
        not be allowed to crash the recognition loop."""
        if fn is None:
            return
        try:
            fn(*args)
        except Exception as e:
            print(f"[ContinuousSignRecognizer] on_*_complete callback raised: {e}", file=sys.stderr)

    def _finalize_segment(self) -> Dict[str, Any]:
        """Runs the classifier on the buffered segment (if long enough) and, if confident and
        not a repeat, appends the recognized word to the sentence buffer."""
        status: Dict[str, Any] = {"state": "no_segment"}
        n_frames = len(self._segment_features)
        if n_frames >= self.cfg.min_segment_frames:
            feature_seq = np.stack(self._segment_features, axis=0)
            result = self.recognizer.predict_from_features(feature_seq, top_k=1)
            if result["success"] and result["confidence"] >= self.cfg.confidence_threshold:
                word = result["predicted_class"]
                is_repeat = (
                    self.cfg.suppress_repeated_words
                    and self._sentence_words
                    and self._sentence_words[-1] == word
                )
                if not is_repeat:
                    self._sentence_words.append(word)
                    self._safe_callback(self.on_word_recognized, word, result["confidence"])
                    if len(self._sentence_words) >= self.cfg.max_sentence_words:
                        sentence = self._finalize_sentence()
                        status = {"state": "sentence_complete", "sentence": sentence}
                    else:
                        status = {"state": "word_recognized", "word": word, "confidence": result["confidence"]}
                else:
                    status = {"state": "repeat_suppressed", "word": word}
            elif not result["success"]:
                status = {"state": "prediction_error", "error": result.get("error")}
            else:
                status = {"state": "low_confidence_discarded", "result": result}
        else:
            status = {"state": "segment_too_short", "frames": n_frames}
        self._segment_features = []
        return status

    def _finalize_sentence(self) -> Optional[str]:
        if not self._sentence_words:
            return None
        sentence = " ".join(self._sentence_words)
        self._sentence_words = []
        self._safe_callback(self.on_sentence_complete, sentence)
        return sentence

    # ---- public API --------------------------------------------------------------------

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """Feed a single BGR frame (e.g. from cv2.VideoCapture). Never raises -- returns a
        status dict describing what happened on this frame:
          state in {"collecting", "word_recognized", "repeat_suppressed",
                    "low_confidence_discarded", "prediction_error", "segment_too_short",
                    "sentence_complete", "idle", "skipped"}
        """
        if not self.ready:
            return {"state": "error", "error": f"Recognizer not initialized: {self.init_error}"}

        # Skip extraction on this frame if frame_step > 1
        self._skip_counter += 1
        if self._skip_counter < self._frame_step:
            return {"state": "skipped"}
        self._skip_counter = 0

        try:
            ms_per_frame = 1000.0 / self._fps
            kp = self._extractor.extract_from_frame(frame_bgr, int(self._frame_idx * ms_per_frame))
            self._frame_idx += self._frame_step

            if kp.hand_present:
                self._no_hand_run = 0
                self._segment_features.append(kp.features)
                if len(self._segment_features) >= self.cfg.max_segment_frames:
                    result = self._finalize_segment()
                    return result
                return {"state": "collecting", "frames": len(self._segment_features)}

            # no hand detected this frame
            self._no_hand_run += self._frame_step

            if self._segment_features and self._no_hand_run >= self.cfg.pause_frames_threshold:
                result = self._finalize_segment()
                return result

            if self._no_hand_run >= self.cfg.sentence_end_frames_threshold and self._sentence_words:
                sentence = self._finalize_sentence()
                return {"state": "sentence_complete", "sentence": sentence}

            return {"state": "idle"}
        except Exception as e:
            return {"state": "error", "error": f"{type(e).__name__}: {e}"}

    def flush(self) -> Dict[str, Any]:
        """Force-finalize any pending segment and sentence (e.g. at end-of-video). Never raises."""
        if not self.ready:
            return {"state": "error", "error": f"Recognizer not initialized: {self.init_error}"}
        try:
            if self._segment_features:
                self._finalize_segment()
            sentence = self._finalize_sentence()
            return {"state": "sentence_complete", "sentence": sentence} if sentence else {"state": "idle"}
        except Exception as e:
            return {"state": "error", "error": f"{type(e).__name__}: {e}"}

    @property
    def current_words(self) -> List[str]:
        return list(self._sentence_words)

    def close(self):
        if self._extractor is not None:
            self._extractor.close()
        self.recognizer.close()


# ========================================================================================
# CLI
# ========================================================================================

def _run_single_clip(args) -> None:
    import cv2  # noqa: F401  (validates opencv is importable before we report a clean error)

    recognizer = SignLanguageRecognizer(
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        models_dir=args.models_dir,
        device=args.device,
    )
    if not recognizer.ready:
        print(f"ERROR: could not initialize model: {recognizer.init_error}", file=sys.stderr)
        sys.exit(1)

    if not args.video:
        print("ERROR: --video is required in single-clip mode (omit --continuous to use it).",
              file=sys.stderr)
        sys.exit(1)

    result = recognizer.predict(args.video, top_k=args.top_k)
    recognizer.close()

    if not result["success"]:
        print(f"Prediction failed: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Predicted sign: {result['predicted_class']}  (confidence: {result['confidence']:.3f})")
    if args.top_k > 1:
        print("Top-k:")
        for entry in result["top_k"]:
            print(f"  {entry['class']:<20s} {entry['confidence']:.3f}")


def _run_continuous(args) -> None:
    import cv2

    last_word = ""
    last_confidence = 0.0
    last_completed_sentence = ""

    def print_word(word: str, confidence: float) -> None:
        nonlocal last_word, last_confidence
        last_word = word
        last_confidence = confidence
        print(f"  + word: {word}  ({confidence:.2f})")

    def print_sentence(sentence: str) -> None:
        nonlocal last_completed_sentence
        last_completed_sentence = sentence
        print(f"\n>>> SENTENCE COMPLETE: \"{sentence}\"\n"
              f"    (hand this string to the LLM refinement component)\n")

    recognizer = ContinuousSignRecognizer(
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        models_dir=args.models_dir,
        device=args.device,
        on_word_recognized=print_word,
        on_sentence_complete=print_sentence,
    )
    if not recognizer.ready:
        print(f"ERROR: could not initialize model: {recognizer.init_error}", file=sys.stderr)
        sys.exit(1)

    source = 0 if args.webcam else args.video
    if source is None:
        print("ERROR: provide --video PATH or --webcam for continuous mode.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open video source: {source}", file=sys.stderr)
        sys.exit(1)

    window_name = "SLR Live Prediction"
    print("Processing stream... (press 'q' to quit)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            status = recognizer.process_frame(frame)
            h, w = frame.shape[:2]

            sentence = " ".join(recognizer.current_words) if recognizer.current_words else "(waiting for signs...)"

            def put_text(img, text, pos, scale, color, thickness=1):
                x, y = pos
                cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), thickness + 1)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, thickness)

            y = 28
            put_text(frame, "SENTENCE:", (14, y), 0.55, (180, 180, 180))
            put_text(frame, sentence, (150, y), 0.55, (0, 255, 255))
            y += 30

            if last_completed_sentence:
                put_text(frame, "COMPLETED:", (14, y), 0.55, (180, 180, 180))
                put_text(frame, last_completed_sentence, (150, y), 0.55, (0, 255, 0))
                y += 30

            if last_word:
                put_text(frame, "WORD:", (14, y), 0.55, (180, 180, 180))
                put_text(frame, f"{last_word}  ({last_confidence:.2f})", (150, y),
                         0.55, (0, 200, 255))
                y += 30

            state = status.get("state", "?")
            hand_yes = ("collecting", "word_recognized", "repeat_suppressed",
                        "low_confidence_discarded", "sentence_complete")
            hand_no = ("idle", "skipped")
            hand_info = "YES" if state in hand_yes else "NO" if state in hand_no else "?"
            put_text(frame, "STATUS:", (14, y), 0.50, (180, 180, 180))
            status_parts = [f"[{state.upper()}]"]
            if state == "collecting":
                status_parts.append(f"frames: {status.get('frames', 0)}")
            put_text(frame, "  ".join(status_parts), (150, y), 0.50, (200, 200, 200))
            y += 22

            hand_color = (0, 255, 0) if hand_info == "YES" else (100, 100, 100)
            put_text(frame, "HAND:", (14, y), 0.50, (180, 180, 180))
            put_text(frame, hand_info, (150, y), 0.50, hand_color)

            put_text(frame, "Press 'q' to quit", (w - 180, h - 14), 0.45, (120, 120, 120))

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.flush()
        recognizer.close()


def main():
    parser = argparse.ArgumentParser(description="WLASL-100 Sign Language Recognition inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt checkpoint")
    parser.add_argument("--label-map", default=None, help="Path to label_map.json (optional if "
                         "embedded in the checkpoint)")
    parser.add_argument("--models-dir", default="models", help="Directory to cache MediaPipe .task files")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--video", default=None, help="Path to an input video file")
    parser.add_argument("--webcam", action="store_true", help="Use the default webcam as input")
    parser.add_argument("--continuous", action="store_true",
                         help="Continuous/sentence mode instead of single-clip prediction")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k classes to report (single-clip mode)")
    args = parser.parse_args()

    if args.continuous:
        _run_continuous(args)
    else:
        _run_single_clip(args)


if __name__ == "__main__":
    main()
