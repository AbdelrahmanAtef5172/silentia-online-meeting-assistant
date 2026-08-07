"""
orchestrate.py — Silentia Pipeline Orchestrator

Runs the full Silentia pipeline:

  Phase 1 (parallel):    Vision ──► gender label
                         SLR    ──► sign text

  Phase 2 (sequential):  LLM    ──► corrected text   (after SLR)

  Phase 3 (sequential):  TTS    ──► audio file       (after LLM + Vision)

Each component runs as a *subprocess* with its own .venv so no shared
virtual environment is required.  No existing component code is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import yaml

from schemas import (
    PipelineInput,
    PipelineResult,
    ProcessingMode,
    SLRMode,
    StageResult,
)


class SilentiaOrchestrator:
    """Orchestrates the 4-component Silentia pipeline via subprocess calls."""

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = os.path.abspath(config_path)
        self._config_dir = os.path.dirname(self._config_path)
        self._cfg = self._load_config()
        self._tmp_dir = os.path.join(self._config_dir, "tmp")
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self._config_path) as f:
            return yaml.safe_load(f)

    def _resolve(self, *parts: str) -> str:
        """Resolve path(s) relative to the config file's directory."""
        return os.path.normpath(os.path.join(self._config_dir, *parts))

    def _python(self, comp: str) -> str:
        """Return the path to the component's venv python binary."""
        venv = self._cfg["components"][comp]["venv"]
        return os.path.join(self._resolve(venv), "bin", "python")

    def _timeout(self, comp: str) -> int:
        return self._cfg["timeouts_seconds"].get(comp, 300)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, input_: PipelineInput) -> PipelineResult:
        start = time.time()

        # ---- Webcam mode: capture to temp file first ----
        video_path = input_.video_path
        if input_.mode == ProcessingMode.WEBCAM:
            print(f"[orch] Capturing {input_.webcam_duration_sec}s from webcam {input_.webcam_id}...")
            video_path = self._capture_webcam(input_)
            if not video_path:
                return self._fail_result("Webcam capture failed")

        print(f"[orch] Phase 1 — Vision + SLR (parallel)")
        vision_res, slr_res = self._phase1_parallel(video_path, input_)

        if slr_res.status == "failed":
            return PipelineResult(
                success=False, vision=vision_res, slr=slr_res,
                llm=StageResult("skipped"), tts=StageResult("skipped"),
                error=f"SLR failed: {slr_res.error}",
            )

        print(f"[orch] Phase 2 — LLM (sequential)")
        llm_res = self._phase2_llm(slr_res, input_)

        print(f"[orch] Phase 3 — TTS (sequential)")
        tts_res = self._phase3_tts(vision_res, llm_res, input_)

        success = (
            vision_res.status == "completed"
            and slr_res.status == "completed"
            and llm_res.status in ("completed", "skipped")
            and tts_res.status in ("completed", "skipped")
        )

        return PipelineResult(
            success=success, vision=vision_res, slr=slr_res,
            llm=llm_res, tts=tts_res,
        )

    # ------------------------------------------------------------------
    # Phase 1 — parallel
    # ------------------------------------------------------------------

    def _phase1_parallel(self, video_path: str, input_: PipelineInput) -> Tuple[StageResult, StageResult]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            vf = pool.submit(self._run_vision, video_path, input_)
            sf = pool.submit(self._run_slr, video_path, input_)
            return vf.result(timeout=self._timeout("vision")), sf.result(timeout=self._timeout("slr"))

    def _run_vision(self, video_path: str, input_: PipelineInput) -> StageResult:
        t0 = time.time()
        c = self._cfg["components"]["vision"]
        out_path = os.path.join(self._tmp_dir, f"vision_{input_.session_id}.json")

        cmd = [
            self._python("vision"),
            c["script"],
            "--input", video_path,
            "--output", out_path,
            "--env", c.get("env", "development"),
        ]
        if c.get("config_path"):
            cmd += ["--config", self._resolve(c["config_path"])]

        try:
            proc = subprocess.run(
                cmd, cwd=self._resolve(c["root"]),
                capture_output=True, text=True,
                timeout=self._timeout("vision"),
            )
            elapsed = (time.time() - t0) * 1000

            if proc.returncode != 0:
                return StageResult("failed", error=proc.stderr or f"exit {proc.returncode}")

            if not os.path.exists(out_path):
                return StageResult("failed", error="Vision output file not created")

            with open(out_path) as f:
                data = json.load(f)

            frames = data.get("frames", [])
            if frames:
                last = frames[-1]
                result = {
                    "label": last.get("label", "no_face"),
                    "confidence": last.get("confidence", 0.0),
                    "total_frames": len(frames),
                    "fps": data.get("metadata", {}).get("fps", 0),
                }
            else:
                result = {"label": "no_face", "confidence": 0.0, "total_frames": 0}

            return StageResult("completed", result=result, latency_ms=elapsed)

        except subprocess.TimeoutExpired:
            return StageResult("failed", error="Vision timed out")
        except Exception as e:
            return StageResult("failed", error=f"{type(e).__name__}: {e}")

    def _run_slr(self, video_path: str, input_: PipelineInput) -> StageResult:
        t0 = time.time()
        c = self._cfg["components"]["slr"]
        out_path = os.path.join(self._tmp_dir, f"slr_{input_.session_id}.json")
        adapter = os.path.join(self._config_dir, "adapters", "run_slr_adapter.py")

        cmd = [
            self._python("slr"),
            adapter,
            "--video", video_path,
            "--checkpoint", self._resolve(c["checkpoint"]),
            "--output", out_path,
            "--device", c.get("device", "cpu"),
            "--component-root", self._resolve(c["root"]),
        ]
        if c.get("label_map"):
            cmd += ["--label-map", self._resolve(c["label_map"])]
        if c.get("models_dir"):
            cmd += ["--models-dir", self._resolve(c["models_dir"])]

        # SLR mode from config (overridable by PipelineInput)
        slr_mode = input_.slr_mode.value if hasattr(input_.slr_mode, "value") else input_.slr_mode
        cmd += ["--mode", slr_mode]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self._timeout("slr"),
            )
            elapsed = (time.time() - t0) * 1000

            if proc.returncode != 0:
                return StageResult("failed", error=proc.stderr or f"exit {proc.returncode}")

            if not os.path.exists(out_path):
                return StageResult("failed", error="SLR output file not created")

            with open(out_path) as f:
                data = json.load(f)

            if not data.get("success"):
                return StageResult("failed", error=data.get("error", "Unknown SLR error"))

            result = {
                "text": data.get("predicted_class") or data.get("sentence", ""),
                "confidence": data.get("confidence", 0.0),
                "top_k": data.get("top_k", []),
                "words": data.get("words", []),
            }
            return StageResult("completed", result=result, latency_ms=elapsed)

        except subprocess.TimeoutExpired:
            return StageResult("failed", error="SLR timed out")
        except Exception as e:
            return StageResult("failed", error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Phase 2 — LLM (sequential after SLR)
    # ------------------------------------------------------------------

    def _phase2_llm(self, slr_res: StageResult, input_: PipelineInput) -> StageResult:
        if slr_res.status != "completed" or not slr_res.result:
            return StageResult("skipped", error="No SLR result")

        slr_text = (slr_res.result.get("text") or "").strip()
        if not slr_text:
            return StageResult("skipped", error="Empty SLR text")

        t0 = time.time()
        c = self._cfg["components"]["llm"]
        out_path = os.path.join(self._tmp_dir, f"llm_{input_.session_id}.txt")

        cmd = [
            self._python("llm"),
            c["script"],
            "--text", slr_text,
            "--output", out_path,
        ]

        try:
            proc = subprocess.run(
                cmd, cwd=self._resolve(c["root"]),
                capture_output=True, text=True,
                timeout=self._timeout("llm"),
            )
            elapsed = (time.time() - t0) * 1000

            if proc.returncode != 0:
                return StageResult("failed", error=proc.stderr or f"exit {proc.returncode}")

            if os.path.exists(out_path):
                with open(out_path) as f:
                    corrected = f.read().strip()
            else:
                corrected = proc.stdout.strip()

            return StageResult("completed", result={
                "corrected_text": corrected,
                "original_text": slr_text,
            }, latency_ms=elapsed)

        except subprocess.TimeoutExpired:
            return StageResult("failed", error="LLM timed out")
        except Exception as e:
            return StageResult("failed", error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Phase 3 — TTS (sequential after LLM + Vision)
    # ------------------------------------------------------------------

    def _phase3_tts(self, vision_res: StageResult, llm_res: StageResult, input_: PipelineInput) -> StageResult:
        # Text to speak: prefer LLM output, fall back to empty
        if llm_res.status == "completed" and llm_res.result:
            text = (llm_res.result.get("corrected_text") or "").strip()
        else:
            text = ""

        if not text:
            return StageResult("skipped", error="No text to synthesize")

        # Gender from Vision
        gender = "unknown"
        if vision_res.status == "completed" and vision_res.result:
            label = vision_res.result.get("label", "no_face")
            if label in ("male", "female"):
                gender = label

        t0 = time.time()
        c = self._cfg["components"]["tts"]
        output_wav = os.path.join(
            self._resolve(c["root"]), "output",
            f"silentia_{input_.session_id}.wav",
        )
        os.makedirs(os.path.dirname(output_wav), exist_ok=True)

        cmd = [
            self._python("tts"),
            c["script"],
            "--text", text,
            "--gender", gender,
            "--mode", input_.tts_output_mode,
            "--output", output_wav,
            "--speed", str(input_.tts_speed),
        ]

        try:
            proc = subprocess.run(
                cmd, cwd=self._resolve(c["root"]),
                capture_output=True, text=True,
                timeout=self._timeout("tts"),
            )
            elapsed = (time.time() - t0) * 1000

            if proc.returncode != 0:
                return StageResult("failed", error=proc.stderr or f"exit {proc.returncode}")

            try:
                tts_data = json.loads(proc.stdout.strip())
            except (json.JSONDecodeError, ValueError):
                try:
                    import ast
                    parsed = ast.literal_eval(proc.stdout.strip())
                    if isinstance(parsed, dict):
                        tts_data = parsed
                    else:
                        tts_data = {"raw_output": proc.stdout.strip()}
                except Exception:
                    tts_data = {"raw_output": proc.stdout.strip()}

            return StageResult("completed", result={
                "audio_path": output_wav if os.path.exists(output_wav) else None,
                "text_spoken": text,
                "gender_used": gender,
                **tts_data,
            }, latency_ms=elapsed)

        except subprocess.TimeoutExpired:
            return StageResult("failed", error="TTS timed out")
        except Exception as e:
            return StageResult("failed", error=f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Webcam capture
    # ------------------------------------------------------------------

    def _capture_webcam(self, input_: PipelineInput) -> Optional[str]:
        out_path = os.path.join(self._tmp_dir, f"webcam_{input_.session_id}.mp4")
        cap = cv2.VideoCapture(input_.webcam_id)
        if not cap.isOpened():
            print(f"[orch] ERROR: cannot open webcam {input_.webcam_id}", file=sys.stderr)
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        total = int(fps * input_.webcam_duration_sec)
        print(f"[orch] Recording {total} frames ({input_.webcam_duration_sec}s @ {fps:.1f} fps)...")
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"[orch] Webcam capture -> {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Cleanup / helpers
    # ------------------------------------------------------------------

    def cleanup(self, session_id: str):
        for f in Path(self._tmp_dir).glob(f"*_{session_id}.*"):
            f.unlink()

    @staticmethod
    def _fail_result(msg: str) -> PipelineResult:
        err = StageResult("failed", error=msg)
        skip = StageResult("skipped")
        return PipelineResult(False, err, skip, skip, skip, error=msg)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Silentia Pipeline Orchestrator"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Input video file path")
    parser.add_argument("--webcam", type=int, default=None, const=0, nargs="?",
                        help="Webcam device ID (capture then process)")
    parser.add_argument("--slr-mode", type=str, default=None,
                        choices=["single", "continuous"],
                        help="SLR mode (overrides config)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Webcam capture duration (seconds)")
    parser.add_argument("--tts-mode", type=str, default="file",
                        choices=["file", "play"],
                        help="TTS output mode")
    parser.add_argument("--tts-speed", type=float, default=1.0,
                        help="TTS speech rate 0.5-2.0")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to orchestrator config.yaml")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete temp files after run")
    args = parser.parse_args()

    if args.video is None and args.webcam is None:
        parser.error("Provide --video PATH or --webcam [ID]")
    if args.video and args.webcam is not None:
        parser.error("Use --video or --webcam, not both")

    mode = ProcessingMode.WEBCAM if args.webcam is not None else ProcessingMode.FILE
    slr_mode = SLRMode(args.slr_mode) if args.slr_mode else SLRMode.SINGLE

    pipe_input = PipelineInput(
        video_path=args.video,
        webcam_id=args.webcam if args.webcam is not None else 0,
        mode=mode,
        slr_mode=slr_mode,
        tts_output_mode=args.tts_mode,
        tts_speed=args.tts_speed,
        webcam_duration_sec=args.duration or 10.0,
    )

    orch = SilentiaOrchestrator(config_path=args.config)
    result = orch.run(pipe_input)

    print("\n" + "=" * 60)
    print("PIPELINE RESULT")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2))

    if args.cleanup:
        orch.cleanup(pipe_input.session_id)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
