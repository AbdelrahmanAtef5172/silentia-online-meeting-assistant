"""
sign_language_model.py
=======================

Core module for the WLASL-100 Sign Language Recognition (SLR) component.

This file defines:
  1. `SignLanguageTransformer`   - the neural network architecture (skeleton/keypoint based).
  2. `KeypointExtractor`         - wraps MediaPipe Tasks (PoseLandmarker + HandLandmarker) to turn
                                    raw video frames into normalized keypoint feature sequences.
  3. `SignLanguageRecognizer`    - a "never-raises" inference interface used by the rest of the
                                    pipeline (LLM text correction, TTS, etc). Loads a trained
                                    checkpoint + label map and exposes `.predict(...)`.

Design notes
------------
- Input representation: MediaPipe **pose (33 landmarks)** + **left hand (21)** + **right hand (21)**
  keypoints, each with (x, y, z), plus 2 binary hand-presence flags -> 227-dim feature per frame.
  This is a *skeleton-based* approach rather than raw-pixel CNN/ViT features. For an isolated,
  100-class, ~2k-clip dataset like WLASL100, skeleton features are dramatically more
  data-efficient and orders of magnitude cheaper to run than a video CNN/ViT backbone, which is
  what lets this model run comfortably on CPU as well as GPU.
- MediaPipe's legacy `mp.solutions.holistic` API has been removed from recent MediaPipe Python
  releases (HolisticLandmarker was dropped from the Python package). This module therefore uses
  the current **MediaPipe Tasks API** (`mediapipe.tasks.python.vision`) with separate
  PoseLandmarker + HandLandmarker models, which is the actively supported path going forward.
- Sequences are resampled to a fixed length (`sequence_length`, default 64 frames) via linear
  interpolation, so clips of any original duration/frame-rate are handled uniformly.
- The inference interface never raises: every public method returns a result dict with a
  `success` flag and an `error` field on failure, matching the "never-raises service contract"
  convention used by the other components in this pipeline (LLM correction, TTS).
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Constants describing the keypoint feature layout. Keep these in sync with config.yaml.
# --------------------------------------------------------------------------------------
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
# pose(33*3) + left_hand(21*3) + right_hand(21*3) + 2 hand-presence flags
FEATURE_DIM = (
    NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK
    + 2 * NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK
    + 2
)  # = 227

MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MEDIAPIPE_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# ========================================================================================
# 1. MODEL ARCHITECTURE
# ========================================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding added to the per-frame embeddings."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionPool(nn.Module):
    """A single learned query attends over the time dimension to produce a pooled summary.

    This is combined with mean pooling in the classifier head; attention pooling lets the
    model learn to emphasize the most discriminative frames of a sign (e.g. the peak hand
    configuration) instead of treating all frames -- including transition/rest frames -- equally.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D); key_padding_mask: (B, T) True = ignore
        q = self.query.expand(x.size(0), -1, -1)  # (B, 1, D)
        scores = torch.bmm(q, x.transpose(1, 2)) * self.scale  # (B, 1, T)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, x)  # (B, 1, D)
        return pooled.squeeze(1)


class SignLanguageTransformer(nn.Module):
    """Transformer encoder over a sequence of per-frame skeleton keypoints.

    Input:  (B, T, input_dim) float tensor of normalized keypoints (see KeypointExtractor).
    Output: (B, num_classes) raw logits.

    The model is intentionally small (a few million parameters) so it runs with low latency
    on CPU as well as GPU -- appropriate given WLASL100's ~2,000 training clips, where a large
    model would overfit badly anyway.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 100,
        dropout: float = 0.2,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = AttentionPool(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        key_padding_mask: optional (B, T) bool tensor, True at positions to ignore
                           (e.g. frames where no landmarks were detected at all).
        """
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        attn_pooled = self.attn_pool(h, key_padding_mask)
        if key_padding_mask is not None:
            keep_mask = (~key_padding_mask).unsqueeze(-1).float()
            mean_pooled = (h * keep_mask).sum(1) / keep_mask.sum(1).clamp(min=1e-6)
        else:
            mean_pooled = h.mean(dim=1)

        combined = torch.cat([attn_pooled, mean_pooled], dim=-1)
        return self.classifier(combined)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(model_cfg: Dict[str, Any]) -> SignLanguageTransformer:
    """Construct a SignLanguageTransformer from a `config.yaml`-style dict (the `model:` section)."""
    return SignLanguageTransformer(
        input_dim=model_cfg.get("input_dim", FEATURE_DIM),
        d_model=model_cfg.get("d_model", 192),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        num_classes=model_cfg.get("num_classes", 100),
        dropout=model_cfg.get("dropout", 0.2),
        max_seq_len=model_cfg.get("max_seq_len", 128),
    )


# ========================================================================================
# 2. KEYPOINT EXTRACTION (MediaPipe Tasks: PoseLandmarker + HandLandmarker)
# ========================================================================================

def ensure_mediapipe_models(models_dir: str) -> Tuple[str, str]:
    """Download the MediaPipe pose/hand .task model bundles if not already present locally.

    Returns (pose_model_path, hand_model_path).
    """
    os.makedirs(models_dir, exist_ok=True)
    pose_path = os.path.join(models_dir, "pose_landmarker_lite.task")
    hand_path = os.path.join(models_dir, "hand_landmarker.task")

    if not os.path.exists(pose_path):
        urllib.request.urlretrieve(MEDIAPIPE_POSE_MODEL_URL, pose_path)
    if not os.path.exists(hand_path):
        urllib.request.urlretrieve(MEDIAPIPE_HAND_MODEL_URL, hand_path)
    return pose_path, hand_path


@dataclass
class FrameKeypoints:
    """Result of running keypoint extraction on a single frame."""
    features: np.ndarray  # shape (FEATURE_DIM,)
    hand_present: bool


class KeypointExtractor:
    """Extracts normalized pose + two-hand keypoints from video frames using MediaPipe Tasks.

    Uses `VIDEO` running mode (frame-by-frame with monotonically increasing timestamps), which
    is appropriate both for offline processing of pre-recorded clips (training/inference on a
    video file) and for sequential processing of a live camera feed frame-by-frame. For a fully
    asynchronous real-time pipeline, MediaPipe's `LIVE_STREAM` mode with a result callback is an
    alternative, but VIDEO mode keeps the calling code simple and synchronous, which is what most
    deployments of this component need.
    """

    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        # Imported lazily so that importing this module doesn't hard-require mediapipe
        # for callers that only need the pure-PyTorch model definition (e.g. exporting to ONNX).
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._VisionRunningMode = mp_vision.RunningMode

        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model_path),
            running_mode=self._VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model_path),
            running_mode=self._VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        self._closed = False
        # MediaPipe's VIDEO running mode requires timestamps to strictly increase for the
        # entire lifetime of the landmarker object -- not just within one clip. Since a single
        # KeypointExtractor is commonly reused across many videos (e.g. a training-set
        # extraction loop, or repeated SignLanguageRecognizer.predict() calls in production),
        # each of which naturally wants to count its own frames from 0, we track the last
        # timestamp actually given to MediaPipe and clamp every new one to be strictly greater
        # -- regardless of what the caller passes in. This makes the extractor safe to reuse
        # across any number of videos/streams without the caller having to manage a global
        # counter itself. See extract_from_frame().
        self._last_timestamp_ms = -1

    def reset_timestamp(self):
        """Optional: call this between logically-unrelated videos/streams if you want the
        next frame's timestamp to be treated as a fresh start for tracking purposes. Not
        required for correctness (extract_from_frame() guards monotonicity unconditionally
        either way) -- this only affects MediaPipe's internal tracking continuity, not
        whether it crashes."""
        self._last_timestamp_ms = -1

    # ---- internal helpers -------------------------------------------------------------

    @staticmethod
    def _landmarks_to_array(landmark_list, count: int) -> np.ndarray:
        arr = np.zeros((count, 3), dtype=np.float32)
        if landmark_list:
            for i, lm in enumerate(landmark_list[:count]):
                arr[i] = (lm.x, lm.y, lm.z)
        return arr

    @staticmethod
    def _normalize(pose: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
        """Center on the shoulder midpoint and scale by shoulder width, so the features are
        invariant to the signer's position/distance from the camera. Falls back to an identity
        transform if shoulders weren't detected (all-zero pose)."""
        left_shoulder, right_shoulder = pose[11], pose[12]
        if np.allclose(left_shoulder, 0) and np.allclose(right_shoulder, 0):
            return np.concatenate([pose, left_hand, right_hand], axis=0)

        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
        scale = scale if scale > 1e-6 else 1.0

        pose_n = (pose - center) / scale
        left_hand_n = (left_hand - center) / scale if left_hand.any() else left_hand
        right_hand_n = (right_hand - center) / scale if right_hand.any() else right_hand
        return np.concatenate([pose_n, left_hand_n, right_hand_n], axis=0)

    def _make_mp_image(self, frame_rgb: np.ndarray):
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

    # ---- public API ---------------------------------------------------------------------

    def extract_from_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameKeypoints:
        """frame_bgr: an OpenCV-style BGR frame (H, W, 3). timestamp_ms is a hint (e.g. "this
        is frame i of this clip, at i * ms_per_frame") -- it does NOT need to be globally
        monotonic across separate calls to extract_from_video()/predict() for different
        videos; this method guards that internally (see __init__)."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._make_mp_image(frame_rgb)

        # Clamp to strictly greater than the last timestamp actually sent to MediaPipe, so
        # reusing this extractor across multiple videos/clips (each restarting its own frame
        # count at 0) never violates MediaPipe's "must be monotonically increasing" requirement.
        safe_timestamp_ms = max(int(timestamp_ms), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = safe_timestamp_ms

        pose_result = self._pose_landmarker.detect_for_video(mp_image, safe_timestamp_ms)
        hand_result = self._hand_landmarker.detect_for_video(mp_image, safe_timestamp_ms)

        pose = np.zeros((NUM_POSE_LANDMARKS, 3), dtype=np.float32)
        if pose_result.pose_landmarks:
            pose = self._landmarks_to_array(pose_result.pose_landmarks[0], NUM_POSE_LANDMARKS)

        left_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        right_hand = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
        left_present, right_present = 0.0, 0.0

        if hand_result.hand_landmarks:
            for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"
                coords = self._landmarks_to_array(landmarks, NUM_HAND_LANDMARKS)
                if label == "Left":
                    left_hand = coords
                    left_present = 1.0
                else:
                    right_hand = coords
                    right_present = 1.0

        normalized = self._normalize(pose, left_hand, right_hand)
        features = np.concatenate([normalized.flatten(), [left_present, right_present]]).astype(
            np.float32
        )
        hand_present = bool(left_present or right_present)
        return FrameKeypoints(features=features, hand_present=hand_present)

    def extract_from_video(
        self, video_path: str, sample_fps: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Runs extraction over an entire video file.

        Returns:
            features: (num_frames, FEATURE_DIM) float32 array
            hand_present_mask: (num_frames,) bool array
        Raises FileNotFoundError / RuntimeError on unreadable video (caller -- e.g.
        SignLanguageRecognizer -- is responsible for catching and converting to a
        never-raises response).
        """
        import cv2

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0
        if sample_fps and sample_fps > 0 and sample_fps < native_fps:
            frame_interval = native_fps / sample_fps

        all_features: List[np.ndarray] = []
        all_present: List[bool] = []
        frame_idx = 0
        next_sample_at = 0.0
        timestamp_ms = 0
        ms_per_frame = 1000.0 / native_fps

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= next_sample_at:
                kp = self.extract_from_frame(frame, int(timestamp_ms))
                all_features.append(kp.features)
                all_present.append(kp.hand_present)
                next_sample_at += frame_interval
            frame_idx += 1
            timestamp_ms += ms_per_frame

        cap.release()

        if not all_features:
            raise RuntimeError(f"No frames could be read from video: {video_path}")

        return np.stack(all_features, axis=0), np.array(all_present, dtype=bool)

    def close(self):
        if not self._closed:
            self._pose_landmarker.close()
            self._hand_landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def resample_sequence(features: np.ndarray, target_len: int) -> np.ndarray:
    """Uniformly resamples a (T, D) sequence to (target_len, D) via linear interpolation.
    Used both to fit the model's fixed sequence length and (during training,
    with randomized target/segment lengths) as a temporal-speed augmentation."""
    src_len = features.shape[0]
    if src_len == target_len:
        return features.astype(np.float32)
    if src_len == 1:
        return np.repeat(features, target_len, axis=0).astype(np.float32)

    src_idx = np.linspace(0, src_len - 1, num=src_len)
    tgt_idx = np.linspace(0, src_len - 1, num=target_len)
    interp_indices = np.clip(np.searchsorted(src_idx, tgt_idx), 1, src_len - 1)
    left, right = features[interp_indices - 1], features[interp_indices]
    frac = ((tgt_idx - src_idx[interp_indices - 1]) / (src_idx[interp_indices] - src_idx[interp_indices - 1]))[:, np.newaxis]
    return (left + frac * (right - left)).astype(np.float32)


# ========================================================================================
# 3. INFERENCE INTERFACE (never-raises contract)
# ========================================================================================

def _default_device(preferred: str = "cpu") -> torch.device:
    return torch.device("cpu")


class SignLanguageRecognizer:
    """Production inference interface for a single trimmed sign clip.

    Follows a never-raises contract: `.predict(...)` and `.predict_from_frames(...)` always
    return a dict and never propagate an exception, so a bug or bad input in this component
    can't take down the calling pipeline (LLM correction / TTS / orchestrator). Check the
    `success` key; on failure `error` contains a short description.

    Expected output contract (consumed by the downstream LLM component):
        {
          "success": True,
          "predicted_class": "hello",       # gloss string
          "predicted_index": 42,             # int class index
          "confidence": 0.87,                # float in [0, 1]
          "top_k": [{"class": "hello", "index": 42, "confidence": 0.87}, ...],
          "error": None,
        }
    """

    def __init__(
        self,
        checkpoint_path: str,
        label_map_path: Optional[str] = None,
        models_dir: str = "models",
        device: str = "cpu",
        sequence_length: int = 64,
        confidence_threshold: float = 0.0,
        create_extractor: bool = True,
    ):
        self.ready = False
        self.init_error: Optional[str] = None
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length
        self._extractor: Optional[KeypointExtractor] = None

        try:
            self.device = _default_device(device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            model_cfg = checkpoint.get("model_config", {})
            self.sequence_length = checkpoint.get("sequence_length", sequence_length)
            self.model = build_model_from_config(model_cfg)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            label_map = checkpoint.get("label_map")
            if label_map is None:
                if label_map_path is None:
                    raise ValueError(
                        "No label_map found in checkpoint and no label_map_path provided."
                    )
                with open(label_map_path, "r") as f:
                    label_map = json.load(f)
            # label_map: {"0": "hello", "1": "thanks", ...} -> normalize to int keys
            self.index_to_label: Dict[int, str] = {int(k): v for k, v in label_map.items()}

            if create_extractor:
                pose_model_path, hand_model_path = ensure_mediapipe_models(models_dir)
                self._extractor = KeypointExtractor(pose_model_path, hand_model_path)

            self.ready = True
        except Exception as e:  # never let construction failures propagate as a hard crash
            self.init_error = f"{type(e).__name__}: {e}"

    # ---- internal ------------------------------------------------------------------------

    def _run_model(self, feature_seq: np.ndarray, top_k: int) -> Dict[str, Any]:
        resampled = resample_sequence(feature_seq, self.sequence_length)
        x = torch.from_numpy(resampled).unsqueeze(0).to(self.device)  # (1, T, D)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        k = min(top_k, probs.shape[0])
        top_indices = np.argsort(probs)[::-1][:k]
        top_k_results = [
            {
                "class": self.index_to_label.get(int(i), str(i)),
                "index": int(i),
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]
        best = top_k_results[0]
        return {
            "success": True,
            "predicted_class": best["class"],
            "predicted_index": best["index"],
            "confidence": best["confidence"],
            "top_k": top_k_results,
            "below_confidence_threshold": best["confidence"] < self.confidence_threshold,
            "error": None,
        }

    @staticmethod
    def _failure(error: str) -> Dict[str, Any]:
        return {
            "success": False,
            "predicted_class": None,
            "predicted_index": None,
            "confidence": 0.0,
            "top_k": [],
            "below_confidence_threshold": True,
            "error": error,
        }

    # ---- public API ----------------------------------------------------------------------

    def predict(self, video_path: str, top_k: int = 3) -> Dict[str, Any]:
        """Predict the sign class for a trimmed video clip on disk. Never raises."""
        if not self.ready:
            return self._failure(f"Recognizer not initialized: {self.init_error}")
        try:
            features, _ = self._extractor.extract_from_video(video_path)
            return self._run_model(features, top_k)
        except Exception as e:
            return self._failure(f"{type(e).__name__}: {e}")

    def predict_from_frames(self, frames: List[np.ndarray], top_k: int = 3) -> Dict[str, Any]:
        """Predict the sign class from an in-memory list/array of BGR frames
        (e.g. already-extracted frames from an upstream video reader). Never raises."""
        if not self.ready:
            return self._failure(f"Recognizer not initialized: {self.init_error}")
        try:
            if len(frames) == 0:
                return self._failure("Received an empty frame sequence.")
            feats = []
            ms_per_frame = 1000.0 / 30.0
            for i, frame in enumerate(frames):
                kp = self._extractor.extract_from_frame(frame, int(i * ms_per_frame))
                feats.append(kp.features)
            return self._run_model(np.stack(feats, axis=0), top_k)
        except Exception as e:
            return self._failure(f"{type(e).__name__}: {e}")

    def predict_from_features(self, feature_seq: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """Predict directly from a precomputed (T, FEATURE_DIM) keypoint array -- useful when
        keypoints were already extracted upstream (e.g. by ContinuousSignRecognizer)."""
        if not self.ready:
            return self._failure(f"Recognizer not initialized: {self.init_error}")
        try:
            if feature_seq.ndim != 2 or feature_seq.shape[1] != self.model.input_dim:
                return self._failure(
                    f"Expected features of shape (T, {self.model.input_dim}), got {feature_seq.shape}"
                )
            return self._run_model(feature_seq, top_k)
        except Exception as e:
            return self._failure(f"{type(e).__name__}: {e}")

    def close(self):
        if self._extractor is not None:
            self._extractor.close()
