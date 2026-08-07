"""
sign_language_model_bilstm.py
==============================

BiLSTM variant of the WLASL-100 Sign Language Recognition model.

Defines:
  1. `SignLanguageBiLSTM`  - the neural network architecture (bidirectional LSTM over keypoints).
  2. `build_model_from_config` - constructs the BiLSTM from a config dict.
  3. `SignLanguageBiLSTMRecognizer` - inference interface, same contract as the original.

Feature layout (identical to the Transformer variant):
  33 pose landmarks + 21 left-hand + 21 right-hand = 75 landmarks × (x, y, z) = 225
  + 2 binary hand-presence flags = 227-dimensional per-frame feature vector.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Feature layout constants (must match the cached .npz files)
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURE_DIM = (
    NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK
    + 2 * NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK
    + 2
)  # = 227


# ========================================================================================
# 1. BiLSTM ARCHITECTURE
# ========================================================================================

class SignLanguageBiLSTM(nn.Module):
    """Bidirectional LSTM over a sequence of per-frame skeleton keypoints.

    Input:  (B, T, input_dim) float tensor of normalized keypoints.
    Output: (B, num_classes) raw logits.

    The model pools over time via mean pooling (respecting a key_padding_mask for
    variable-length sequences) and passes the pooled representation through an MLP classifier.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_out_dim = hidden_dim * num_directions

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        key_padding_mask: optional (B, T) bool tensor, True at positions to ignore
                           (padded frames or frames with no detected landmarks).
        """
        h = self.input_proj(x)  # (B, T, hidden_dim)

        if key_padding_mask is not None:
            lengths = (~key_padding_mask).sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=h.size(1)
            )
        else:
            lstm_out, (h_n, _) = self.lstm(h)

        # Mean pooling over time, respecting padding
        if key_padding_mask is not None:
            keep_mask = (~key_padding_mask).unsqueeze(-1).float()
            pooled = (lstm_out * keep_mask).sum(1) / keep_mask.sum(1).clamp(min=1e-6)
        else:
            pooled = lstm_out.mean(dim=1)

        return self.classifier(pooled)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(model_cfg: Dict[str, Any]) -> SignLanguageBiLSTM:
    """Construct a SignLanguageBiLSTM from a config dict (the `model:` section)."""
    return SignLanguageBiLSTM(
        input_dim=model_cfg.get("input_dim", FEATURE_DIM),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 2),
        num_classes=model_cfg.get("num_classes", 100),
        dropout=model_cfg.get("dropout", 0.3),
        bidirectional=model_cfg.get("bidirectional", True),
    )


# ========================================================================================
# 2. RESAMPLE UTILITY (shared with the Transformer version)
# ========================================================================================

def resample_sequence(features: np.ndarray, target_len: int) -> np.ndarray:
    """Uniformly resamples a (T, D) sequence to (target_len, D) via linear interpolation.
    Used for fitting the model's fixed sequence length."""
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
# 3. NEVER-RAISES INFERENCE INTERFACE
# ========================================================================================

def _default_device(device_str: str) -> torch.device:
    return torch.device("cpu")


class SignLanguageBiLSTMRecognizer:
    """Production inference interface for a single trimmed sign clip.

    Loads a BiLSTM checkpoint saved by this notebook and runs prediction on
    pre-extracted keypoint features or from a video file (using MediaPipe).
    All public methods return a dict with a `success` flag and never raise.
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
        self._extractor = None

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
                    raise ValueError("No label_map in checkpoint and no label_map_path provided.")
                with open(label_map_path) as f:
                    label_map = json.load(f)
            self.index_to_label: Dict[int, str] = {int(k): v for k, v in label_map.items()}

            if create_extractor:
                try:
                    from sign_language_model import ensure_mediapipe_models, KeypointExtractor
                    pose_path, hand_path = ensure_mediapipe_models(models_dir)
                    self._extractor = KeypointExtractor(pose_path, hand_path)
                except Exception as mp_err:
                    print(f"Warning: MediaPipe extractor could not be created ({mp_err}). "
                          "predict_from_frames/video will be unavailable; predict_from_features will still work.")

            self.ready = True
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"

    def _run_model(self, feature_seq: np.ndarray, top_k: int) -> Dict[str, Any]:
        resampled = resample_sequence(feature_seq, self.sequence_length)
        x = torch.from_numpy(resampled).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        k = min(top_k, probs.shape[0])
        top_indices = np.argsort(probs)[::-1][:k]
        top_k_results = [
            {"class": self.index_to_label.get(int(i), str(i)), "index": int(i), "confidence": float(probs[i])}
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

    def _failure(self, error_msg: str) -> Dict[str, Any]:
        return {"success": False, "predicted_class": None, "predicted_index": None,
                "confidence": 0.0, "top_k": [], "below_confidence_threshold": False,
                "error": error_msg}

    def predict(self, video_path: str, top_k: int = 3) -> Dict[str, Any]:
        if not self.ready:
            return self._failure(f"Recognizer not initialized: {self.init_error}")
        if self._extractor is None:
            return self._failure("No MediaPipe extractor available.")
        try:
            features, _ = self._extractor.extract_from_video(video_path)
            return self._run_model(features, top_k)
        except Exception as e:
            return self._failure(f"{type(e).__name__}: {e}")

    def predict_from_features(self, feature_seq: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        if not self.ready:
            return self._failure(f"Recognizer not initialized: {self.init_error}")
        try:
            if feature_seq.ndim != 2 or feature_seq.shape[1] != self.model.input_dim:
                return self._failure(
                    f"Expected features of shape (T, {self.model.input_dim}), got {feature_seq.shape}"
                )
            if feature_seq.shape[0] == 0:
                return self._failure("Empty feature sequence.")
            return self._run_model(feature_seq, top_k)
        except Exception as e:
            return self._failure(f"{type(e).__name__}: {e}")

    def close(self):
        if self._extractor is not None:
            try:
                self._extractor.close()
            except Exception:
                pass

    @property
    def model_input_dim(self) -> int:
        return self.model.input_dim if self.ready else FEATURE_DIM


print('sign_language_model_bilstm.py written successfully')
