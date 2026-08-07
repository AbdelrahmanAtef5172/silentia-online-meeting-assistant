"""
engine/gender_classifier.py
───────────────────────────
Wraps the ViT-B/16 + gender head for inference.

Input:  AlignedFace (224x224x3, RGB, float32 in [0,1])
Output: RawPrediction(label, confidence, logits, probabilities)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from engine.schemas import AlignedFace, RawPrediction, GenderLabel
from engine.transforms import get_inference_transforms
from engine.gender_head import GenderClassifier
from engine.device import get_device
import logging

logger = logging.getLogger(__name__)

LABEL_MAP = {0: GenderLabel.MALE, 1: GenderLabel.FEMALE}
CONFIDENCE_FLOOR = 0.5   # predictions below this floor are flagged as LOW_CONFIDENCE

class GenderInference:
    """
    Inference-mode wrapper for the ViT gender classifier.
    Usage:
        infer = GenderInference(weights_path="weights/vit_b16_gender_phase2.pth")
        prediction: RawPrediction = infer.predict(aligned_face)
    """

    def __init__(
        self,
        weights_path: str,   # path to the model weights file ("weights/vit_b16_phase2.pth")
        device: Optional[str] = None,  
        use_fp16: bool = False,
    ):
        self.device = get_device(device)
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.transforms = get_inference_transforms()
        self.model = self._load_model(weights_path)
        logger.info(f"GenderInference initialized | device={self.device} | fp16={self.use_fp16}")

    def _load_model(self, weights_path: str) -> GenderClassifier:
        model = GenderClassifier()
        state = torch.load(weights_path, map_location=self.device)
        # Check if state is a dict with 'model_state_dict' or just the state dict
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        if self.use_fp16:
            model.half()
            logger.info("GenderClassifier running in FP16 mode")

        return model

    @torch.no_grad()
    def predict(self, aligned_face: AlignedFace) -> RawPrediction:
        """
        Args:
            aligned_face: AlignedFace with .image of shape (224, 224, 3), float32, RGB, [0,1]

        Returns:
            RawPrediction with label, confidence, logits, and probabilities
        """
        tensor = self.transforms(aligned_face.image)      # → (3, 224, 224) normalized tensor
        tensor = tensor.unsqueeze(0).to(self.device)       # → (1, 3, 224, 224)

        if self.use_fp16:
            tensor = tensor.half()

        logits = self.model(tensor)                        # → (1, 2)
        probs = F.softmax(logits.float(), dim=-1)          # always float32 for probabilities
        pred_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_idx].item()

        return RawPrediction(
            label=LABEL_MAP[pred_idx],
            confidence=confidence,
            logits=logits[0].cpu().float().numpy(),
            probabilities=probs[0].cpu().numpy(),
            is_low_confidence=confidence < CONFIDENCE_FLOOR,
            aligned_face=aligned_face
        )
