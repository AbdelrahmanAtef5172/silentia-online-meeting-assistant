"""
engine/device.py
────────────────
Auto-selection of the best available compute device (CUDA, MPS, or CPU).
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def get_device(preferred: Optional[str] = "auto") -> torch.device:
    """
    Returns the best available torch.device.
    
    Args:
        preferred: "auto" | "cuda" | "mps" | "cpu"
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return torch.device("cpu")
