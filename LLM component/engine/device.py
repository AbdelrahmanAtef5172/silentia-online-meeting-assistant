"""
utils/device.py
────────────────
Environment information utility used for logging, telemetry, and
diagnostics. Provides a snapshot of the runtime environment without
importing heavy or hardware-specific libraries.
"""

import os
import sys
import platform
from dataclasses import dataclass, field


@dataclass
class EnvironmentInfo:
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=lambda: platform.platform())
    env_name: str = field(default_factory=lambda: os.environ.get("ENV", "development"))
    cuda_available: bool = False


def get_environment_info() -> EnvironmentInfo:
    """
    Gather runtime environment information for logging and telemetry.
    Returns an EnvironmentInfo dataclass.
    """
    return EnvironmentInfo(
        cuda_available=_check_cuda(),
    )


def _check_cuda() -> bool:
    """Check if CUDA is available without importing torch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
