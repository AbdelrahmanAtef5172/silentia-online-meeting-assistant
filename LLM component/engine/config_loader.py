"""
utils/config_loader.py
───────────────────────
Loads and deep-merges configs/config.yaml for the active environment.
Same pattern as the Vision component's config loader.
"""

import os
import copy

import yaml


def load_config(path: str = "configs/config.yaml", env: str = None) -> dict:
    """
    Load config and merge the active environment section on top of defaults.

    Priority: function arg > ENV env-var > 'development'
    """
    env = env or os.environ.get("ENV", "development")

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = copy.deepcopy(raw.get("defaults", {}))

    if env in raw:
        config = _deep_merge(config, raw[env])

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
