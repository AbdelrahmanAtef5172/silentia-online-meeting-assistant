import os
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml",
)


def load_config(
    path: Optional[str] = None,
    env: Optional[str] = None,
) -> dict:
    if path is None:
        path = DEFAULT_CONFIG_PATH

    env = env or os.getenv("ENV", "development")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    env_specific = raw.get(env, {})

    config = _deep_merge(defaults, env_specific)
    config["_env"] = env
    config["_config_path"] = path

    logger.debug("Config loaded from %s | env=%s", path, env)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
