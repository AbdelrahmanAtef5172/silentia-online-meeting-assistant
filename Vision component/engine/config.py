import os
import copy
import yaml
from dotenv import load_dotenv

def load_config(path: str = "configs/config.yaml", env: str = None) -> dict:
    """
    Load and merge config for the given environment.
    """
    load_dotenv()
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
