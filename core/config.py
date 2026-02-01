# core/config.py
from __future__ import annotations
import yaml
import copy
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenConfig:
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def save_yaml(self, path: Path) -> None:
        with path.open("w") as f:
            yaml.safe_dump(self.data, f, sort_keys=True)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_config(
    base_config_path: Path,
    overrides: Dict[str, Any] | None = None,
) -> FrozenConfig:
    base = load_yaml(base_config_path)
    overrides = overrides or {}

    resolved = _deep_merge(base, overrides)

    _validate_config(resolved)

    return FrozenConfig(resolved)


def _validate_config(cfg: Dict[str, Any]) -> None:
    required = ["training", "model", "seed"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
