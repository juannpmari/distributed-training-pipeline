from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pathlib import Path

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]
    config_path: Path

def load_config(path: str) -> Config:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return Config(raw=raw, config_path=path)


