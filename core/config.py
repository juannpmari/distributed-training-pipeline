# core/config.py
import yaml
from types import MappingProxyType

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Shallow immutability (enough for now)
    return MappingProxyType(cfg)


