# core/paths.py
from pathlib import Path

def build_run_paths(base_dir: str, run_id: str):
    root = Path(base_dir) / run_id
    return {
        "root": root,
        "logs": root / "logs",
        "checkpoints": root / "checkpoints",
        "artifacts": root / "artifacts",
    }
