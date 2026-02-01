# core/run_context.py
from dataclasses import dataclass
from pathlib import Path
import time
import hashlib
import json

@dataclass(frozen=True)
class RunContext:
    run_id: str
    root_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    created_at: float


def _hash_dict(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def create_run_context(
    base_dir: Path,
    experiment_name: str,
    resolved_config: dict,
) -> RunContext:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cfg_hash = _hash_dict(resolved_config)

    run_id = f"{experiment_name}_{timestamp}_{cfg_hash}"
    root = base_dir / run_id

    logs = root / "logs"
    ckpts = root / "checkpoints"
    artifacts = root / "artifacts"

    for d in (root, logs, ckpts, artifacts):
        d.mkdir(parents=True, exist_ok=False)

    return RunContext(
        run_id=run_id,
        root_dir=root,
        logs_dir=logs,
        checkpoints_dir=ckpts,
        artifacts_dir=artifacts,
        created_at=time.time(),
    )
