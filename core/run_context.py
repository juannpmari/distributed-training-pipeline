from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time


@dataclass(frozen=True)
class RunContext:
    run_id: str
    root_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path
    created_at: float


def _hash_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def create_run_context(base_dir: str, experiment_name: str, config: dict) -> RunContext:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config_hash = _hash_config(config)
    run_id = f"{experiment_name}_{timestamp}_{config_hash}"

    root = Path(base_dir).resolve() / run_id
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
