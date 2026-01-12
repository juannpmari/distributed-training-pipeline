# core/logging.py
import logging
from pathlib import Path


def setup_logging(log_dir: Path, rank: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"rank_{rank}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logging.info("Logging initialized")
