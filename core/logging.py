# core/logging.py
import logging
import sys

def setup_logger(name: str, log_file: str | None, is_master: bool):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if is_master:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(log_file)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
