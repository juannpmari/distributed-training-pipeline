# core/tracking/logger.py
import json
import time


class EventLogger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, event_type: str, payload: dict):
        record = {
            "time": time.time(),
            "event": event_type,
            **payload,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
