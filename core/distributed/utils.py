# core/distributed/utils.py
import torch
from datetime import timedelta

def choose_backend():
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"

def default_timeout():
    # Realistic lab default
    return timedelta(minutes=30)
