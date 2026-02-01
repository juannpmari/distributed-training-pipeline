# core/seeding.py
import os
import random
import numpy as np
import torch

def seed_everything(
    base_seed: int,
    rank: int = 0,
    deterministic: bool = True,
) -> int:
    seed = base_seed + rank

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
