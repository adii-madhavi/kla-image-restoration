"""Reproducibility helpers.

Sets python/numpy/torch seeds together so that experiments (and the
submitted checkpoint) can be reproduced from a config + seed alone.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch RNGs.

    Args:
        seed: integer seed recorded in every experiment config/checkpoint.
        deterministic: if True, ask cuDNN for deterministic algorithms.
            This can slow training down slightly but makes results
            reproducible run-to-run on the same hardware/driver stack.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can be faster but breaks strict reproducibility
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn companion so each worker gets a distinct
    but deterministic seed derived from torch's initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
