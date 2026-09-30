"""Shared imports, device selection and small helpers used by all modules."""

import os
import re
import time
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Seed torch, CUDA and numpy for reproducible runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def torch_to_np(A):
    if isinstance(A, np.ndarray):
        return A
    if isinstance(A, torch.Tensor):
        return A.detach().cpu().numpy()
    return np.array(A)


def np_to_torch(A):
    """Convert to a float32 tensor on the working device with requires_grad=True."""
    if isinstance(A, torch.Tensor):
        return A.to(dtype=torch.float32, device=device).requires_grad_()
    if isinstance(A, np.ndarray):
        return torch.tensor(A, dtype=torch.float32, device=device, requires_grad=True)
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(A)}")


def clear_memory(verbose=False):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print("Memory cleared")


set_seed(1)
