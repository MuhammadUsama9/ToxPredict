"""
src/utils/seed_utils.py
-----------------------
Centralised seed-setting so every experiment starts from the same
random state. Call `set_all_seeds(42)` before any data loading or
model instantiation.
"""

import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Remaining non-determinism:
    - CuDNN benchmark mode scans fast conv algorithms at runtime;
      we disable it here, but CUDA atomic operations in scatter-add
      (used by PyG message passing) are still non-deterministic on GPU.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # reproducibility > speed
    print(f"[seed_utils] All seeds set to {seed}.")
