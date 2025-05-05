import logging
import sys
from typing import Optional

import torch
import os
import numpy as np
import random as rd


def setup_seed(seed: int = 21, cuda: bool = False):
    rd.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHON-SEED'] = str(seed)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benckmark = True
    torch.backends.enabled = True


def build_logger(
        name: Optional[str] = None,
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        mode: str = 'w',
        fmt: str = '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt: str = '%Y-%m-%d %H:%M:%S',
        log_type: str = 'train'
) -> logging.Logger:
    if name is None:
        name = __name__

    logger = logging.getLogger(name)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(ch)
        ch.setLevel(level)
    logger.setLevel(level)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{log_type}.log')
        fh = logging.FileHandler(log_path, mode=mode)
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(fh)
        fh.setLevel(level)

    return logger
