"""Logging utilities for training and evaluation."""

import os
import json
import logging
import datetime
from pathlib import Path


def setup_logger(name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    """
    Set up a logger that writes to both stdout and a file.
    Only rank-0 process logs to file in distributed training.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh  = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def save_metrics(metrics: dict, output_path: str):
    """Save metrics dict to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in metrics.items()},
            f, indent=2,
        )
