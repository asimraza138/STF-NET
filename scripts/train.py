#!/usr/bin/env python3
"""
STF-Net Training Script

Supports two training modes:

1. Standard (single-process or DataParallel):
       python scripts/train.py --config config/default.yaml

2. Distributed (4× GPU, recommended for A100s):
       torchrun --nproc_per_node=4 scripts/train.py \
           --config config/default.yaml --distributed

The script runs both training stages:
  Stage 1 — ACS complexity estimator (skipped if checkpoint exists)
  Stage 2 — Full STF-Net (10 seeds, best val-F1 checkpoint per seed)
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="Train TSF-Net")
    parser.add_argument("--config",       type=str,
                        default="config/default.yaml")
    parser.add_argument("--distributed",  action="store_true",
                        help="Use DistributedDataParallel (launch with torchrun)")
    parser.add_argument("--skip_stage1",  action="store_true",
                        help="Skip ACS estimator training (use existing checkpoint)")
    parser.add_argument("--seeds_only",   type=str, default=None,
                        help="Comma-separated seed offsets to run, e.g. '0,1,2'")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Distributed setup ─────────────────────────────────────────────────
    if args.distributed:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank       = 0
        world_size = 1

    is_main = (rank == 0)

    # ── Stage 1: ACS complexity estimator ────────────────────────────────
    acs_ckpt = os.path.join(
        cfg["paths"]["checkpoints"], "complexity_estimator", "best.pt"
    )
    if not args.skip_stage1 and not os.path.exists(acs_ckpt) and is_main:
        print("\n" + "="*60)
        print("  Stage 1: Training ACS Complexity Estimator")
        print("="*60)
        from training.complexity_trainer import train_complexity_estimator
        train_complexity_estimator(cfg)

    if args.distributed:
        dist.barrier()

    # ── Stage 2: STF-Net multi-seed training ──────────────────────────────
    if is_main:
        print("\n" + "="*60)
        print("  Stage 2: Training STF-Net")
        print("="*60)

    from training.trainer import train_one_seed, set_seed
    from utils.metrics    import aggregate_seeds
    from utils.logging_utils import save_metrics

    base_seed = cfg["training"]["seed"]
    n_seeds   = cfg["training"]["n_seeds"]

    if args.seeds_only:
        seed_offsets = [int(x) for x in args.seeds_only.split(",")]
    else:
        seed_offsets = list(range(n_seeds))

    results = []
    for offset in seed_offsets:
        seed = base_seed + offset
        if is_main:
            print(f"\n{'─'*50}")
            print(f"  Seed {offset+1}/{len(seed_offsets)}  (seed={seed})")
            print(f"{'─'*50}")
        metrics = train_one_seed(cfg, seed=seed, rank=rank,
                                  world_size=world_size)
        if is_main:
            results.append(metrics)

    if is_main and len(results) > 1:
        aggregated = aggregate_seeds(results)
        save_metrics(
            aggregated,
            os.path.join(cfg["paths"]["logs"], "aggregated_results.json")
        )
        print("\n" + "="*60)
        print("  Final Aggregated Results (mean ± std over seeds)")
        print("="*60)
        for k, v in aggregated.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.4f}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
