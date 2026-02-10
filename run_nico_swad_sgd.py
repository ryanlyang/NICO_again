#!/usr/bin/env python3
"""
Run SWAD using DomainBed's training loop, but with SGD and a fixed epoch budget.

This is a thin wrapper around `python -m domainbed.scripts.train` that:
  - computes the step count for a requested epoch budget
  - sets SGD + CosineAnnealingLR
  - passes hparams (lr/weight_decay/batch_size/momentum)
  - enables --swad

It mirrors the DomainBed SWAD path (LossValley) as closely as possible.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile


ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]


def count_lines(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for _ in f)


def ensure_nni_stub_if_missing(env: dict) -> dict:
    try:
        import nni  # noqa: F401
        return env
    except Exception:
        stub_dir = tempfile.mkdtemp(prefix="nni_stub_")
        stub_path = os.path.join(stub_dir, "nni.py")
        with open(stub_path, "w") as f:
            f.write(
                "def get_next_parameter():\n"
                "    return {}\n"
                "def report_intermediate_result(_):\n"
                "    pass\n"
                "def report_final_result(_):\n"
                "    pass\n"
            )
        env = dict(env)
        env["PYTHONPATH"] = stub_dir + os.pathsep + env.get("PYTHONPATH", "")
        return env


def main():
    p = argparse.ArgumentParser(description="SWAD (DomainBed) on NICO++ with SGD + fixed epochs")

    p.add_argument("--txtdir", type=str, default="/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist")
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--seed", type=int, default=59)

    # SGD settings (match your other runs)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # DomainBed train.py args
    p.add_argument("--output_dir", type=str, default="train_output")
    p.add_argument("--checkpoint_freq", type=int, default=1)
    p.add_argument("--stepval_freq", type=int, default=20)
    p.add_argument("--checkpoint_last", type=int, default=5)
    p.add_argument("--pretrain", action="store_true", default=True)
    p.add_argument("--no_pretrain", action="store_true", default=False)
    p.add_argument("--mix", action="store_true", default=False)

    args = p.parse_args()

    targets = [t.lower() for t in args.target]
    unknown = [t for t in targets if t not in ALL_DOMAINS]
    if unknown:
        raise SystemExit(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise SystemExit("Target domains cover all domains; no source domains remain to train on.")

    # Compute steps for a fixed epoch budget.
    train_sizes = []
    for d in sources:
        txt_path = os.path.join(args.txtdir, args.dataset, f"{d}_train.txt")
        if not os.path.exists(txt_path):
            raise SystemExit(f"Missing txtlist: {txt_path}")
        train_sizes.append(count_lines(txt_path))
    min_train = min(train_sizes)
    steps_per_epoch = max(1, math.ceil(min_train / args.batch_size))
    steps = args.epochs * steps_per_epoch

    hparams = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
    }

    cmd = [
        sys.executable, "-m", "domainbed.scripts.train",
        "--txtdir", args.txtdir,
        "--dataset", args.dataset,
        "--algorithm", "ERM",
        "--source", *sources,
        "--target", *targets,
        "--num_classes", str(args.num_classes),
        "--seed", str(args.seed),
        "--optimizer", "SGD",
        "--scheduler", "CosineAnnealingLR",
        "--swad",
        "--steps", str(steps),
        "--checkpoint_freq", str(args.checkpoint_freq),
        "--stepval_freq", str(args.stepval_freq),
        "--checkpoint_last", str(args.checkpoint_last),
        "--output_dir", args.output_dir,
        "--hparams", json.dumps(hparams),
    ]
    if args.mix:
        cmd.append("--mix")
    if args.pretrain and not args.no_pretrain:
        cmd.append("--pretrain")

    print("=== SWAD SGD Run ===")
    print(f"Targets: {targets}")
    print(f"Sources: {sources}")
    print(f"Steps per epoch: {steps_per_epoch}  |  Epochs: {args.epochs}  |  Steps: {steps}")
    print(f"SGD: lr={args.lr} momentum={args.momentum} weight_decay={args.weight_decay} batch_size={args.batch_size}")
    print("Command:")
    print("  " + " ".join(cmd))

    env = ensure_nni_stub_if_missing(os.environ)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
