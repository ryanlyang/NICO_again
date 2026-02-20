#!/usr/bin/env python3
"""
Time-boxed Optuna guided sweep on NICO++ with per-epoch test evaluation.

Behavior:
- Uses held-out targets (default: autumn + rock), trains on remaining domains.
- Samples hyperparameters with Optuna (TPE) each trial.
- Trains guided model (30 epochs default) with SGD and attention switch.
- Every epoch, computes:
    - val_acc
    - IG forward KL
    - log_optim_num = log(val_acc) - beta * ig_fwd_kl
    - optim_num = exp(log_optim_num)
    - test overall accuracy (plus per-target test accuracies)
- Stops launching/continuing training once time budget is reached.
- Writes CSV logs and generates scatter plot: optim_num vs test_overall.
"""

import argparse
import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    import optuna
    from optuna.trial import TrialState
except Exception as exc:
    raise SystemExit("optuna is required for this script. Install it in your env.") from exc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from run_guided_optuna_sgd import (
    ALL_DOMAINS,
    BATCH_SIZE,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MASK_ROOT,
    DEFAULT_TXTLIST_DIR,
    DEVICE,
    FIXED_NUM_EPOCHS,
    MOMENTUM,
    NICOWithMasks,
    WEIGHT_DECAY,
    _get_param_groups,
    build_train_val_datasets,
    compute_attn_losses,
    compute_loss,
    evaluate_test,
    integrated_gradients_saliency,
    make_cam_model,
    seed_everything,
    seed_worker,
)


def suggest_hparams(trial: "optuna.trial.Trial", args):
    return {
        "base_lr": trial.suggest_float("base_lr", args.base_lr_low, args.base_lr_high, log=True),
        "classifier_lr": trial.suggest_float(
            "classifier_lr", args.classifier_lr_low, args.classifier_lr_high, log=True
        ),
        "lr2_mult": trial.suggest_float("lr2_mult", args.lr2_mult_low, args.lr2_mult_high, log=True),
        "attention_epoch": trial.suggest_int("attention_epoch", args.att_epoch_min, args.att_epoch_max),
        "kl_lambda_start": trial.suggest_float(
            "kl_lambda_start", args.kl_start_low, args.kl_start_high, log=True
        ),
        "kl_increment": float(args.kl_increment),
    }


def train_one_trial_timeboxed(
    trial_id,
    model,
    dataloaders,
    test_loaders,
    hparams,
    args,
    sweep_start_time,
    epoch_writer,
):
    best_log_optim = -float("inf")
    best_val_acc = -float("inf")
    best_log_epoch = None
    best_val_epoch = None
    best_log_test_overall = None
    best_val_test_overall = None

    param_groups = _get_param_groups(model, hparams["base_lr"], hparams["classifier_lr"])
    opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    kl_lambda_real = hparams["kl_lambda_start"]
    budget_reached = False
    epochs_completed = 0

    for epoch in range(args.num_epochs):
        elapsed = time.time() - sweep_start_time
        if elapsed >= args.time_budget_seconds:
            budget_reached = True
            print(
                f"[TRIAL {trial_id}] budget reached before epoch {epoch} "
                f"(elapsed={elapsed / 3600.0:.2f}h).",
                flush=True,
            )
            break

        if epoch == hparams["attention_epoch"]:
            base_lr_post = hparams["base_lr"] * hparams["lr2_mult"]
            classifier_lr_post = hparams["classifier_lr"] * hparams["lr2_mult"]
            param_groups = _get_param_groups(model, base_lr_post, classifier_lr_post)
            opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

        if epoch > hparams["attention_epoch"]:
            kl_lambda_real += hparams["kl_increment"]

        # -------------------- train --------------------
        model.train()
        for batch in dataloaders["train"]:
            inputs, labels, gt_masks, _ = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)

            opt.zero_grad()
            outputs, feats = model(inputs)
            gt_small = nn.functional.interpolate(
                gt_masks, size=feats.shape[-2:], mode="nearest"
            ).squeeze(1)

            if epoch < hparams["attention_epoch"]:
                loss, _ = compute_loss(outputs, labels, feats, gt_small, kl_lambda=333, only_ce=True)
            else:
                loss, _ = compute_loss(
                    outputs, labels, feats, gt_small, kl_lambda=kl_lambda_real, only_ce=False
                )

            loss.backward()
            opt.step()

        sch.step()

        # -------------------- val: acc + IG forward KL --------------------
        model.eval()
        val_correct = 0
        val_total = 0
        sum_ig_fwd = 0.0

        for batch in dataloaders["val"]:
            inputs, labels, gt_masks, _ = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            bsz = inputs.size(0)

            with torch.no_grad():
                outputs, _ = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += bsz

            with torch.enable_grad():
                ig_sal = integrated_gradients_saliency(
                    model, inputs, labels, steps=args.ig_steps
                )

            ig_fwd_kl, _ = compute_attn_losses(ig_sal, gt_masks.squeeze(1))
            sum_ig_fwd += ig_fwd_kl.item() * bsz

        val_acc = val_correct / max(val_total, 1)
        ig_fwd_kl = sum_ig_fwd / max(val_total, 1)
        log_optim_num = math.log(max(val_acc, args.optim_eps)) - (args.beta * ig_fwd_kl)
        optim_num = math.exp(log_optim_num)

        # -------------------- test (every epoch) --------------------
        test_results = evaluate_test(model, test_loaders, args.target)
        test_overall = float(test_results.get("overall", float("nan")))
        elapsed = time.time() - sweep_start_time

        row = {
            "trial": trial_id,
            "epoch": epoch,
            "elapsed_hours": elapsed / 3600.0,
            "base_lr": hparams["base_lr"],
            "classifier_lr": hparams["classifier_lr"],
            "lr2_mult": hparams["lr2_mult"],
            "attention_epoch": hparams["attention_epoch"],
            "kl_lambda_start": hparams["kl_lambda_start"],
            "kl_increment": hparams["kl_increment"],
            "val_acc": val_acc,
            "ig_fwd_kl": ig_fwd_kl,
            "log_optim_num": log_optim_num,
            "optim_num": optim_num,
            "test_overall": test_overall,
        }
        for domain in args.target:
            row[f"test_{domain}"] = float(test_results.get(domain, float("nan")))
        epoch_writer.writerow(row)

        print(
            f"[TRIAL {trial_id}][Epoch {epoch}] val_acc={val_acc:.6f} "
            f"ig_fwd_kl={ig_fwd_kl:.6f} log_optim_num={log_optim_num:.6f} "
            f"optim_num={optim_num:.6e} test={test_results}",
            flush=True,
        )
        epochs_completed += 1

        if log_optim_num > best_log_optim:
            best_log_optim = log_optim_num
            best_log_epoch = epoch
            best_log_test_overall = test_overall
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_overall = test_overall

        if elapsed >= args.time_budget_seconds:
            budget_reached = True
            print(
                f"[TRIAL {trial_id}] budget reached after epoch {epoch} "
                f"(elapsed={elapsed / 3600.0:.2f}h).",
                flush=True,
            )
            break

    trial_summary = {
        "trial": trial_id,
        "base_lr": hparams["base_lr"],
        "classifier_lr": hparams["classifier_lr"],
        "lr2_mult": hparams["lr2_mult"],
        "attention_epoch": hparams["attention_epoch"],
        "kl_lambda_start": hparams["kl_lambda_start"],
        "kl_increment": hparams["kl_increment"],
        "epochs_completed": epochs_completed,
        "best_log_optim_num": best_log_optim,
        "best_log_epoch": best_log_epoch,
        "best_log_test_overall": best_log_test_overall,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "best_val_test_overall": best_val_test_overall,
    }
    return trial_summary, budget_reached


def generate_scatter(epoch_csv_path: str, scatter_png_path: str):
    xs = []
    ys = []

    with open(epoch_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["optim_num"])
                y = float(row["test_overall"])
            except (ValueError, KeyError):
                continue
            if math.isfinite(x) and math.isfinite(y) and x > 0.0:
                xs.append(x)
                ys.append(y)

    if not xs:
        print(f"No valid points found in {epoch_csv_path}; skipping scatter generation.", flush=True)
        return

    x_arr = np.array(xs, dtype=np.float64)
    y_arr = np.array(ys, dtype=np.float64)

    if len(x_arr) > 1 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    else:
        r = float("nan")

    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(x_arr, y_arr, s=14, alpha=0.65)
    plt.xscale("log")
    plt.xlabel("optim_num = exp(log_optim_num)")
    plt.ylabel("test_overall (%)")
    plt.title(f"Guided Time-box Sweep (Autumn+Rock) | n={len(x_arr)} | Pearson r={r:.4f}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(scatter_png_path, dpi=180)
    plt.close()

    print(f"Saved scatter: {scatter_png_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Time-boxed Optuna guided sweep with optim_num vs test scatter."
    )
    p.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--mask_root", type=str, default=DEFAULT_MASK_ROOT)
    p.add_argument("--target", nargs="+", default=["autumn", "rock"])
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_epochs", type=int, default=FIXED_NUM_EPOCHS)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--optim_eps", type=float, default=1e-12)
    p.add_argument("--time_budget_hours", type=float, default=22.0)
    p.add_argument("--max_trials", type=int, default=10_000)
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ryreu/guided_cnn/NICO_runs/output/guided_timebox_autumn_rock",
    )
    p.add_argument("--study_name", type=str, default="nico_guided_timebox_scatter_autumn_rock")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--load_if_exists", action="store_true")
    p.add_argument("--optuna_seed", type=int, default=59)

    # Optuna search ranges
    p.add_argument("--base_lr_low", type=float, default=1e-5)
    p.add_argument("--base_lr_high", type=float, default=5e-2)
    p.add_argument("--classifier_lr_low", type=float, default=1e-5)
    p.add_argument("--classifier_lr_high", type=float, default=5e-2)
    p.add_argument("--lr2_mult_low", type=float, default=1e-3)
    p.add_argument("--lr2_mult_high", type=float, default=1.0)
    p.add_argument("--att_epoch_min", type=int, default=1)
    p.add_argument("--att_epoch_max", type=int, default=29)
    p.add_argument("--kl_start_low", type=float, default=0.1)
    p.add_argument("--kl_start_high", type=float, default=50.0)
    p.add_argument("--kl_increment", type=float, default=0.0)
    args = p.parse_args()

    args.time_budget_seconds = max(1.0, args.time_budget_hours * 3600.0)

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")
    args.target = targets

    os.makedirs(args.output_dir, exist_ok=True)
    epoch_csv_path = os.path.join(args.output_dir, "timebox_epoch_metrics.csv")
    trial_csv_path = os.path.join(args.output_dir, "timebox_trial_summary.csv")
    scatter_png_path = os.path.join(args.output_dir, "optim_num_vs_test_overall.png")
    best_json_path = os.path.join(args.output_dir, "timebox_optuna_best.json")
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(args.output_dir, 'timebox_optuna.db')}"

    seed_everything(args.seed)
    base_g = torch.Generator()
    base_g.manual_seed(args.seed)

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "eval": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset, val_dataset, has_val, _ = build_train_val_datasets(
        args, sources, data_transforms, base_g
    )
    if not has_val:
        print("Val split: 16% split from train (no *_val.txt found)", flush=True)

    test_datasets = [
        NICOWithMasks(
            args.txtdir,
            args.dataset,
            [domain],
            "test",
            args.mask_root,
            args.image_root,
            data_transforms["eval"],
            None,
        )
        for domain in args.target
    ]

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )

    print(
        f"Running time-boxed Optuna guided sweep | targets={args.target}, sources={sources}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}, budget={args.time_budget_hours:.2f}h, "
        f"beta={args.beta}, ig_steps={args.ig_steps}, device={DEVICE}, "
        f"study={args.study_name}, storage={args.storage}",
        flush=True,
    )

    epoch_fields = [
        "trial", "epoch", "elapsed_hours",
        "base_lr", "classifier_lr", "lr2_mult", "attention_epoch", "kl_lambda_start", "kl_increment",
        "val_acc", "ig_fwd_kl", "log_optim_num", "optim_num", "test_overall",
    ] + [f"test_{d}" for d in args.target]
    trial_fields = [
        "trial",
        "trial_state",
        "base_lr", "classifier_lr", "lr2_mult", "attention_epoch", "kl_lambda_start", "kl_increment",
        "epochs_completed",
        "best_log_optim_num", "best_log_epoch", "best_log_test_overall",
        "best_val_acc", "best_val_epoch", "best_val_test_overall",
    ]

    with open(epoch_csv_path, "w", newline="") as epoch_f, open(trial_csv_path, "w", newline="") as trial_f:
        epoch_writer = csv.DictWriter(epoch_f, fieldnames=epoch_fields)
        trial_writer = csv.DictWriter(trial_f, fieldnames=trial_fields)
        epoch_writer.writeheader()
        trial_writer.writeheader()
        epoch_f.flush()
        trial_f.flush()

        sweep_start_time = time.time()
        trials_started = 0
        budget_reached = False

        while trials_started < args.max_trials:
            elapsed = time.time() - sweep_start_time
            if elapsed >= args.time_budget_seconds:
                budget_reached = True
                print("Budget reached before starting next trial.", flush=True)
                break

            trial = study.ask()
            hparams = suggest_hparams(trial, args)
            trial_id = trial.number
            print(
                f"\n[TRIAL {trial_id}] start params={hparams} "
                f"(elapsed={elapsed / 3600.0:.2f}h)",
                flush=True,
            )
            trials_started += 1

            trial_seed = args.seed + trial_id
            seed_everything(trial_seed)
            g = torch.Generator()
            g.manual_seed(trial_seed)

            dataloaders = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
            }
            test_loaders = [
                DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    worker_init_fn=seed_worker,
                    generator=g,
                )
                for ds in test_datasets
            ]

            model = make_cam_model(args.num_classes).to(DEVICE)
            try:
                summary, reached = train_one_trial_timeboxed(
                    trial_id,
                    model,
                    dataloaders,
                    test_loaders,
                    hparams,
                    args,
                    sweep_start_time,
                    epoch_writer,
                )
                trial_state = "PRUNED"
                if summary["epochs_completed"] > 0 and math.isfinite(summary["best_log_optim_num"]):
                    study.tell(trial, summary["best_log_optim_num"])
                    trial_state = "COMPLETE"
                else:
                    study.tell(trial, state=TrialState.PRUNED)

                summary["trial_state"] = trial_state
                trial_writer.writerow(summary)
                epoch_f.flush()
                trial_f.flush()
                if reached:
                    budget_reached = True
                    break
            except Exception:
                study.tell(trial, state=TrialState.FAIL)
                raise
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        elapsed = time.time() - sweep_start_time
        print(
            f"\nSweep stopped | trials_started={trials_started} "
            f"| elapsed={elapsed / 3600.0:.2f}h | budget_hit={budget_reached}",
            flush=True,
        )

    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        best_blob = {
            "study_name": args.study_name,
            "storage": args.storage,
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "best_value": best_trial.value,
            "best_params": best_trial.params,
        }
        with open(best_json_path, "w") as f:
            import json
            json.dump(best_blob, f, indent=2)
        print(f"Saved Optuna best: {best_json_path}", flush=True)
    else:
        print("No completed Optuna trials; skipping best JSON.", flush=True)

    generate_scatter(epoch_csv_path, scatter_png_path)

    print("\n=== DONE ===", flush=True)
    print(f"Epoch metrics CSV: {epoch_csv_path}", flush=True)
    print(f"Trial summary CSV: {trial_csv_path}", flush=True)
    print(f"Scatter PNG: {scatter_png_path}", flush=True)
    print(f"Optuna best JSON: {best_json_path}", flush=True)


if __name__ == "__main__":
    main()
