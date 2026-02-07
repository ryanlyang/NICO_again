#!/usr/bin/env python3
"""
Optuna hyperparameter search for vanilla ERM on NICO++ (official txtlist splits).

Setup:
- Train on all domains except the target (held-out) domain(s).
- Model: ImageNet-pretrained ResNet-50, cross-entropy only (no masks / no guidance).
- Objective: maximize validation accuracy.
- Test: report held-out domain test accuracy per trial.

This is meant to be a clean "best effort ERM" baseline under the same data/split
plumbing as the guided runs.
"""

import os
import time
import copy
import json
import csv
import math
import argparse
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
import torchvision.models as models

try:
    import optuna
except Exception as exc:
    raise SystemExit("optuna is required for this script. Install it in your env.") from exc


# =============================================================================
# BASE SETTINGS
# =============================================================================
VAL_SPLIT_RATIO = 0.16
SEED = 59

# Fixed SGD training setup (per user request)
BATCH_SIZE_DEFAULT = 32
NUM_EPOCHS_DEFAULT = 30
MOMENTUM_DEFAULT = 0.9
WEIGHT_DECAY_DEFAULT = 1e-5

DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"

ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SEEDING
# =============================================================================
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# DATASET (no masks)
# =============================================================================
class NICOERM(Dataset):
    """NICO++ dataset loader (image, label) from official txtlist."""

    def __init__(self, txtdir, dataset_name, domains, phase, image_root=None, transform=None):
        self.image_root = image_root
        self.transform = transform

        from domainbed.datasets import _dataset_info

        all_names = []
        all_labels = []
        for domain in domains:
            txt_file = os.path.join(txtdir, dataset_name, f"{domain}_{phase}.txt")
            names, labels = _dataset_info(txt_file)
            all_names.extend(names)
            all_labels.extend(labels)

        self.image_paths = [self._resolve_path(p) for p in all_names]
        self.labels = all_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        if self.image_root is None:
            return path
        rel = path.lstrip(os.sep)
        root_norm = os.path.normpath(self.image_root)
        root_name = os.path.basename(root_norm)
        rel_head = rel.split(os.sep, 1)[0]
        if rel_head == root_name:
            base = os.path.dirname(root_norm)
        else:
            base = root_norm
        return os.path.normpath(os.path.join(base, rel))


def build_train_val_datasets(args, sources, data_transforms, generator):
    val_paths = [
        os.path.join(args.txtdir, args.dataset, f"{domain}_val.txt")
        for domain in sources
    ]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_dataset = NICOERM(
            args.txtdir, args.dataset, sources, "train",
            image_root=args.image_root,
            transform=data_transforms["train"],
        )
        val_dataset = NICOERM(
            args.txtdir, args.dataset, sources, "val",
            image_root=args.image_root,
            transform=data_transforms["eval"],
        )
        return train_dataset, val_dataset, has_val

    # If no official val, create deterministic split from the train list.
    full_train_base = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=None,
    )
    n_total = len(full_train_base)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val

    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    train_idx_list = train_indices.indices
    val_idx_list = val_indices.indices

    train_dataset = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=data_transforms["train"],
    )
    val_dataset = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=data_transforms["eval"],
    )

    return Subset(train_dataset, train_idx_list), Subset(val_dataset, val_idx_list), has_val


# =============================================================================
# MODEL
# =============================================================================
def make_resnet50(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base = models.resnet50(weights=weights)
    except AttributeError:
        base = models.resnet50(pretrained=True)
    in_features = base.fc.in_features
    # torchvision ResNet doesn't include dropout by default; emulate DomainBed's
    # resnet_dropout by inserting it right before the classifier.
    dropout_p = float(getattr(base, "_vanilla_dropout_p", 0.0))
    base.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, num_classes))
    return base


def make_resnet50_with_dropout(num_classes: int, dropout_p: float) -> nn.Module:
    model = make_resnet50(num_classes)
    # Stash for make_resnet50() to pick up before replacing fc.
    model._vanilla_dropout_p = float(dropout_p)  # type: ignore[attr-defined]
    # Rebuild fc with the correct dropout.
    in_features = model.fc[1].in_features  # Sequential(Dropout, Linear)
    model.fc = nn.Sequential(nn.Dropout(p=float(dropout_p)), nn.Linear(in_features, num_classes))
    return model


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def train_one_trial(model, dataloaders, dataset_sizes, args, trial=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    opt = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    for epoch in range(args.num_epochs):
        for phase in ["train", "val"]:
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_corrects = 0
            total = 0

            for batch in dataloaders[phase]:
                inputs, labels, _ = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    loss = nn.functional.cross_entropy(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if is_train:
                        loss.backward()
                        opt.step()

                running_corrects += (preds == labels).sum().item()
                total += inputs.size(0)

            if is_train:
                sch.step()

            if phase == "val":
                epoch_acc = running_corrects / max(total, 1)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())

        if trial is not None:
            # Keep reporting for dashboards; pruning is disabled in main().
            trial.report(float(best_val_acc), epoch)

    model.load_state_dict(best_wts)
    return float(best_val_acc)


@torch.no_grad()
def evaluate_test(model, test_loaders, target_domains):
    model.eval()

    results = {}
    all_correct = 0
    all_total = 0

    for test_loader, domain_name in zip(test_loaders, target_domains):
        total, correct = 0, 0
        for batch in test_loader:
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        acc = 100.0 * correct / max(total, 1)
        results[domain_name] = acc
        all_correct += correct
        all_total += total

    results["overall"] = 100.0 * all_correct / max(all_total, 1)
    return results


# =============================================================================
# OPTUNA HELPERS
# =============================================================================
def suggest_loguniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high, log=True)

def resolve_weight_decay(best_params: dict) -> float:
    """
    Best params may omit 'weight_decay' if wd_is_zero=True was chosen.
    Normalize that into an actual float.
    """
    if best_params.get("wd_is_zero", False):
        return 0.0
    if "weight_decay" in best_params:
        return float(best_params["weight_decay"])
    # Backwards compatibility if search space is changed.
    return 0.0


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def main():
    parser = argparse.ArgumentParser(description="Optuna search for vanilla ERM on NICO++")

    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR, help="Path to txt lists")
    parser.add_argument("--dataset", type=str, default="NICO", help="Dataset name")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT, help="Root directory for images")
    parser.add_argument("--target", nargs="+", required=True, help="Target (held-out) domains")
    parser.add_argument("--num_classes", type=int, default=60, help="Number of classes")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")

    # Fixed SGD training params
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--momentum", type=float, default=MOMENTUM_DEFAULT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds)")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage (e.g., sqlite:///study.db)")
    parser.add_argument("--optuna_seed", type=int, default=SEED, help="Seed for Optuna sampler")
    parser.add_argument("--load_if_exists", action="store_true", help="Reuse study if it exists")

    # Search space
    parser.add_argument("--lr_low", type=float, default=1e-5)
    parser.add_argument("--lr_high", type=float, default=3e-4)
    parser.add_argument("--dropout_choices", type=str, default="0.0,0.1,0.5", help="Comma-separated dropout ps to search.")

    # After sweep: rerun the best hyperparameters for multiple seeds.
    parser.add_argument("--rerun_best", type=int, default=1, help="After sweep, rerun best params for multiple seeds (1/0).")
    parser.add_argument("--rerun_seed_start", type=int, default=59, help="Start seed for reruns (if --rerun_seeds not set).")
    parser.add_argument("--rerun_num_seeds", type=int, default=5, help="Number of rerun seeds (if --rerun_seeds not set).")
    parser.add_argument("--rerun_seeds", type=str, default="", help="Comma-separated explicit rerun seeds.")

    args = parser.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    args.target = targets

    run_name = f"vanilla_target_{'-'.join(args.target)}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

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

    train_dataset, val_dataset, has_val = build_train_val_datasets(args, sources, data_transforms, base_g)
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

    if not has_val:
        print(f"Val split: {int(VAL_SPLIT_RATIO*100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOERM(
            args.txtdir, args.dataset, [domain], "test",
            image_root=args.image_root,
            transform=data_transforms["eval"],
        )
        for domain in args.target
    ]

    dropout_choices = [float(x) for x in args.dropout_choices.split(",") if x.strip()]

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

    trial_log_path = os.path.join(output_dir, "optuna_trials.csv")

    def log_trial(trial, best_val_acc, test_results):
        row = {
            "trial": trial.number,
            "best_val_acc": best_val_acc,
            "test_results": json.dumps(test_results),
            "params": json.dumps(trial.params),
        }
        write_header = not os.path.exists(trial_log_path)
        with open(trial_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def objective(trial):
        lr = suggest_loguniform(trial, "lr", args.lr_low, args.lr_high)
        resnet_dropout = trial.suggest_categorical("resnet_dropout", dropout_choices)

        print(
            f"[TRIAL {trial.number}] start params={{'lr': {lr}, 'momentum': {args.momentum}, "
            f"'weight_decay': {args.weight_decay}, 'batch_size': {args.batch_size}, "
            f"'resnet_dropout': {resnet_dropout}, 'num_epochs': {args.num_epochs}}}",
            flush=True,
        )

        trial_seed = args.seed + trial.number
        seed_everything(trial_seed)
        g = torch.Generator()
        g.manual_seed(trial_seed)

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            ),
            "val": DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            ),
        }

        test_loaders = [
            DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            )
            for ds in test_datasets
        ]

        model = make_resnet50_with_dropout(args.num_classes, resnet_dropout).to(DEVICE)
        trial_args = argparse.Namespace(
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
        )

        try:
            best_val_acc = train_one_trial(model, dataloaders, dataset_sizes, trial_args, trial=trial)
            test_results = evaluate_test(model, test_loaders, args.target)
            trial.set_user_attr("test_results", test_results)
            log_trial(trial, best_val_acc, test_results)
            print(f"[TRIAL {trial.number}] done best_val_acc={best_val_acc:.6f} test={test_results}", flush=True)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_val_acc

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_trial = study.best_trial
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_test_results": best_trial.user_attrs.get("test_results", None),
        "n_trials": len(study.trials),
    }

    best_path = os.path.join(output_dir, "optuna_best.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print("Best validation accuracy:", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial test results:", best["best_test_results"])
    print("Saved:", best_path)

    if int(args.rerun_best) == 1:
        seeds = parse_seeds(args.rerun_seed_start, args.rerun_num_seeds, args.rerun_seeds)
        best_lr = float(study.best_params["lr"])
        best_dropout = float(study.best_params.get("resnet_dropout", dropout_choices[0]))

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_header = [
            "seed", "lr", "momentum", "weight_decay", "batch_size", "resnet_dropout",
            "num_epochs", "best_val_acc", "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: lr={best_lr} momentum={args.momentum} weight_decay={args.weight_decay} "
            f"batch_size={args.batch_size} resnet_dropout={best_dropout} num_epochs={args.num_epochs}",
            flush=True,
        )

        for seed in seeds:
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_dataset_s, val_dataset_s, _ = build_train_val_datasets(args, sources, data_transforms, g)
            dataset_sizes_s = {"train": len(train_dataset_s), "val": len(val_dataset_s)}

            dataloaders_s = {
                "train": DataLoader(
                    train_dataset_s, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                ),
                "val": DataLoader(
                    val_dataset_s, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                ),
            }

            test_loaders = [
                DataLoader(
                    ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                )
                for ds in test_datasets
            ]

            model = make_resnet50_with_dropout(args.num_classes, best_dropout).to(DEVICE)
            run_args = argparse.Namespace(
                lr=best_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                num_epochs=args.num_epochs,
            )

            start = time.time()
            best_val_acc = train_one_trial(model, dataloaders_s, dataset_sizes_s, run_args, trial=None)
            test_results = evaluate_test(model, test_loaders, args.target)
            minutes = (time.time() - start) / 60.0

            print(
                f"[RERUN seed={seed}] best_val_acc={best_val_acc:.6f} test={test_results} "
                f"time={minutes:.1f}m",
                flush=True,
            )

            with open(rerun_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    seed,
                    best_lr,
                    args.momentum,
                    args.weight_decay,
                    args.batch_size,
                    best_dropout,
                    args.num_epochs,
                    best_val_acc,
                    json.dumps(test_results),
                    minutes,
                ])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Saved reruns:", rerun_path, flush=True)


if __name__ == "__main__":
    main()
