#!/usr/bin/env python3
"""
Optuna sweep for Vanilla ERM + SWAD (LossValley) on NICO++ with SGD.

Objective: maximize SWAD model validation accuracy (no test leakage).
After sweep: rerun best params for multiple seeds and report test.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import time

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

from domainbed.lib.swad import LossValley
from domainbed.lib import swa_utils


VAL_SPLIT_RATIO = 0.16
SEED = 59

DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"

ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    val_paths = [os.path.join(args.txtdir, args.dataset, f"{d}_val.txt") for d in sources]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_ds = NICOERM(args.txtdir, args.dataset, sources, "train", image_root=args.image_root, transform=data_transforms["train"])
        val_ds = NICOERM(args.txtdir, args.dataset, sources, "val", image_root=args.image_root, transform=data_transforms["eval"])
        return train_ds, val_ds, has_val

    full_train = NICOERM(args.txtdir, args.dataset, sources, "train", image_root=args.image_root, transform=None)
    n_total = len(full_train)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx_list = train_idx.indices
    val_idx_list = val_idx.indices

    train_ds = NICOERM(args.txtdir, args.dataset, sources, "train", image_root=args.image_root, transform=data_transforms["train"])
    val_ds = NICOERM(args.txtdir, args.dataset, sources, "train", image_root=args.image_root, transform=data_transforms["eval"])
    return Subset(train_ds, train_idx_list), Subset(val_ds, val_idx_list), has_val


def make_resnet50_with_dropout(num_classes: int, dropout_p: float) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base = models.resnet50(weights=weights)
    except AttributeError:
        base = models.resnet50(pretrained=True)
    in_features = base.fc.in_features
    base.fc = nn.Sequential(nn.Dropout(p=float(dropout_p)), nn.Linear(in_features, num_classes))
    return base


def _get_param_groups(model: nn.Module, base_lr: float, classifier_lr: float):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            classifier_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if classifier_params:
        groups.append({"params": classifier_params, "lr": classifier_lr})
    return groups


def evaluate_loss_acc(model, loader):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels, reduction="sum")
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


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


def train_swad_one_trial(model, dataloaders, args, swad_args):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    opt = optim.SGD(
        _get_param_groups(model, args.base_lr, args.classifier_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    swad = LossValley(swad_args["n_converge"], swad_args["n_tolerance"], swad_args["tolerance_ratio"])
    swad_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            opt.step()

            swad_segment.update_parameters(model, step=global_step)
            global_step += 1

        sch.step()

        val_loss, val_acc = evaluate_loss_acc(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        print(f"[Epoch {epoch}] val_acc={val_acc:.6f} val_loss={val_loss:.6f}", flush=True)

        swad.update_and_evaluate(swad_segment, val_acc, val_loss)
        if swad.dead_valley:
            print("SWAD valley is dead -> early stop!", flush=True)
            break
        swad_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

    final_model = swad.get_final_model()
    swad_val_loss, swad_val_acc = evaluate_loss_acc(final_model, val_loader)

    print(f"[SWAD] val_acc={swad_val_acc:.6f} val_loss={swad_val_loss:.6f} (best plain val_acc={best_val_acc:.6f} @ {best_epoch})", flush=True)
    return swad_val_acc, final_model


def suggest_loguniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high, log=True)


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def main():
    p = argparse.ArgumentParser(description="Optuna SWAD (vanilla ERM) on NICO++ with SGD")
    p.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)

    # fixed training config
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--resnet_dropout", type=float, default=0.0)

    # optuna settings
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--optuna_seed", type=int, default=SEED)
    p.add_argument("--load_if_exists", action="store_true")

    # sweep ranges
    p.add_argument("--base_lr_low", type=float, default=1e-5)
    p.add_argument("--base_lr_high", type=float, default=5e-2)
    p.add_argument("--classifier_lr_low", type=float, default=1e-5)
    p.add_argument("--classifier_lr_high", type=float, default=5e-2)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--tolerance_ratio_low", type=float, default=0.1)
    p.add_argument("--tolerance_ratio_high", type=float, default=0.5)
    p.add_argument("--n_converge", type=int, default=3)
    p.add_argument("--n_tolerance_min", type=int, default=3)
    p.add_argument("--n_tolerance_max", type=int, default=10)

    # rerun best
    p.add_argument("--rerun_best", type=int, default=1)
    p.add_argument("--rerun_seed_start", type=int, default=59)
    p.add_argument("--rerun_num_seeds", type=int, default=5)
    p.add_argument("--rerun_seeds", type=str, default="")

    args = p.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    run_name = f"target_{'-'.join(targets)}"
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
    if not has_val:
        print(f"Val split: {int(VAL_SPLIT_RATIO*100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOERM(args.txtdir, args.dataset, [domain], "test", image_root=args.image_root, transform=data_transforms["eval"])
        for domain in targets
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
        base_lr = suggest_loguniform(trial, "base_lr", args.base_lr_low, args.base_lr_high)
        classifier_lr = suggest_loguniform(trial, "classifier_lr", args.classifier_lr_low, args.classifier_lr_high)
        weight_decay = args.weight_decay

        tolerance_ratio = trial.suggest_float("tolerance_ratio", args.tolerance_ratio_low, args.tolerance_ratio_high)
        n_tolerance = trial.suggest_int("n_tolerance", args.n_tolerance_min, args.n_tolerance_max)

        print(
            f"[TRIAL {trial.number}] start params={{'base_lr': {base_lr}, 'classifier_lr': {classifier_lr}, "
            f"'momentum': {args.momentum}, 'weight_decay': {weight_decay} (fixed), 'batch_size': {args.batch_size}, "
            f"'resnet_dropout': {args.resnet_dropout}, 'n_converge': {args.n_converge}, "
            f"'n_tolerance': {n_tolerance}, 'tolerance_ratio': {tolerance_ratio}, "
            f"'num_epochs': {args.num_epochs}}}",
            flush=True,
        )

        trial_seed = args.seed + trial.number
        seed_everything(trial_seed)
        g = torch.Generator()
        g.manual_seed(trial_seed)

        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
            "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
        }

        test_loaders = [
            DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
            for ds in test_datasets
        ]

        model = make_resnet50_with_dropout(args.num_classes, args.resnet_dropout).to(DEVICE)
        trial_args = argparse.Namespace(
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            num_epochs=args.num_epochs,
        )
        swad_args = dict(
            n_converge=args.n_converge,
            n_tolerance=n_tolerance,
            tolerance_ratio=tolerance_ratio,
        )

        try:
            best_val_acc, swad_model = train_swad_one_trial(model, dataloaders, trial_args, swad_args)
            test_results = evaluate_test(swad_model, test_loaders, targets)
            trial.set_user_attr("test_results", test_results)
            log_trial(trial, best_val_acc, test_results)
            print(f"[TRIAL {trial.number}] done swad_val_acc={best_val_acc:.6f} test={test_results}", flush=True)
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

    print("Best SWAD val accuracy:", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial test results:", best["best_test_results"])
    print("Saved:", best_path)

    if int(args.rerun_best) == 1:
        seeds = parse_seeds(args.rerun_seed_start, args.rerun_num_seeds, args.rerun_seeds)
        best_base_lr = float(study.best_params["base_lr"])
        best_classifier_lr = float(study.best_params["classifier_lr"])
        best_dropout = float(args.resnet_dropout)
        best_weight_decay = float(args.weight_decay)
        best_n_tolerance = int(study.best_params.get("n_tolerance", args.n_tolerance_min))
        best_tolerance_ratio = float(study.best_params.get("tolerance_ratio", args.tolerance_ratio_low))

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_header = [
            "seed", "base_lr", "classifier_lr", "momentum", "weight_decay", "batch_size", "resnet_dropout",
            "n_converge", "n_tolerance", "tolerance_ratio",
            "num_epochs", "swad_val_acc", "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: base_lr={best_base_lr} classifier_lr={best_classifier_lr} "
            f"momentum={args.momentum} weight_decay={best_weight_decay} "
            f"batch_size={args.batch_size} resnet_dropout={best_dropout} "
            f"n_converge={args.n_converge} n_tolerance={best_n_tolerance} tolerance_ratio={best_tolerance_ratio}",
            flush=True,
        )

        for seed in seeds:
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_dataset_s, val_dataset_s, _ = build_train_val_datasets(args, sources, data_transforms, g)

            dataloaders_s = {
                "train": DataLoader(train_dataset_s, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
                "val": DataLoader(val_dataset_s, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
            }
            test_loaders = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
                for ds in test_datasets
            ]

            model = make_resnet50_with_dropout(args.num_classes, best_dropout).to(DEVICE)
            run_args = argparse.Namespace(
                base_lr=best_base_lr,
                classifier_lr=best_classifier_lr,
                momentum=args.momentum,
                weight_decay=best_weight_decay,
                num_epochs=args.num_epochs,
            )
            swad_args = dict(
                n_converge=args.n_converge,
                n_tolerance=best_n_tolerance,
                tolerance_ratio=best_tolerance_ratio,
            )

            start = time.time()
            swad_val_acc, swad_model = train_swad_one_trial(model, dataloaders_s, run_args, swad_args)
            test_results = evaluate_test(swad_model, test_loaders, targets)
            minutes = (time.time() - start) / 60.0

            print(
                f"[RERUN seed={seed}] swad_val_acc={swad_val_acc:.6f} test={test_results} time={minutes:.1f}m",
                flush=True,
            )

            with open(rerun_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    seed,
                    best_base_lr,
                    best_classifier_lr,
                    args.momentum,
                    best_weight_decay,
                    args.batch_size,
                    best_dropout,
                    args.n_converge,
                    best_n_tolerance,
                    best_tolerance_ratio,
                    args.num_epochs,
                    swad_val_acc,
                    json.dumps(test_results),
                    minutes,
                ])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Saved reruns:", rerun_path, flush=True)


if __name__ == "__main__":
    main()
