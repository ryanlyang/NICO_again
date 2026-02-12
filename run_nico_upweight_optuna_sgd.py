#!/usr/bin/env python3
"""
Optuna hyperparameter search for UpWeight ERM on NICO++ (official txtlist splits).

Setup:
- Single held-out target domain(s); train on remaining domains.
- Model: ImageNet-pretrained ResNet-50.
- Optimizer: SGD with base/classifier parameter groups.
- Loss: class-upweighted cross-entropy (inverse frequency from train split).
- Objective: maximize validation accuracy.
- Test: report held-out domain test accuracy per trial.
"""

import argparse
import copy
import csv
import json
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


VAL_SPLIT_RATIO = 0.16
SEED = 59

BATCH_SIZE_DEFAULT = 32
NUM_EPOCHS_DEFAULT = 30
MOMENTUM_DEFAULT = 0.9

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


def _resolve_path(path: str, image_root: str) -> str:
    if os.path.isabs(path):
        return path
    if image_root is None:
        return path
    rel = path.lstrip(os.sep)
    root_norm = os.path.normpath(image_root)
    root_name = os.path.basename(root_norm)
    rel_head = rel.split(os.sep, 1)[0]
    if rel_head == root_name:
        base = os.path.dirname(root_norm)
    else:
        base = root_norm
    return os.path.normpath(os.path.join(base, rel))


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

        self.image_paths = [_resolve_path(p, image_root) for p in all_names]
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


def build_train_val_datasets(args, sources, data_transforms, generator):
    val_paths = [os.path.join(args.txtdir, args.dataset, f"{domain}_val.txt") for domain in sources]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_ds = NICOERM(
            args.txtdir, args.dataset, sources, "train",
            image_root=args.image_root,
            transform=data_transforms["train"],
        )
        val_ds = NICOERM(
            args.txtdir, args.dataset, sources, "val",
            image_root=args.image_root,
            transform=data_transforms["eval"],
        )
        return train_ds, val_ds, has_val

    full_base = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=None,
    )
    n_total = len(full_base)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val

    train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx_list = train_indices.indices
    val_idx_list = val_indices.indices

    train_ds = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=data_transforms["train"],
    )
    val_ds = NICOERM(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        transform=data_transforms["eval"],
    )
    return Subset(train_ds, train_idx_list), Subset(val_ds, val_idx_list), has_val


def _extract_labels(dataset):
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [base.labels[i] for i in dataset.indices]
    return list(dataset.labels)


def _compute_upweight_vector(train_dataset, num_classes):
    labels = _extract_labels(train_dataset)
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)

    present = counts > 0
    if present.any():
        inv = 1.0 / counts[present]
        inv = inv / inv.mean()  # keep average present-class weight at 1.0
        weights[present] = inv

    # Keep absent classes neutral (won't be used in train anyway).
    weights[~present] = 1.0
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def make_resnet50(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base = models.resnet50(weights=weights)
    except AttributeError:
        base = models.resnet50(pretrained=True)
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, num_classes)
    return base


def _get_param_groups(model: nn.Module, base_lr: float, classifier_lr: float):
    base_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc.") or ".fc." in name:
            classifier_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    groups.append({"params": classifier_params, "lr": classifier_lr})
    return groups


def train_one_trial(model, dataloaders, dataset_sizes, args, class_weights, trial=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    opt = optim.SGD(
        _get_param_groups(model, args.base_lr, args.classifier_lr),
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
                    if is_train:
                        loss = nn.functional.cross_entropy(outputs, labels, weight=class_weights)
                    else:
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


def suggest_loguniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high, log=True)


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def main():
    parser = argparse.ArgumentParser(description="Optuna search for UpWeight ERM on NICO++ with SGD")

    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    parser.add_argument("--dataset", type=str, default="NICO")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--target", nargs="+", required=True)
    parser.add_argument("--num_classes", type=int, default=60)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)

    # fixed training setup
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--momentum", type=float, default=MOMENTUM_DEFAULT)

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--optuna_seed", type=int, default=SEED)
    parser.add_argument("--load_if_exists", action="store_true")

    # Search space (requested)
    parser.add_argument("--base_lr_low", type=float, default=1e-5)
    parser.add_argument("--base_lr_high", type=float, default=1e-3)
    parser.add_argument("--classifier_lr_low", type=float, default=1e-4)
    parser.add_argument("--classifier_lr_high", type=float, default=1e-2)
    parser.add_argument("--weight_decay_low", type=float, default=1e-6)
    parser.add_argument("--weight_decay_high", type=float, default=1e-3)

    # rerun best
    parser.add_argument("--rerun_best", type=int, default=1)
    parser.add_argument("--rerun_seed_start", type=int, default=59)
    parser.add_argument("--rerun_num_seeds", type=int, default=5)
    parser.add_argument("--rerun_seeds", type=str, default="")

    args = parser.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    args.target = targets

    run_name = f"upweight_target_{'-'.join(args.target)}"
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
        print(f"Val split: {int(VAL_SPLIT_RATIO * 100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOERM(
            args.txtdir, args.dataset, [domain], "test",
            image_root=args.image_root,
            transform=data_transforms["eval"],
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
        weight_decay = suggest_loguniform(trial, "weight_decay", args.weight_decay_low, args.weight_decay_high)

        print(
            f"[TRIAL {trial.number}] start params={{'base_lr': {base_lr}, 'classifier_lr': {classifier_lr}, "
            f"'weight_decay': {weight_decay}, 'momentum': {args.momentum}, "
            f"'batch_size': {args.batch_size}, 'num_epochs': {args.num_epochs}}}",
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

        class_weights = _compute_upweight_vector(train_dataset, args.num_classes)

        model = make_resnet50(args.num_classes).to(DEVICE)
        trial_args = argparse.Namespace(
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            momentum=args.momentum,
            weight_decay=weight_decay,
            num_epochs=args.num_epochs,
        )

        try:
            best_val_acc = train_one_trial(model, dataloaders, dataset_sizes, trial_args, class_weights, trial=trial)
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

        best_base_lr = float(study.best_params["base_lr"])
        best_classifier_lr = float(study.best_params["classifier_lr"])
        best_weight_decay = float(study.best_params["weight_decay"])

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_header = [
            "seed", "base_lr", "classifier_lr", "momentum", "weight_decay", "batch_size",
            "num_epochs", "best_val_acc", "test_results_json", "minutes"
        ]
        if not os.path.exists(rerun_path):
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: base_lr={best_base_lr} classifier_lr={best_classifier_lr} "
            f"weight_decay={best_weight_decay} momentum={args.momentum} "
            f"batch_size={args.batch_size} num_epochs={args.num_epochs}",
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

            class_weights = _compute_upweight_vector(train_dataset_s, args.num_classes)

            model = make_resnet50(args.num_classes).to(DEVICE)
            run_args = argparse.Namespace(
                base_lr=best_base_lr,
                classifier_lr=best_classifier_lr,
                momentum=args.momentum,
                weight_decay=best_weight_decay,
                num_epochs=args.num_epochs,
            )

            start = time.time()
            best_val_acc = train_one_trial(model, dataloaders_s, dataset_sizes_s, run_args, class_weights, trial=None)
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
                    best_base_lr,
                    best_classifier_lr,
                    args.momentum,
                    best_weight_decay,
                    args.batch_size,
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
