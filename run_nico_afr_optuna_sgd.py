#!/usr/bin/env python3
"""
Optuna hyperparameter search for AFR (Automatic Feature Reweighting) on NICO++.

AFR setup (vision style):
- Stage 1: train an ERM ResNet-50 checkpoint on D_ERM.
- Stage 2: freeze features and retrain only the last layer on D_RW using
  weighted CE where weights are based on ERM confidence:
      mu_i ~ beta_{y_i} * exp(-gamma * p_true_i)
  with optional L2 regularization to the stage-1 last-layer weights.

This script follows the NICO sweep conventions in this repo:
- train on source domains, test on held-out target domain(s)
- Optuna search with SQLite support
- rerun best hyperparameters for multiple seeds
"""

import argparse
import copy
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models

try:
    import optuna
except Exception as exc:
    raise SystemExit("optuna is required for this script. Install it in your env.") from exc


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"

VAL_SPLIT_RATIO = 0.16
SEED = 59

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
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


@dataclass
class Record:
    path: str
    label: int


class NICORecords(Dataset):
    def __init__(self, records: Sequence[Record], transform=None):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rec.label, rec.path


def read_records(txtdir: str, dataset_name: str, domains: Sequence[str], phase: str,
                 image_root: str) -> List[Record]:
    from domainbed.datasets import _dataset_info

    records: List[Record] = []
    for domain in domains:
        txt_file = os.path.join(txtdir, dataset_name, f"{domain}_{phase}.txt")
        names, labels = _dataset_info(txt_file)
        for name, label in zip(names, labels):
            records.append(Record(path=_resolve_path(name, image_root), label=int(label)))
    return records


def _split_indices(num_items: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(num_items)


def build_afr_splits(
    train_records: Sequence[Record],
    val_records_official: Sequence[Record],
    seed: int,
    val_split_ratio: float,
    erm_prop: float,
) -> Tuple[List[Record], List[Record], List[Record], bool]:
    """
    Returns (erm_records, rw_records, val_records, has_official_val).

    If official val records are available, val is taken from those and train_records
    are split into ERM/RW.
    Otherwise, train_records are split into train_pool/val first, then train_pool
    is split into ERM/RW.
    """
    records = list(train_records)
    perm = _split_indices(len(records), seed)

    has_official_val = len(val_records_official) > 0

    if has_official_val:
        val_records = list(val_records_official)
        train_pool_idx = perm
    else:
        n_val = max(1, int(val_split_ratio * len(records)))
        n_val = min(n_val, len(records) - 2)  # leave room for ERM and RW
        val_idx = perm[:n_val]
        train_pool_idx = perm[n_val:]
        val_records = [records[i] for i in val_idx]

    n_pool = len(train_pool_idx)
    n_erm = int(round(erm_prop * n_pool))
    n_erm = max(1, min(n_erm, n_pool - 1))

    erm_idx = train_pool_idx[:n_erm]
    rw_idx = train_pool_idx[n_erm:]

    erm_records = [records[i] for i in erm_idx]
    rw_records = [records[i] for i in rw_idx]

    return erm_records, rw_records, val_records, has_official_val


# -----------------------------------------------------------------------------
# Model + training
# -----------------------------------------------------------------------------
def make_resnet50(num_classes: int, dropout_p: float = 0.0) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=float(dropout_p)),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_stage1_erm(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    base_lr: float,
    classifier_lr: float,
    momentum: float,
    weight_decay: float,
) -> nn.Module:
    model.train()

    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            classifier_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    opt = optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_correct = 0
        running_total = 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            opt.zero_grad()
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()

            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += images.size(0)

        sch.step()

        acc = running_correct / max(running_total, 1)
        print(f"[STAGE1][Epoch {epoch}] train_acc={acc:.6f}", flush=True)

    return model


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract penultimate features by temporarily replacing ResNet fc with Identity.
    Returns embeddings, labels, paths on DEVICE.
    """
    was_training = model.training
    model.eval()

    original_fc = model.fc
    model.fc = nn.Identity()

    embs = []
    labels = []
    paths = []

    for images, y, p in loader:
        images = images.to(DEVICE)
        feat = model(images)
        embs.append(feat.detach())
        labels.append(y.to(DEVICE))
        paths.extend(p)

    model.fc = original_fc
    if was_training:
        model.train()

    return torch.cat(embs, dim=0), torch.cat(labels, dim=0), paths


@torch.no_grad()
def evaluate_last_layer(last_layer: nn.Linear, embeddings: torch.Tensor,
                        labels: torch.Tensor) -> float:
    logits = last_layer(embeddings)
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def evaluate_last_layer_test(
    last_layer: nn.Linear,
    test_embs: Sequence[torch.Tensor],
    test_labels: Sequence[torch.Tensor],
    target_domains: Sequence[str],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    total_correct = 0
    total_seen = 0

    for domain, emb, y in zip(target_domains, test_embs, test_labels):
        logits = last_layer(emb)
        preds = logits.argmax(dim=1)
        correct = int((preds == y).sum().item())
        seen = int(y.numel())
        acc = 100.0 * correct / max(seen, 1)
        results[domain] = acc
        total_correct += correct
        total_seen += seen

    results["overall"] = 100.0 * total_correct / max(total_seen, 1)
    return results


def compute_afr_weights(
    stage1_logits_rw: torch.Tensor,
    rw_labels: torch.Tensor,
    gamma: float,
    num_classes: int,
) -> torch.Tensor:
    probs = stage1_logits_rw.softmax(dim=1)
    p_true = probs.gather(1, rw_labels.unsqueeze(1)).squeeze(1)

    class_counts = torch.bincount(rw_labels, minlength=num_classes).float()
    beta = torch.ones_like(class_counts)
    present = class_counts > 0
    beta[present] = 1.0 / class_counts[present]

    weights = beta[rw_labels] * torch.exp(-gamma * p_true)
    weights = weights / weights.sum().clamp_min(1e-12)
    return weights


def run_stage2_afr_trial(
    init_weight: torch.Tensor,
    init_bias: torch.Tensor,
    rw_emb: torch.Tensor,
    rw_y: torch.Tensor,
    val_emb: torch.Tensor,
    val_y: torch.Tensor,
    test_embs: Sequence[torch.Tensor],
    test_labels: Sequence[torch.Tensor],
    target_domains: Sequence[str],
    stage1_logits_rw: torch.Tensor,
    num_classes: int,
    gamma: float,
    reg_coeff: float,
    stage2_lr: float,
    stage2_epochs: int,
    trial=None,
) -> Tuple[float, int, Dict[str, float], nn.Linear]:
    feat_dim = int(init_weight.shape[1])

    last_layer = nn.Linear(feat_dim, num_classes).to(DEVICE)
    with torch.no_grad():
        last_layer.weight.copy_(init_weight)
        last_layer.bias.copy_(init_bias)

    w0 = init_weight.detach().clone()
    b0 = init_bias.detach().clone()

    afr_weights = compute_afr_weights(stage1_logits_rw, rw_y, gamma=gamma, num_classes=num_classes)

    opt = optim.SGD(last_layer.parameters(), lr=stage2_lr, momentum=0.0, weight_decay=0.0)

    best_val_acc = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(last_layer.state_dict())

    for epoch in range(stage2_epochs):
        last_layer.train()
        opt.zero_grad()

        logits_rw = last_layer(rw_emb)
        ce_vec = nn.functional.cross_entropy(logits_rw, rw_y, reduction="none")
        weighted_ce = torch.sum(afr_weights * ce_vec)

        reg = ((last_layer.weight - w0).pow(2).sum() + (last_layer.bias - b0).pow(2).sum())
        loss = weighted_ce + reg_coeff * reg

        loss.backward()
        clip_grad_norm_(last_layer.parameters(), max_norm=1.0)
        opt.step()

        last_layer.eval()
        val_acc = evaluate_last_layer(last_layer, val_emb, val_y)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(last_layer.state_dict())

        if trial is not None:
            trial.report(float(best_val_acc), epoch)

    last_layer.load_state_dict(best_state)
    test_results = evaluate_last_layer_test(last_layer, test_embs, test_labels, target_domains)

    return float(best_val_acc), int(best_epoch), test_results, last_layer


# -----------------------------------------------------------------------------
# Optuna helpers
# -----------------------------------------------------------------------------
def suggest_gamma(trial, low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(trial.suggest_float("gamma", low, high))


def suggest_reg_coeff(trial, reg_choices: Sequence[float]) -> float:
    if len(reg_choices) == 1:
        return float(reg_choices[0])
    return float(trial.suggest_categorical("reg_coeff", list(reg_choices)))


def parse_float_csv(values: str) -> List[float]:
    out = []
    for token in values.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("Expected at least one float in comma-separated list.")
    return out


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str) -> List[int]:
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


# -----------------------------------------------------------------------------
# Pipeline setup
# -----------------------------------------------------------------------------
def build_transforms():
    return {
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


def load_split_records(args, sources: Sequence[str], targets: Sequence[str], seed: int):
    train_records = read_records(args.txtdir, args.dataset, sources, "train", args.image_root)

    source_val_paths = [
        os.path.join(args.txtdir, args.dataset, f"{d}_val.txt")
        for d in sources
    ]
    has_official_val = all(os.path.exists(p) for p in source_val_paths)
    val_records_official: List[Record] = []
    if has_official_val:
        val_records_official = read_records(args.txtdir, args.dataset, sources, "val", args.image_root)

    erm_records, rw_records, val_records, has_official_val = build_afr_splits(
        train_records=train_records,
        val_records_official=val_records_official,
        seed=seed,
        val_split_ratio=args.val_split_ratio,
        erm_prop=args.erm_train_prop,
    )

    test_records_per_domain = [
        read_records(args.txtdir, args.dataset, [domain], "test", args.image_root)
        for domain in targets
    ]

    return erm_records, rw_records, val_records, test_records_per_domain, has_official_val


def run_single_seed_pipeline(
    args,
    seed: int,
    sources: Sequence[str],
    targets: Sequence[str],
):
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    data_transforms = build_transforms()

    erm_records, rw_records, val_records, test_records_per_domain, has_official_val = load_split_records(
        args, sources, targets, seed
    )

    if not has_official_val:
        print(f"Val split: {int(args.val_split_ratio*100)}% split from train (no *_val.txt found)")

    erm_train_ds = NICORecords(erm_records, transform=data_transforms["train"])
    rw_eval_ds = NICORecords(rw_records, transform=data_transforms["eval"])
    val_eval_ds = NICORecords(val_records, transform=data_transforms["eval"])
    test_eval_ds = [NICORecords(rec, transform=data_transforms["eval"]) for rec in test_records_per_domain]

    erm_loader = DataLoader(
        erm_train_ds,
        batch_size=args.stage1_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    rw_loader = DataLoader(
        rw_eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loaders = [
        DataLoader(
            ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        for ds in test_eval_ds
    ]

    stage1_base_lr = args.stage1_base_lr if args.stage1_base_lr is not None else args.stage1_lr
    stage1_classifier_lr = (
        args.stage1_classifier_lr if args.stage1_classifier_lr is not None else stage1_base_lr
    )

    model = make_resnet50(args.num_classes, dropout_p=args.stage1_dropout).to(DEVICE)
    print(
        f"Stage1 ERM: seed={seed}, train={len(erm_train_ds)}, rw={len(rw_eval_ds)}, val={len(val_eval_ds)}, "
        f"epochs={args.stage1_epochs}, base_lr={stage1_base_lr}, classifier_lr={stage1_classifier_lr}, "
        f"dropout={args.stage1_dropout}, wd={args.stage1_weight_decay}",
        flush=True,
    )
    model = train_stage1_erm(
        model,
        erm_loader,
        num_epochs=args.stage1_epochs,
        base_lr=stage1_base_lr,
        classifier_lr=stage1_classifier_lr,
        momentum=args.stage1_momentum,
        weight_decay=args.stage1_weight_decay,
    )

    stage1_classifier = copy.deepcopy(model.fc).to(DEVICE).eval()
    if isinstance(stage1_classifier, nn.Sequential):
        stage1_linear = stage1_classifier[-1]
    else:
        stage1_linear = stage1_classifier
    if not isinstance(stage1_linear, nn.Linear):
        raise RuntimeError(f"Expected final classifier to be nn.Linear, got {type(stage1_linear)}")
    init_weight = stage1_linear.weight.detach().clone()
    init_bias = stage1_linear.bias.detach().clone()

    rw_emb, rw_y, _ = extract_embeddings(model, rw_loader)
    val_emb, val_y, _ = extract_embeddings(model, val_loader)

    test_embs = []
    test_labels = []
    for loader in test_loaders:
        emb, y, _ = extract_embeddings(model, loader)
        test_embs.append(emb)
        test_labels.append(y)

    with torch.no_grad():
        stage1_logits_rw = stage1_classifier(rw_emb)

    del model
    del stage1_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache = {
        "init_weight": init_weight,
        "init_bias": init_bias,
        "rw_emb": rw_emb,
        "rw_y": rw_y,
        "val_emb": val_emb,
        "val_y": val_y,
        "test_embs": test_embs,
        "test_labels": test_labels,
        "stage1_logits_rw": stage1_logits_rw,
    }

    return cache


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna AFR (2-stage) on NICO++")

    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    parser.add_argument("--dataset", type=str, default="NICO")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--target", nargs="+", required=True)
    parser.add_argument("--num_classes", type=int, default=60)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)

    # Split config
    parser.add_argument("--val_split_ratio", type=float, default=VAL_SPLIT_RATIO,
                        help="Used only if source *_val.txt files are unavailable.")
    parser.add_argument("--erm_train_prop", type=float, default=0.8,
                        help="Fraction of source train pool used for stage-1 ERM (rest goes to D_RW).")

    # Stage 1 (ERM)
    parser.add_argument("--stage1_epochs", type=int, default=30)
    parser.add_argument("--stage1_batch_size", type=int, default=32)
    parser.add_argument("--stage1_base_lr", type=float, default=None,
                        help="Stage-1 backbone LR (defaults to --stage1_lr if unset).")
    parser.add_argument("--stage1_classifier_lr", type=float, default=None,
                        help="Stage-1 classifier LR (defaults to stage1 backbone LR if unset).")
    parser.add_argument("--stage1_lr", type=float, default=3e-3)
    parser.add_argument("--stage1_dropout", type=float, default=0.0)
    parser.add_argument("--stage1_momentum", type=float, default=0.9)
    parser.add_argument("--stage1_weight_decay", type=float, default=1e-4)

    # Stage 2 (AFR last-layer retraining)
    parser.add_argument("--stage2_epochs", type=int, default=500)
    parser.add_argument("--stage2_lr", type=float, default=1e-2)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    # Optuna
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--optuna_seed", type=int, default=SEED)
    parser.add_argument("--load_if_exists", action="store_true")

    # AFR search space (Waterbirds-style)
    parser.add_argument("--gamma_low", type=float, default=4.0)
    parser.add_argument("--gamma_high", type=float, default=20.0)
    parser.add_argument("--reg_coeff_choices", type=str, default="0.0,0.1,0.2,0.3,0.4")

    # Rerun best with multiple seeds
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

    run_name = f"afr_target_{'-'.join(targets)}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    reg_coeff_choices = parse_float_csv(args.reg_coeff_choices)
    stage1_base_lr = args.stage1_base_lr if args.stage1_base_lr is not None else args.stage1_lr
    stage1_classifier_lr = (
        args.stage1_classifier_lr if args.stage1_classifier_lr is not None else stage1_base_lr
    )

    print(
        f"Running AFR sweep | targets={targets}, sources={sources}, device={DEVICE}, "
        f"stage1=(epochs={args.stage1_epochs}, base_lr={stage1_base_lr}, "
        f"classifier_lr={stage1_classifier_lr}, dropout={args.stage1_dropout}, "
        f"wd={args.stage1_weight_decay}), "
        f"stage2=(epochs={args.stage2_epochs}, lr={args.stage2_lr}), "
        f"search: gamma=[{args.gamma_low}, {args.gamma_high}] reg_coeff={reg_coeff_choices}",
        flush=True,
    )

    # Build stage-1 checkpoint + embeddings once for the sweep seed.
    sweep_cache = run_single_seed_pipeline(args, seed=args.seed, sources=sources, targets=targets)

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

    def log_trial(trial_num: int, params: Dict, best_val_acc: float, best_epoch: int,
                  test_results: Dict[str, float], minutes: float):
        row = {
            "trial": trial_num,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "test_results": json.dumps(test_results),
            "params": json.dumps(params),
            "minutes": minutes,
        }
        write_header = not os.path.exists(trial_log_path)
        with open(trial_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def objective(trial):
        gamma = suggest_gamma(trial, args.gamma_low, args.gamma_high)
        reg_coeff = suggest_reg_coeff(trial, reg_coeff_choices)

        params = {
            "gamma": gamma,
            "reg_coeff": reg_coeff,
            "stage2_lr": args.stage2_lr,
            "stage2_epochs": args.stage2_epochs,
        }
        print(f"[TRIAL {trial.number}] start params={params}", flush=True)

        t0 = time.time()
        best_val_acc, best_epoch, test_results, last_layer = run_stage2_afr_trial(
            init_weight=sweep_cache["init_weight"],
            init_bias=sweep_cache["init_bias"],
            rw_emb=sweep_cache["rw_emb"],
            rw_y=sweep_cache["rw_y"],
            val_emb=sweep_cache["val_emb"],
            val_y=sweep_cache["val_y"],
            test_embs=sweep_cache["test_embs"],
            test_labels=sweep_cache["test_labels"],
            target_domains=targets,
            stage1_logits_rw=sweep_cache["stage1_logits_rw"],
            num_classes=args.num_classes,
            gamma=gamma,
            reg_coeff=reg_coeff,
            stage2_lr=args.stage2_lr,
            stage2_epochs=args.stage2_epochs,
            trial=trial,
        )
        minutes = (time.time() - t0) / 60.0

        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("test_results", test_results)
        log_trial(trial.number, params, best_val_acc, best_epoch, test_results, minutes)

        print(
            f"[TRIAL {trial.number}] done best_val_acc={best_val_acc:.6f} best_epoch={best_epoch} "
            f"test={test_results} time={minutes:.1f}m",
            flush=True,
        )

        del last_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_acc

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_trial = study.best_trial
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_epoch": best_trial.user_attrs.get("best_epoch", None),
        "best_test_results": best_trial.user_attrs.get("test_results", None),
        "n_trials": len(study.trials),
        "stage1": {
            "epochs": args.stage1_epochs,
            "batch_size": args.stage1_batch_size,
            "base_lr": stage1_base_lr,
            "classifier_lr": stage1_classifier_lr,
            "dropout": args.stage1_dropout,
            "momentum": args.stage1_momentum,
            "weight_decay": args.stage1_weight_decay,
            "erm_train_prop": args.erm_train_prop,
        },
        "stage2": {
            "epochs": args.stage2_epochs,
            "lr": args.stage2_lr,
            "reg_coeff_choices": reg_coeff_choices,
            "gamma_low": args.gamma_low,
            "gamma_high": args.gamma_high,
        },
    }

    best_path = os.path.join(output_dir, "optuna_best.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    print("Best validation accuracy:", study.best_value)
    print("Best params:", study.best_params)
    print("Best trial test results:", best["best_test_results"])
    print("Saved:", best_path)

    if int(args.rerun_best) == 1:
        rerun_seeds = parse_seeds(args.rerun_seed_start, args.rerun_num_seeds, args.rerun_seeds)
        best_gamma = float(study.best_params.get("gamma", args.gamma_low))
        best_reg = float(study.best_params.get("reg_coeff", reg_coeff_choices[0]))

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        header = [
            "seed", "gamma", "reg_coeff", "stage1_epochs", "stage2_epochs",
            "best_val_acc", "best_epoch", "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(header)

        print(f"\n=== Rerun best params for {len(rerun_seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: gamma={best_gamma} reg_coeff={best_reg} stage2_lr={args.stage2_lr}",
            flush=True,
        )

        for seed in rerun_seeds:
            t0 = time.time()

            cache = run_single_seed_pipeline(args, seed=seed, sources=sources, targets=targets)
            best_val_acc, best_epoch, test_results, last_layer = run_stage2_afr_trial(
                init_weight=cache["init_weight"],
                init_bias=cache["init_bias"],
                rw_emb=cache["rw_emb"],
                rw_y=cache["rw_y"],
                val_emb=cache["val_emb"],
                val_y=cache["val_y"],
                test_embs=cache["test_embs"],
                test_labels=cache["test_labels"],
                target_domains=targets,
                stage1_logits_rw=cache["stage1_logits_rw"],
                num_classes=args.num_classes,
                gamma=best_gamma,
                reg_coeff=best_reg,
                stage2_lr=args.stage2_lr,
                stage2_epochs=args.stage2_epochs,
                trial=None,
            )
            minutes = (time.time() - t0) / 60.0

            with open(rerun_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    seed,
                    best_gamma,
                    best_reg,
                    args.stage1_epochs,
                    args.stage2_epochs,
                    f"{best_val_acc:.8f}",
                    best_epoch,
                    json.dumps(test_results),
                    f"{minutes:.2f}",
                ])

            print(
                f"[RERUN seed={seed}] best_val_acc={best_val_acc:.6f} best_epoch={best_epoch} "
                f"test={test_results} time={minutes:.1f}m",
                flush=True,
            )

            del last_layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Saved reruns: {rerun_path}", flush=True)


if __name__ == "__main__":
    main()
