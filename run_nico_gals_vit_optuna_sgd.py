#!/usr/bin/env python3
"""
Optuna hyperparameter search for GALS-style training on NICO++ (official txtlist splits).

Setup:
- Single held-out target domain(s); train on remaining domains.
- Model: ImageNet-pretrained ResNet-50.
- Optimizer: SGD, 30 epochs (default), with base/classifier param groups.
- Loss: cross-entropy + lambda * RRR-style gradient-outside loss using precomputed
        CLIP ViT attention maps.
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
from typing import Dict, List, Tuple

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
WEIGHT_DECAY_DEFAULT = 1e-5

DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
DEFAULT_ATTENTION_ROOT = "/home/ryreu/guided_cnn/NICO_runs/attention_maps/nico_gals_vit_b32"

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
    rel = path.lstrip(os.sep)
    root_norm = os.path.normpath(image_root)
    root_name = os.path.basename(root_norm)
    rel_head = rel.split(os.sep, 1)[0]
    if rel_head == root_name:
        base = os.path.dirname(root_norm)
    else:
        base = root_norm
    return os.path.normpath(os.path.join(base, rel))


def _relative_from_image_root(path: str, image_root: str) -> str:
    root_norm = os.path.normpath(image_root)
    path_norm = os.path.normpath(path)
    try:
        rel = os.path.relpath(path_norm, root_norm)
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass
    parts = path_norm.split(os.sep)
    if len(parts) >= 3:
        return os.path.join(parts[-3], parts[-2], parts[-1])
    return os.path.basename(path_norm)


def _normalize_map(att: torch.Tensor) -> torch.Tensor:
    # att: (1,H,W) or (H,W)
    if att.ndim == 2:
        att = att.unsqueeze(0)
    flat = att.view(1, -1)
    maxv = flat.max(dim=1, keepdim=True)[0]
    maxv[maxv == 0] = 1.0
    flat = flat - flat.min(dim=1, keepdim=True)[0]
    flat = flat / maxv
    return flat.view_as(att)


def _combine_prompt_attention(attn_tensor: torch.Tensor) -> torch.Tensor:
    """
    Combine prompt-wise maps with GALS-style 'average non-zero' behavior.
    Input can be:
      - (K,1,H,W)
      - (K,H,W)
      - (1,H,W)
      - (H,W)
    Returns:
      - (1,H,W), normalized to [0,1]
    """
    if attn_tensor.ndim == 4:
        maps = attn_tensor[:, 0, :, :]
    elif attn_tensor.ndim == 3:
        maps = attn_tensor
    elif attn_tensor.ndim == 2:
        maps = attn_tensor.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attention shape: {tuple(attn_tensor.shape)}")

    valid = []
    for i in range(maps.shape[0]):
        if not torch.all(maps[i] == 0):
            valid.append(maps[i])

    if len(valid) == 0:
        out = torch.zeros(1, maps.shape[-2], maps.shape[-1], dtype=torch.float32)
    else:
        stacked = torch.stack(valid, dim=0).float()
        out = stacked.mean(dim=0, keepdim=True)

    out = _normalize_map(out)
    return torch.clamp(out, 0.0, 1.0)


class NICOGALS(Dataset):
    """
    NICO loader that returns:
      image tensor, label, attention tensor (1,H,W), path
    """
    def __init__(self, txtdir, dataset_name, domains, phase, image_root, attention_root,
                 image_transform=None):
        self.image_root = image_root
        self.attention_root = attention_root
        self.image_transform = image_transform

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

    def _load_attention(self, img_path: str, fallback_size=(7, 7)) -> Image.Image:
        rel = _relative_from_image_root(img_path, self.image_root)
        att_path = os.path.join(self.attention_root, os.path.splitext(rel)[0] + ".pth")

        if not os.path.exists(att_path):
            arr = np.zeros(fallback_size, dtype=np.uint8)
            return Image.fromarray(arr, mode="L")

        payload = torch.load(att_path, map_location="cpu")
        if isinstance(payload, dict):
            if "attentions" in payload:
                att = payload["attentions"]
            elif "unnormalized_attentions" in payload:
                att = payload["unnormalized_attentions"]
            else:
                att = None
        elif torch.is_tensor(payload):
            att = payload
        else:
            att = None

        if att is None:
            arr = np.zeros(fallback_size, dtype=np.uint8)
            return Image.fromarray(arr, mode="L")

        combined = _combine_prompt_attention(att.float())  # (1,H,W), [0,1]
        arr = (combined.squeeze(0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    def _apply_train_transforms(self, image: Image.Image, att_img: Image.Image):
        import torchvision.transforms.functional as TF
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        att_img = TF.resized_crop(att_img, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)

        if random.random() > 0.5:
            image = TF.hflip(image)
            att_img = TF.hflip(att_img)

        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        att = TF.to_tensor(att_img)
        att = torch.clamp(att, 0.0, 1.0)
        return image, att

    def _apply_eval_transforms(self, image: Image.Image, att_img: Image.Image):
        import torchvision.transforms.functional as TF
        image = TF.resize(image, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        att_img = TF.resize(att_img, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        att = TF.to_tensor(att_img)
        att = torch.clamp(att, 0.0, 1.0)
        return image, att

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        att_img = self._load_attention(img_path)

        if self.image_transform is not None and "RandomResizedCrop" in str(self.image_transform):
            image, att = self._apply_train_transforms(image, att_img)
        else:
            image, att = self._apply_eval_transforms(image, att_img)

        return image, label, att, img_path


def build_train_val_datasets(args, sources, data_transforms, generator):
    val_paths = [os.path.join(args.txtdir, args.dataset, f"{domain}_val.txt") for domain in sources]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_ds = NICOGALS(
            args.txtdir, args.dataset, sources, "train",
            image_root=args.image_root,
            attention_root=args.attention_root,
            image_transform=data_transforms["train"],
        )
        val_ds = NICOGALS(
            args.txtdir, args.dataset, sources, "val",
            image_root=args.image_root,
            attention_root=args.attention_root,
            image_transform=data_transforms["eval"],
        )
        return train_ds, val_ds, has_val

    full_base = NICOGALS(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        attention_root=args.attention_root,
        image_transform=None,
    )
    n_total = len(full_base)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val

    train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx_list = train_indices.indices
    val_idx_list = val_indices.indices

    train_ds = NICOGALS(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        attention_root=args.attention_root,
        image_transform=data_transforms["train"],
    )
    val_ds = NICOGALS(
        args.txtdir, args.dataset, sources, "train",
        image_root=args.image_root,
        attention_root=args.attention_root,
        image_transform=data_transforms["eval"],
    )
    return Subset(train_ds, train_idx_list), Subset(val_ds, val_idx_list), has_val


class NICOERM(Dataset):
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


def make_resnet50_with_dropout(num_classes: int, dropout_p: float) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except AttributeError:
        model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=float(dropout_p)), nn.Linear(in_features, num_classes))
    return model


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


def compute_gals_loss(model, inputs, labels, att_map, gals_lambda, grad_criterion):
    outputs = model(inputs)
    cls_loss = nn.functional.cross_entropy(outputs, labels)
    dy_dx = torch.autograd.grad(cls_loss, inputs, retain_graph=True, create_graph=True)[0]
    att_rgb = att_map.expand_as(inputs)
    if str(grad_criterion).upper() == "L2":
        att_loss = nn.functional.mse_loss(dy_dx, dy_dx * att_rgb)
    else:
        att_loss = nn.functional.l1_loss(dy_dx, dy_dx * att_rgb)
    total_loss = cls_loss + gals_lambda * att_loss
    return total_loss, cls_loss, att_loss, outputs


def train_one_trial(model, dataloaders, args, trial=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    opt = optim.SGD(
        _get_param_groups(model, args.base_lr, args.classifier_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    for epoch in range(args.num_epochs):
        # train
        model.train()
        train_correct = 0
        train_total = 0
        for batch in dataloaders["train"]:
            inputs, labels, att, _ = batch
            inputs = inputs.to(DEVICE).requires_grad_(True)
            labels = labels.to(DEVICE)
            att = att.to(DEVICE)

            opt.zero_grad()
            total_loss, _, _, outputs = compute_gals_loss(
                model, inputs, labels, att, args.gals_lambda, args.grad_criterion
            )
            total_loss.backward()
            opt.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)

        sch.step()

        # val (accuracy only)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in dataloaders["val"]:
                inputs, labels, _, _ = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_acc = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        print(
            f"[Epoch {epoch}] train_acc={train_acc:.6f} val_acc={val_acc:.6f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
        total = 0
        correct = 0
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
    parser = argparse.ArgumentParser(description="Optuna GALS-ViT attention SGD on NICO++")
    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    parser.add_argument("--dataset", type=str, default="NICO")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--attention_root", type=str, default=DEFAULT_ATTENTION_ROOT)
    parser.add_argument("--target", nargs="+", required=True)
    parser.add_argument("--num_classes", type=int, default=60)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)

    # Fixed run style
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--momentum", type=float, default=MOMENTUM_DEFAULT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY_DEFAULT)

    # Optuna
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--optuna_seed", type=int, default=SEED)
    parser.add_argument("--load_if_exists", action="store_true")

    # Search space
    parser.add_argument("--base_lr_low", type=float, default=1e-5)
    parser.add_argument("--base_lr_high", type=float, default=5e-2)
    parser.add_argument("--classifier_lr_low", type=float, default=1e-5)
    parser.add_argument("--classifier_lr_high", type=float, default=5e-2)
    parser.add_argument("--resnet_dropout", type=float, default=0.1)
    parser.add_argument("--grad_criterion_choices", type=str, default="L1,L2")
    parser.add_argument("--gals_lambda_low", type=float, default=1e2)
    parser.add_argument("--gals_lambda_high", type=float, default=1e4)

    # Rerun best
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

    if not os.path.isdir(args.attention_root):
        raise FileNotFoundError(f"Attention root not found: {args.attention_root}")

    run_name = f"gals_vit_target_{'-'.join(args.target)}"
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
        print(f"Val split: {int(VAL_SPLIT_RATIO * 100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOERM(
            args.txtdir, args.dataset, [domain], "test",
            image_root=args.image_root,
            transform=data_transforms["eval"],
        )
        for domain in args.target
    ]

    grad_criterion_choices = [x.strip().upper() for x in args.grad_criterion_choices.split(",") if x.strip()]
    if not grad_criterion_choices:
        grad_criterion_choices = ["L1", "L2"]
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
        gals_lambda = suggest_loguniform(trial, "gals_lambda", args.gals_lambda_low, args.gals_lambda_high)
        grad_criterion = trial.suggest_categorical("grad_criterion", grad_criterion_choices)
        batch_size = args.batch_size
        weight_decay = args.weight_decay

        print(
            f"[TRIAL {trial.number}] start params={{'base_lr': {base_lr}, 'classifier_lr': {classifier_lr}, "
            f"'weight_decay': {weight_decay} (fixed), 'batch_size': {batch_size} (fixed), "
            f"'resnet_dropout': {args.resnet_dropout} (fixed), 'grad_criterion': '{grad_criterion}', "
            f"'gals_lambda': {gals_lambda}, 'momentum': {args.momentum}, 'num_epochs': {args.num_epochs}}}",
            flush=True,
        )

        trial_seed = args.seed + trial.number
        seed_everything(trial_seed)
        g = torch.Generator()
        g.manual_seed(trial_seed)

        dataloaders = {
            "train": DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            ),
            "val": DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            ),
        }

        test_loaders = [
            DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
            )
            for ds in test_datasets
        ]

        model = make_resnet50_with_dropout(args.num_classes, args.resnet_dropout).to(DEVICE)
        trial_args = argparse.Namespace(
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            gals_lambda=gals_lambda,
            grad_criterion=grad_criterion,
            momentum=args.momentum,
            weight_decay=weight_decay,
            num_epochs=args.num_epochs,
        )

        try:
            best_val_acc = train_one_trial(model, dataloaders, trial_args, trial=trial)
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
        best_gals_lambda = float(study.best_params["gals_lambda"])
        best_dropout = float(args.resnet_dropout)
        best_grad_criterion = str(study.best_params.get("grad_criterion", "L1")).upper()
        best_batch_size = int(args.batch_size)
        best_weight_decay = float(args.weight_decay)

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_header = [
            "seed", "base_lr", "classifier_lr", "momentum", "weight_decay",
            "batch_size", "resnet_dropout", "grad_criterion", "gals_lambda", "num_epochs",
            "best_val_acc", "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: base_lr={best_base_lr} classifier_lr={best_classifier_lr} "
            f"momentum={args.momentum} weight_decay={best_weight_decay} batch_size={best_batch_size} "
            f"resnet_dropout={best_dropout} grad_criterion={best_grad_criterion} "
            f"gals_lambda={best_gals_lambda} num_epochs={args.num_epochs}",
            flush=True,
        )

        for seed in seeds:
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_dataset_s, val_dataset_s, _ = build_train_val_datasets(args, sources, data_transforms, g)
            dataloaders_s = {
                "train": DataLoader(
                    train_dataset_s, batch_size=best_batch_size, shuffle=True,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                ),
                "val": DataLoader(
                    val_dataset_s, batch_size=best_batch_size, shuffle=False,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                ),
            }
            test_loaders_s = [
                DataLoader(
                    ds, batch_size=best_batch_size, shuffle=False,
                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
                )
                for ds in test_datasets
            ]

            model = make_resnet50_with_dropout(args.num_classes, best_dropout).to(DEVICE)
            run_args = argparse.Namespace(
                base_lr=best_base_lr,
                classifier_lr=best_classifier_lr,
                gals_lambda=best_gals_lambda,
                grad_criterion=best_grad_criterion,
                momentum=args.momentum,
                weight_decay=best_weight_decay,
                num_epochs=args.num_epochs,
            )

            start = time.time()
            best_val_acc = train_one_trial(model, dataloaders_s, run_args, trial=None)
            test_results = evaluate_test(model, test_loaders_s, args.target)
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
                    best_batch_size,
                    best_dropout,
                    best_grad_criterion,
                    best_gals_lambda,
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
