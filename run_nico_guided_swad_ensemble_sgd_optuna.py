#!/usr/bin/env python3
"""
Optuna sweep for Guided + two-phase SWAD ensemble on NICO++ with SGD.

Training builds two SWAD models:
- pre-attention phase SWAD
- post-attention phase SWAD

Validation/test predictions are an ensemble of the two models' logits.
Objective: maximize ensemble validation accuracy.
After sweep: rerun best params for multiple seeds; optionally across extra mask roots.
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
DEFAULT_MASK_ROOT = "/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
DEFAULT_EXTRA_MASK_ROOTS = ",".join([
    "/home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
    "/home/ryreu/guided_cnn/code/SwitchCLIP/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap",
])

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


class NICOWithMasks(Dataset):
    def __init__(self, txtdir, dataset_name, domains, phase, mask_root,
                 image_root=None, image_transform=None, mask_transform=None):
        self.mask_root = mask_root
        self.image_root = image_root
        self.image_transform = image_transform
        self.mask_transform = mask_transform

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

        mask_path = self._get_mask_path(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        if self.image_transform is not None:
            if "RandomResizedCrop" in str(self.image_transform):
                image, mask = self._apply_train_transforms(image, mask)
            else:
                image, mask = self._apply_eval_transforms(image, mask)

        return image, label, mask, img_path

    def _apply_train_transforms(self, image, mask):
        import torchvision.transforms.functional as TF
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
        )
        image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.NEAREST)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        mask = torch.clamp(mask * 8.0, 0.0, 1.0)
        return image, mask

    def _apply_eval_transforms(self, image, mask):
        import torchvision.transforms.functional as TF
        image = TF.resize(image, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (224, 224), interpolation=TF.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        return image, mask

    def _resolve_path(self, path):
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

    def _get_mask_path(self, img_path):
        norm_path = os.path.normpath(img_path)
        parts = norm_path.split(os.sep)
        if len(parts) >= 3:
            domain = parts[-3]
            class_name = parts[-2]
            filename = parts[-1]
            basename = os.path.splitext(filename)[0]
            mask_filename = f"{domain}_{class_name}_{basename}.png"
            return os.path.join(self.mask_root, mask_filename)
        basename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        return os.path.join(self.mask_root, basename)


def build_train_val_datasets(args, sources, data_transforms, generator, mask_root_override=None, split_indices=None):
    mask_root = mask_root_override or args.mask_root
    val_paths = [os.path.join(args.txtdir, args.dataset, f"{d}_val.txt") for d in sources]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_ds = NICOWithMasks(args.txtdir, args.dataset, sources, "train", mask_root, args.image_root, data_transforms["train"], None)
        val_ds = NICOWithMasks(args.txtdir, args.dataset, sources, "val", mask_root, args.image_root, data_transforms["eval"], None)
        return train_ds, val_ds, has_val, split_indices

    full_train = NICOWithMasks(args.txtdir, args.dataset, sources, "train", mask_root, args.image_root, None, None)
    n_total = len(full_train)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val

    if split_indices is None:
        train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
        split_indices = (train_idx.indices, val_idx.indices)
    train_idx_list, val_idx_list = split_indices

    train_ds = NICOWithMasks(args.txtdir, args.dataset, sources, "train", mask_root, args.image_root, data_transforms["train"], None)
    val_ds = NICOWithMasks(args.txtdir, args.dataset, sources, "train", mask_root, args.image_root, data_transforms["eval"], None)
    return Subset(train_ds, train_idx_list), Subset(val_ds, val_idx_list), has_val, split_indices


def make_cam_model(num_classes):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base = models.resnet50(weights=weights)
    except AttributeError:
        base = models.resnet50(pretrained=True)
    num_features = base.fc.in_features
    base.fc = nn.Linear(num_features, num_classes)

    class CAMWrapResNet(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.features = None
            self.base.layer4.register_forward_hook(self._hook_fn)

        def _hook_fn(self, module, inp, out):
            self.features = out

        def forward(self, x):
            out = self.base(x)
            return out, self.features

    return CAMWrapResNet(base)


def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_ce):
    ce_loss = nn.functional.cross_entropy(outputs, labels)
    B, C, Hf, Wf = cams.shape
    cam_avg = cams.mean(dim=1)
    cam_flat = cam_avg.view(B, -1)
    gt_flat = gt_masks.view(B, -1)
    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)
    if only_ce:
        return ce_loss, attn_loss
    return ce_loss + kl_lambda * attn_loss, attn_loss


def _get_param_groups(model, base_lr, classifier_lr):
    base_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".fc" in name or "classifier" in name:
            classifier_params.append(param)
        else:
            base_params.append(param)
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})
    else:
        param_groups.append({"params": base_params, "lr": base_lr})
    return param_groups


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        outputs, _ = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        images = batch[0].to(DEVICE)
        labels = batch[1]
        outputs, _ = model(images)
        logits_all.append(outputs.detach().cpu())
        labels_all.append(labels.cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


@torch.no_grad()
def evaluate_ensemble_accuracy(pre_model, post_model, loader, ensemble_weight):
    pre_model.eval()
    post_model.eval()
    correct = 0
    total = 0
    w = float(ensemble_weight)
    for batch in loader:
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        logits_pre, _ = pre_model(images)
        logits_post, _ = post_model(images)
        logits = w * logits_pre + (1.0 - w) * logits_post
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / max(total, 1)


def select_ensemble_weight(pre_model, post_model, val_loader, weight_grid):
    pre_logits, labels = collect_logits(pre_model, val_loader)
    post_logits, labels_post = collect_logits(post_model, val_loader)
    if not torch.equal(labels, labels_post):
        raise RuntimeError("Validation label order mismatch while selecting ensemble weight.")

    best_w = float(weight_grid[0])
    best_acc = -1.0
    labels = labels.long()
    for w in weight_grid:
        logits = float(w) * pre_logits + (1.0 - float(w)) * post_logits
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_w = float(w)
    return best_w, float(best_acc)


@torch.no_grad()
def evaluate_test_ensemble(pre_model, post_model, test_loaders, target_domains, ensemble_weight):
    pre_model.eval()
    post_model.eval()
    results = {}
    all_correct = 0
    all_total = 0
    w = float(ensemble_weight)
    for test_loader, domain_name in zip(test_loaders, target_domains):
        total, correct = 0, 0
        for batch in test_loader:
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            logits_pre, _ = pre_model(images)
            logits_post, _ = post_model(images)
            logits = w * logits_pre + (1.0 - w) * logits_post
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        acc = 100.0 * correct / max(total, 1)
        results[domain_name] = acc
        all_correct += correct
        all_total += total
    results["overall"] = 100.0 * all_correct / max(all_total, 1)
    return results


def _model_from_state(num_classes, state_dict):
    m = make_cam_model(num_classes).to(DEVICE)
    m.load_state_dict(state_dict)
    return m


def train_two_phase_swad_guided(model, dataloaders, args, swad_args):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    param_groups = _get_param_groups(model, args.base_lr, args.classifier_lr)
    opt = optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    pre_swad = LossValley(swad_args["n_converge"], swad_args["n_tolerance"], swad_args["tolerance_ratio"])
    post_swad = LossValley(swad_args["n_converge"], swad_args["n_tolerance"], swad_args["tolerance_ratio"])
    pre_segment = swa_utils.AveragedModel(model, rm_optimizer=True)
    post_segment = None

    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0
    kl_lambda_real = args.kl_lambda_start
    pre_phase_end_state = copy.deepcopy(model.state_dict())
    post_phase_last_state = None

    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            pre_phase_end_state = copy.deepcopy(model.state_dict())
            base_lr_post = args.base_lr * args.lr2_mult
            classifier_lr_post = args.classifier_lr * args.lr2_mult
            param_groups = _get_param_groups(model, base_lr_post, classifier_lr_post)
            opt = optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            kl_lambda_real = args.kl_lambda_start
            post_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        # ---- train ----
        model.train()
        for batch in train_loader:
            inputs, labels, gt_masks, _ = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)

            opt.zero_grad()
            outputs, feats = model(inputs)
            weights = model.base.fc.weight[labels]
            cams = torch.einsum("bc,bchw->bhw", weights, feats)
            cams = torch.relu(cams)
            gt_small = nn.functional.interpolate(gt_masks, size=cams.shape[1:], mode="nearest").squeeze(1)

            if epoch < args.attention_epoch:
                loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=0.0, only_ce=True)
            else:
                loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=kl_lambda_real, only_ce=False)

            loss.backward()
            opt.step()

            if epoch < args.attention_epoch:
                pre_segment.update_parameters(model, step=global_step)
            else:
                if post_segment is None:
                    post_segment = swa_utils.AveragedModel(model, rm_optimizer=True)
                post_segment.update_parameters(model, step=global_step)
            global_step += 1

        sch.step()

        # ---- validate ----
        model.eval()
        running_loss = 0.0
        running_correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels, gt_masks, _ = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                gt_masks = gt_masks.to(DEVICE)

                outputs, feats = model(inputs)
                preds = outputs.argmax(dim=1)
                weights = model.base.fc.weight[labels]
                cams = torch.einsum("bc,bchw->bhw", weights, feats)
                cams = torch.relu(cams)
                gt_small = nn.functional.interpolate(gt_masks, size=cams.shape[1:], mode="nearest").squeeze(1)

                if epoch < args.attention_epoch:
                    loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=0.0, only_ce=True)
                else:
                    loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=kl_lambda_real, only_ce=False)

                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                total += inputs.size(0)

        val_loss = running_loss / max(total, 1)
        val_acc = running_correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        print(f"[Epoch {epoch}] val_acc={val_acc:.6f} val_loss={val_loss:.6f}", flush=True)
        if epoch < args.attention_epoch and epoch >= args.swad_start_epoch:
            pre_swad.update_and_evaluate(pre_segment, val_acc, val_loss)
            if pre_swad.dead_valley:
                print("[SWAD PRE] Valley is dead; stop collecting pre-attention SWAD updates.", flush=True)
            else:
                pre_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

        post_swad_start = max(args.swad_start_epoch, args.attention_epoch)
        if epoch >= post_swad_start:
            if post_segment is None:
                post_segment = swa_utils.AveragedModel(model, rm_optimizer=True)
            post_swad.update_and_evaluate(post_segment, val_acc, val_loss)
            if post_swad.dead_valley:
                print("SWAD post valley is dead -> early stop!", flush=True)
                post_phase_last_state = copy.deepcopy(model.state_dict())
                break
            post_segment = swa_utils.AveragedModel(model, rm_optimizer=True)
            post_phase_last_state = copy.deepcopy(model.state_dict())

    if post_phase_last_state is None:
        post_phase_last_state = copy.deepcopy(model.state_dict())

    if pre_swad.is_converged:
        pre_model = pre_swad.get_final_model()
    else:
        pre_model = _model_from_state(args.num_classes, pre_phase_end_state)

    if post_swad.is_converged:
        post_model = post_swad.get_final_model()
    else:
        post_model = _model_from_state(args.num_classes, post_phase_last_state)

    print(
        f"[SWAD 2PHASE] pre_converged={pre_swad.is_converged} post_converged={post_swad.is_converged} "
        f"(best plain val_acc={best_val_acc:.6f} @ {best_epoch})",
        flush=True,
    )
    return pre_model, post_model, float(best_val_acc)


def suggest_uniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high)


def suggest_loguniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high, log=True)


def suggest_int(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_int(name, low, high)


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def parse_float_grid(csv_values: str):
    vals = [float(v) for v in csv_values.split(",") if v.strip()]
    if not vals:
        vals = [0.5]
    vals = sorted(set(vals))
    clipped = [min(1.0, max(0.0, v)) for v in vals]
    return clipped


def main():
    p = argparse.ArgumentParser(description="Optuna Guided two-phase SWAD ensemble (SGD) on NICO++")
    p.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--mask_root", type=str, default=DEFAULT_MASK_ROOT)
    p.add_argument("--extra_mask_roots", type=str, default=DEFAULT_EXTRA_MASK_ROOTS)
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED)

    # fixed training config
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    p.add_argument("--base_lr_low", type=float, default=1e-5)
    p.add_argument("--base_lr_high", type=float, default=5e-2)
    p.add_argument("--classifier_lr_low", type=float, default=1e-5)
    p.add_argument("--classifier_lr_high", type=float, default=5e-2)
    p.add_argument("--lr2_mult_low", type=float, default=1e-3)
    p.add_argument("--lr2_mult_high", type=float, default=1.0)
    p.add_argument("--attention_epoch", type=int, default=15)
    p.add_argument("--kl_lambda_low", type=float, default=0.1)
    p.add_argument("--kl_lambda_high", type=float, default=50.0)
    p.add_argument("--kl_increment", type=float, default=0.0)

    # SWAD sweep params
    p.add_argument("--n_converge_min", type=int, default=1)
    p.add_argument("--n_converge_max", type=int, default=5)
    p.add_argument("--n_tolerance", type=int, default=6)
    p.add_argument("--tolerance_ratio", type=float, default=0.3)
    p.add_argument("--swad_start_epoch", type=int, default=0)
    p.add_argument("--ensemble_weight_grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    # optuna settings
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--optuna_seed", type=int, default=SEED)
    p.add_argument("--load_if_exists", action="store_true")

    # rerun best
    p.add_argument("--rerun_best", type=int, default=1)
    p.add_argument("--rerun_seed_start", type=int, default=59)
    p.add_argument("--rerun_num_seeds", type=int, default=5)
    p.add_argument("--rerun_seeds", type=str, default="")

    args = p.parse_args()
    ensemble_weight_grid = parse_float_grid(args.ensemble_weight_grid)

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    run_name = f"guided_swad_ensemble_target_{'-'.join(targets)}"
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

    train_dataset, val_dataset, has_val, split_indices = build_train_val_datasets(
        args, sources, data_transforms, base_g
    )
    if not has_val:
        print(f"Val split: {int(VAL_SPLIT_RATIO*100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOWithMasks(args.txtdir, args.dataset, [domain], "test", args.mask_root, args.image_root, data_transforms["eval"], None)
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

    def log_trial(trial, ensemble_val_acc, ensemble_weight, pre_val_acc, post_val_acc, test_results):
        row = {
            "trial": trial.number,
            "ensemble_val_acc": ensemble_val_acc,
            "ensemble_weight": ensemble_weight,
            "pre_val_acc": pre_val_acc,
            "post_val_acc": post_val_acc,
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
        lr2_mult = suggest_loguniform(trial, "lr2_mult", args.lr2_mult_low, args.lr2_mult_high)
        kl_lambda_start = suggest_loguniform(trial, "kl_lambda_start", args.kl_lambda_low, args.kl_lambda_high)
        n_converge = suggest_int(trial, "n_converge", args.n_converge_min, args.n_converge_max)
        kl_increment = args.kl_increment

        print(
            f"[TRIAL {trial.number}] start params={{'base_lr': {base_lr}, 'classifier_lr': {classifier_lr}, "
            f"'lr2_mult': {lr2_mult}, 'attention_epoch': {args.attention_epoch}, "
            f"'kl_lambda_start': {kl_lambda_start}, 'kl_increment': {kl_increment}, "
            f"'n_converge': {n_converge}, 'n_tolerance': {args.n_tolerance}, 'tolerance_ratio': {args.tolerance_ratio}}}",
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

        model = make_cam_model(args.num_classes).to(DEVICE)
        trial_args = argparse.Namespace(
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            lr2_mult=lr2_mult,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            attention_epoch=args.attention_epoch,
            kl_lambda_start=kl_lambda_start,
            kl_increment=kl_increment,
            swad_start_epoch=args.swad_start_epoch,
            num_epochs=args.num_epochs,
            num_classes=args.num_classes,
        )
        swad_args = dict(
            n_converge=n_converge,
            n_tolerance=args.n_tolerance,
            tolerance_ratio=args.tolerance_ratio,
        )

        try:
            pre_model, post_model, best_plain_val_acc = train_two_phase_swad_guided(model, dataloaders, trial_args, swad_args)
            pre_val_acc = evaluate_accuracy(pre_model, dataloaders["val"])
            post_val_acc = evaluate_accuracy(post_model, dataloaders["val"])
            ensemble_weight, ensemble_val_acc = select_ensemble_weight(
                pre_model, post_model, dataloaders["val"], ensemble_weight_grid
            )
            test_results = evaluate_test_ensemble(pre_model, post_model, test_loaders, targets, ensemble_weight)
            trial.set_user_attr("test_results", test_results)
            trial.set_user_attr("ensemble_weight", float(ensemble_weight))
            trial.set_user_attr("pre_val_acc", float(pre_val_acc))
            trial.set_user_attr("post_val_acc", float(post_val_acc))
            trial.set_user_attr("best_plain_val_acc", float(best_plain_val_acc))
            log_trial(trial, ensemble_val_acc, ensemble_weight, pre_val_acc, post_val_acc, test_results)
            print(
                f"[TRIAL {trial.number}] done ensemble_val_acc={ensemble_val_acc:.6f} "
                f"(pre={pre_val_acc:.6f}, post={post_val_acc:.6f}, w={ensemble_weight:.2f}, "
                f"best_plain={best_plain_val_acc:.6f}) test={test_results}",
                flush=True,
            )
        finally:
            if "pre_model" in locals():
                del pre_model
            if "post_model" in locals():
                del post_model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ensemble_val_acc

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

    print("Best ensemble val accuracy:", study.best_value)
    print("Best params:", study.best_params)
    if best_trial.user_attrs.get("ensemble_weight") is not None:
        print("Best trial ensemble weight:", best_trial.user_attrs.get("ensemble_weight"))
    if best_trial.user_attrs.get("pre_val_acc") is not None and best_trial.user_attrs.get("post_val_acc") is not None:
        print(
            "Best trial pre/post val_acc:",
            best_trial.user_attrs.get("pre_val_acc"),
            best_trial.user_attrs.get("post_val_acc"),
        )
    print("Best trial test results:", best["best_test_results"])
    print("Saved:", best_path)

    if int(args.rerun_best) == 1:
        seeds = parse_seeds(args.rerun_seed_start, args.rerun_num_seeds, args.rerun_seeds)
        best_base_lr = float(study.best_params["base_lr"])
        best_classifier_lr = float(study.best_params["classifier_lr"])
        best_kl_lambda_start = float(study.best_params["kl_lambda_start"])
        best_lr2_mult = float(study.best_params.get("lr2_mult", args.lr2_mult_low))
        best_kl_increment = args.kl_increment
        best_n_converge = int(study.best_params.get("n_converge", args.n_converge_min))
        best_ensemble_weight = float(best_trial.user_attrs.get("ensemble_weight", 0.5))

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_masks_path = os.path.join(output_dir, "best_rerun_seeds_masks.csv")
        rerun_header = [
            "seed", "base_lr", "classifier_lr", "lr2_mult",
            "attention_epoch", "kl_lambda_start", "kl_increment",
            "n_converge", "n_tolerance", "tolerance_ratio", "swad_start_epoch",
            "ensemble_weight", "pre_val_acc", "post_val_acc", "ensemble_val_acc",
            "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)
        rerun_masks_header = ["mask_root"] + rerun_header
        write_header_masks = not os.path.exists(rerun_masks_path)
        if write_header_masks:
            with open(rerun_masks_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_masks_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            f"Best params resolved: base_lr={best_base_lr} classifier_lr={best_classifier_lr} "
            f"lr2_mult={best_lr2_mult} attention_epoch={args.attention_epoch} "
            f"kl_lambda_start={best_kl_lambda_start} kl_increment={best_kl_increment} "
            f"n_converge={best_n_converge} n_tolerance={args.n_tolerance} "
            f"tolerance_ratio={args.tolerance_ratio} swad_start_epoch={args.swad_start_epoch} "
            f"ensemble_weight={best_ensemble_weight}",
            flush=True,
        )

        mask_roots = [args.mask_root]
        if args.extra_mask_roots:
            mask_roots.extend([m for m in args.extra_mask_roots.split(",") if m.strip()])

        for mask_root in mask_roots:
            train_ds_m, val_ds_m, _, split_indices_m = build_train_val_datasets(
                args, sources, data_transforms, base_g, mask_root_override=mask_root, split_indices=split_indices
            )
            test_datasets_m = [
                NICOWithMasks(args.txtdir, args.dataset, [domain], "test", mask_root, args.image_root, data_transforms["eval"], None)
                for domain in targets
            ]
            test_loaders_m = [
                DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, worker_init_fn=seed_worker, generator=base_g)
                for ds in test_datasets_m
            ]

            print(f"\n=== Rerun masks: {mask_root} ===", flush=True)

            for seed in seeds:
                seed_everything(seed)
                g = torch.Generator()
                g.manual_seed(seed)

                dataloaders_m = {
                    "train": DataLoader(train_ds_m, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
                    "val": DataLoader(val_ds_m, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
                }

                model = make_cam_model(args.num_classes).to(DEVICE)
                run_args = argparse.Namespace(
                    base_lr=best_base_lr,
                    classifier_lr=best_classifier_lr,
                    lr2_mult=best_lr2_mult,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    attention_epoch=args.attention_epoch,
                    kl_lambda_start=best_kl_lambda_start,
                    kl_increment=best_kl_increment,
                    swad_start_epoch=args.swad_start_epoch,
                    num_epochs=args.num_epochs,
                    num_classes=args.num_classes,
                )
                swad_args = dict(
                    n_converge=best_n_converge,
                    n_tolerance=args.n_tolerance,
                    tolerance_ratio=args.tolerance_ratio,
                )

                start = time.time()
                pre_model, post_model, best_plain_val_acc = train_two_phase_swad_guided(
                    model, dataloaders_m, run_args, swad_args
                )
                pre_val_acc = evaluate_accuracy(pre_model, dataloaders_m["val"])
                post_val_acc = evaluate_accuracy(post_model, dataloaders_m["val"])
                ensemble_val_acc = evaluate_ensemble_accuracy(
                    pre_model, post_model, dataloaders_m["val"], best_ensemble_weight
                )
                test_results = evaluate_test_ensemble(
                    pre_model, post_model, test_loaders_m, targets, best_ensemble_weight
                )
                minutes = (time.time() - start) / 60.0

                print(
                    f"[RERUN mask={os.path.basename(mask_root)} seed={seed}] "
                    f"ensemble_val_acc={ensemble_val_acc:.6f} (pre={pre_val_acc:.6f}, post={post_val_acc:.6f}, "
                    f"w={best_ensemble_weight:.2f}, best_plain={best_plain_val_acc:.6f}) "
                    f"test={test_results} time={minutes:.1f}m",
                    flush=True,
                )

                row = [
                    seed,
                    best_base_lr,
                    best_classifier_lr,
                    best_lr2_mult,
                    args.attention_epoch,
                    best_kl_lambda_start,
                    best_kl_increment,
                    best_n_converge,
                    args.n_tolerance,
                    args.tolerance_ratio,
                    args.swad_start_epoch,
                    best_ensemble_weight,
                    pre_val_acc,
                    post_val_acc,
                    ensemble_val_acc,
                    json.dumps(test_results),
                    minutes,
                ]

                if mask_root == args.mask_root:
                    with open(rerun_path, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                with open(rerun_masks_path, "a", newline="") as f:
                    csv.writer(f).writerow([mask_root] + row)

                del pre_model
                del post_model
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("Saved reruns:", rerun_path, flush=True)
        print("Saved reruns (all masks):", rerun_masks_path, flush=True)


if __name__ == "__main__":
    main()
