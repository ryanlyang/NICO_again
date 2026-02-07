#!/usr/bin/env python3
"""
Single-run guided SGD training on NICO++ with per-epoch logging:
- prints val_acc, reverse-KL (input-grad saliency vs GT mask), optim_value, and test accuracy every epoch
- uses the same NICO txtlist plumbing and mask mapping as the guided optuna scripts

This is intended for "oracle curve" inspection: how test evolves relative to optim_value.
"""

import os
import time
import copy
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
    """
    NICO++ dataset loader with GT masks.
    Returns: (image, label, mask, path)
    """
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


def build_train_val_datasets(txtdir, dataset, sources, mask_root, image_root, data_transforms, generator, val_split_ratio=0.16):
    val_paths = [os.path.join(txtdir, dataset, f"{d}_val.txt") for d in sources]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_ds = NICOWithMasks(txtdir, dataset, sources, "train", mask_root, image_root, data_transforms["train"], None)
        val_ds = NICOWithMasks(txtdir, dataset, sources, "val", mask_root, image_root, data_transforms["eval"], None)
        return train_ds, val_ds, has_val

    full_train_base = NICOWithMasks(txtdir, dataset, sources, "train", mask_root, image_root, None, None)
    n_total = len(full_train_base)
    n_val = max(1, int(val_split_ratio * n_total))
    n_train = n_total - n_val

    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_idx_list = train_idx.indices
    val_idx_list = val_idx.indices

    train_ds = NICOWithMasks(txtdir, dataset, sources, "train", mask_root, image_root, data_transforms["train"], None)
    val_ds = NICOWithMasks(txtdir, dataset, sources, "train", mask_root, image_root, data_transforms["eval"], None)
    return Subset(train_ds, train_idx_list), Subset(val_ds, val_idx_list), has_val


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

    B, C, _, _ = cams.shape
    cam_avg = cams.mean(dim=1)  # (B,H,W)
    cam_flat = cam_avg.view(B, -1)
    gt_flat = gt_masks.view(B, -1)

    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)

    kl_div = nn.KLDivLoss(reduction="batchmean")
    attn_loss = kl_div(log_p, gt_prob)

    if only_ce:
        return ce_loss, attn_loss
    return ce_loss + kl_lambda * attn_loss, attn_loss


def compute_attn_losses(saliency, gt_masks):
    B, _, _ = saliency.shape
    sal_flat = saliency.view(B, -1)
    gt_flat = gt_masks.view(B, -1)

    log_sal = nn.functional.log_softmax(sal_flat, dim=1)
    sal_prob = nn.functional.softmax(sal_flat, dim=1)

    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    log_gt = torch.log(gt_prob + 1e-8)

    kl_div = nn.KLDivLoss(reduction="batchmean")
    forward_kl = kl_div(log_sal, gt_prob)
    reverse_kl = kl_div(log_gt, sal_prob)
    return forward_kl, reverse_kl


def input_grad_saliency(model, inputs, labels):
    x = inputs.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs, _ = model(x)
    class_scores = outputs[torch.arange(labels.size(0), device=outputs.device), labels]
    class_scores.sum().backward()
    grads = x.grad.detach().abs().sum(dim=1)  # (B,H,W)
    flat = grads.view(grads.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(grads)
    return sal_norm


@torch.no_grad()
def eval_acc(model, loader):
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


def main():
    p = argparse.ArgumentParser(description="Guided SGD run with per-epoch test/optim_value logging (NICO++)")

    p.add_argument("--txtdir", type=str, default="/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist")
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default="/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG")
    p.add_argument("--mask_root", type=str, required=True)
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--num_workers", type=int, default=8)

    # Fixed SGD setup (defaults match your SGD runs)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # Guided hyperparams
    p.add_argument("--pre_lr", type=float, required=True)
    p.add_argument("--post_lr", type=float, required=True)
    p.add_argument("--attention_epoch", type=int, required=True)
    p.add_argument("--kl_lambda_start", type=float, required=True)
    p.add_argument("--kl_increment", type=float, required=True)
    p.add_argument("--beta", type=float, default=0.1)

    args = p.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"epoch_metrics_target_{'-'.join(targets)}.csv")

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

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

    train_ds, val_ds, has_val = build_train_val_datasets(
        args.txtdir, args.dataset, sources, args.mask_root, args.image_root, data_transforms, g, val_split_ratio=0.16
    )
    if not has_val:
        print("Val split: 16% split from train (no *_val.txt found)", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
    )

    test_loaders = {}
    for d in targets:
        test_ds = NICOWithMasks(args.txtdir, args.dataset, [d], "test", args.mask_root, args.image_root, data_transforms["eval"], None)
        test_loaders[d] = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g
        )

    model = make_cam_model(num_classes=60).to(DEVICE)

    opt = optim.SGD(model.parameters(), lr=args.pre_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    best_optim = -1e9
    best_optim_epoch = -1
    best_val_acc = -1.0
    best_val_epoch = -1
    best_test = {d: -1.0 for d in targets}
    best_test_epoch = {d: -1 for d in targets}

    kl_lambda_real = args.kl_lambda_start

    header = [
        "epoch", "lr", "kl_lambda_real",
        "val_acc", "rev_kl", "optim_value",
    ] + [f"test_acc_{d}" for d in targets]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    print(f"Targets={targets} Sources={sources}", flush=True)
    print(
        f"SGD: bs={args.batch_size} epochs={args.num_epochs} momentum={args.momentum} wd={args.weight_decay}",
        flush=True,
    )
    print(
        f"Guided: pre_lr={args.pre_lr} post_lr={args.post_lr} att_epoch={args.attention_epoch} "
        f"kl_start={args.kl_lambda_start} kl_inc={args.kl_increment} beta={args.beta}",
        flush=True,
    )

    t0 = time.time()
    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            opt = optim.SGD(model.parameters(), lr=args.post_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            kl_lambda_real = args.kl_lambda_start

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        # ----------------- train -----------------
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
                # Use class-specific CAMs for the guidance loss (even though CE-only before attention_epoch).
                loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=0.0, only_ce=True)
            else:
                loss, _ = compute_loss(outputs, labels, cams.unsqueeze(1), gt_small, kl_lambda=kl_lambda_real, only_ce=False)

            loss.backward()
            opt.step()

        sch.step()

        # ----------------- val + optim_value -----------------
        model.eval()
        running_correct = 0
        running_rev_kl = 0.0
        total = 0

        for batch in val_loader:
            inputs, labels, gt_masks, _ = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)

            with torch.no_grad():
                outputs, _ = model(inputs)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                total += inputs.size(0)

            with torch.enable_grad():
                sal = input_grad_saliency(model, inputs, labels)  # (B,H,W)
            _, rev_kl = compute_attn_losses(sal, gt_masks.squeeze(1))
            running_rev_kl += float(rev_kl.item()) * inputs.size(0)

        val_acc = running_correct / max(total, 1)
        rev_kl_val = running_rev_kl / max(total, 1)
        optim_value = val_acc * math.exp(-float(args.beta) * rev_kl_val)

        if optim_value > best_optim:
            best_optim = optim_value
            best_optim_epoch = epoch
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch

        # ----------------- test per epoch -----------------
        test_accs = {}
        for d, loader in test_loaders.items():
            acc = eval_acc(model, loader)
            test_accs[d] = acc
            if acc > best_test[d]:
                best_test[d] = acc
                best_test_epoch[d] = epoch

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:02d}] lr={lr_now:.3g} kl={kl_lambda_real:.3g} "
            f"val_acc={val_acc*100:.2f}% rev_kl={rev_kl_val:.4f} optim={optim_value:.6f} "
            + " ".join([f"test_{d}={test_accs[d]*100:.2f}%" for d in targets]),
            flush=True,
        )

        row = [
            epoch, lr_now, kl_lambda_real,
            val_acc, rev_kl_val, optim_value,
        ] + [test_accs[d] for d in targets]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    minutes = (time.time() - t0) / 60.0
    print("\n=== Best epochs ===", flush=True)
    print(f"Best optim_value={best_optim:.6f} @ epoch {best_optim_epoch}", flush=True)
    print(f"Best val_acc={best_val_acc*100:.2f}% @ epoch {best_val_epoch}", flush=True)
    for d in targets:
        print(f"Best test_{d}={best_test[d]*100:.2f}% @ epoch {best_test_epoch[d]}", flush=True)
    print(f"Saved epoch metrics: {csv_path}", flush=True)
    print(f"Total time: {minutes:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
