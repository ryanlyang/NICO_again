#!/usr/bin/env python3
"""
Guided SWAD (LossValley) on NICO++ with SGD and fixed epochs.

This mirrors DomainBed's SWAD logic but runs the guided training loop:
  - ResNet50 CAM + mask guidance (CE + KL after attention_epoch)
  - SGD with base/classifier param groups
  - SWAD LossValley on validation loss (epoch-level segments)

Val loss drives the SWAD valley; final test is on the SWAD-averaged model.
"""

import argparse
import copy
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

from domainbed.lib.swad import LossValley
from domainbed.lib import swa_utils


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
            outputs, _ = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        acc = 100.0 * correct / max(total, 1)
        results[domain_name] = acc
        all_correct += correct
        all_total += total
    results["overall"] = 100.0 * all_correct / max(all_total, 1)
    return results


def main():
    p = argparse.ArgumentParser(description="Guided SWAD (SGD) on NICO++")
    p.add_argument("--txtdir", type=str, default="/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist")
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default="/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG")
    p.add_argument("--mask_root", type=str, required=True)
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--classifier_lr", type=float, default=1e-4)
    p.add_argument("--lr2_mult", type=float, default=1.0)
    p.add_argument("--attention_epoch", type=int, default=15)
    p.add_argument("--kl_lambda_start", type=float, default=10.0)
    p.add_argument("--kl_increment", type=float, default=1.0)

    p.add_argument("--n_converge", type=int, default=3)
    p.add_argument("--n_tolerance", type=int, default=6)
    p.add_argument("--tolerance_ratio", type=float, default=0.3)

    args = p.parse_args()

    targets = [t.lower() for t in args.target]
    unknown = [t for t in targets if t not in ALL_DOMAINS]
    if unknown:
        raise SystemExit(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise SystemExit("Target domains cover all domains; no source domains remain to train on.")

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
    test_datasets = [
        NICOWithMasks(args.txtdir, args.dataset, [d], "test", args.mask_root, args.image_root, data_transforms["eval"], None)
        for d in targets
    ]
    test_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        for ds in test_datasets
    ]

    model = make_cam_model(num_classes=60).to(DEVICE)
    param_groups = _get_param_groups(model, args.base_lr, args.classifier_lr)
    opt = optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    swad = LossValley(args.n_converge, args.n_tolerance, args.tolerance_ratio)
    swad_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

    best_val_acc = -1.0
    best_epoch = -1

    kl_lambda_real = args.kl_lambda_start
    global_step = 0

    start_time = time.time()
    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            base_lr_post = args.base_lr * args.lr2_mult
            classifier_lr_post = args.classifier_lr * args.lr2_mult
            param_groups = _get_param_groups(model, base_lr_post, classifier_lr_post)
            opt = optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        # --- train ---
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

            swad_segment.update_parameters(model, step=global_step)
            global_step += 1

        sch.step()

        # --- validate ---
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

        # SWAD update
        swad.update_and_evaluate(swad_segment, val_acc, val_loss)
        if swad.dead_valley:
            print("SWAD valley is dead -> early stop!", flush=True)
            break
        swad_segment = swa_utils.AveragedModel(model, rm_optimizer=True)

    # --- final model ---
    final_model = swad.get_final_model()
    test_results = evaluate_test(final_model, test_loaders, targets)
    minutes = (time.time() - start_time) / 60.0

    print("\n=== Guided SWAD Result ===", flush=True)
    print(f"Best val_acc (plain): {best_val_acc:.6f} @ epoch {best_epoch}", flush=True)
    print(f"SWAD test: {test_results}", flush=True)
    print(f"Time: {minutes:.1f} minutes", flush=True)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"swad_guided_result_target_{'-'.join(targets)}.txt")
        with open(out_path, "w") as f:
            f.write(f"best_val_acc={best_val_acc:.6f} epoch={best_epoch}\n")
            f.write(f"swad_test={test_results}\n")
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
