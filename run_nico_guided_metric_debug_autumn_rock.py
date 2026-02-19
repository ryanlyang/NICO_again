#!/usr/bin/env python3
"""
Guided NICO++ debug runner (autumn+rock held-out) with per-epoch attribution metrics.

For each epoch, prints:
- validation accuracy
- Gradient x Input forward/reverse KL
- Integrated Gradients forward/reverse KL
- CAM forward/reverse KL
- test accuracy on held-out domains

Runs three fixed hyperparameter setups in sequence.
"""

import argparse
import copy
import json
import math
import os
from typing import Dict, List, Optional, Tuple

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
    make_cam_model,
    seed_everything,
    seed_worker,
)


TRIAL_SETUPS = [
    {
        "name": "trial_36",
        "base_lr": 0.012972276913609016,
        "classifier_lr": 6.79761541785652e-05,
        "lr2_mult": 0.018463196348278824,
        "attention_epoch": 11,
        "kl_lambda_start": 0.46479079030852405,
        "kl_increment": 0.0,
    },
    {
        "name": "trial_34",
        "base_lr": 0.00021433680976098852,
        "classifier_lr": 0.0007434605528635105,
        "lr2_mult": 0.045599132525539994,
        "attention_epoch": 8,
        "kl_lambda_start": 0.14669902456222098,
        "kl_increment": 0.0,
    },
    {
        "name": "trial_27",
        "base_lr": 0.0014669918347337053,
        "classifier_lr": 0.004683650727475706,
        "lr2_mult": 0.21931433491573496,
        "attention_epoch": 12,
        "kl_lambda_start": 0.10018509707172872,
        "kl_increment": 0.0,
    },
]


def gradient_x_input_saliency(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor):
    """
    Gradient x Input saliency:
      |x * d(logit_y)/dx| summed over channels, normalized to [0,1].
    """
    x = inputs.detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    outputs, _ = model(x)
    class_scores = outputs[torch.arange(labels.size(0), device=outputs.device), labels]
    grads = torch.autograd.grad(class_scores.sum(), x, retain_graph=False, create_graph=False)[0]
    gx = (x * grads).abs().sum(dim=1)

    flat = gx.view(gx.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    gx_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(gx)
    return gx_norm


def integrated_gradients_saliency(
    model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, steps: int = 16
):
    """
    Integrated gradients saliency:
      IG = (x - x0) * average_{alpha in (0,1]} d(logit_y)/d(x0 + alpha*(x-x0))
    Uses zero baseline.
    Returns |IG| channel-summed and normalized to [0,1].
    """
    baseline = torch.zeros_like(inputs)
    delta = inputs - baseline
    total_grads = torch.zeros_like(inputs)

    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=inputs.device):
        x = (baseline + alpha * delta).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs, _ = model(x)
        class_scores = outputs[torch.arange(labels.size(0), device=outputs.device), labels]
        grads = torch.autograd.grad(class_scores.sum(), x, retain_graph=False, create_graph=False)[0]
        total_grads += grads

    avg_grads = total_grads / float(steps)
    ig = (delta * avg_grads).abs().sum(dim=1)

    flat = ig.view(ig.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    ig_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(ig)
    return ig_norm


def cam_saliency_from_feats(model: nn.Module, feats: torch.Tensor, labels: torch.Tensor):
    """
    Class activation map:
      ReLU(sum_c w_yc * feat_c)
    normalized per sample to [0,1].
    """
    weights = model.base.fc.weight[labels]  # (B, C)
    cams = torch.einsum("bc,bchw->bhw", weights, feats)
    cams = torch.relu(cams)
    flat = cams.view(cams.size(0), -1)
    mn, _ = flat.min(dim=1, keepdim=True)
    mx, _ = flat.max(dim=1, keepdim=True)
    cams_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)
    return cams_norm


def evaluate_test(
    model: nn.Module,
    test_loaders: List[DataLoader],
    target_domains: List[str],
    max_batches: Optional[int] = None,
):
    model.eval()
    results = {}
    all_correct = 0
    all_total = 0

    with torch.no_grad():
        for test_loader, domain_name in zip(test_loaders, target_domains):
            total = 0
            correct = 0
            for batch_idx, batch in enumerate(test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
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


def _metric_spec():
    return {
        "val_acc": ("max", -float("inf")),
        "gxi_rev_kl": ("min", float("inf")),
        "gxi_fwd_kl": ("min", float("inf")),
        "ig_rev_kl": ("min", float("inf")),
        "ig_fwd_kl": ("min", float("inf")),
        "cam_rev_kl": ("min", float("inf")),
        "cam_fwd_kl": ("min", float("inf")),
    }


def _update_best(
    best: Dict[str, Dict],
    values: Dict[str, float],
    epoch: int,
    test_results: Dict[str, float],
):
    for key, val in values.items():
        mode = best[key]["mode"]
        cur = best[key]["value"]
        better = (val > cur) if mode == "max" else (val < cur)
        if better:
            best[key]["value"] = float(val)
            best[key]["epoch"] = int(epoch)
            best[key]["test_results"] = dict(test_results)


def run_single_setup(
    setup: Dict[str, float],
    args,
    train_dataset,
    val_dataset,
    test_datasets,
):
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

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
    param_groups = _get_param_groups(model, setup["base_lr"], setup["classifier_lr"])
    opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
    kl_lambda_real = setup["kl_lambda_start"]

    best = {}
    for k, (mode, init) in _metric_spec().items():
        best[k] = {"mode": mode, "value": init, "epoch": None, "test_results": None}

    print(
        f"\n=== {setup['name']} start params={{'base_lr': {setup['base_lr']}, "
        f"'classifier_lr': {setup['classifier_lr']}, 'lr2_mult': {setup['lr2_mult']}, "
        f"'attention_epoch': {setup['attention_epoch']}, 'kl_lambda_start': {setup['kl_lambda_start']}, "
        f"'kl_increment': {setup['kl_increment']}}} ===",
        flush=True,
    )

    for epoch in range(args.num_epochs):
        if epoch == setup["attention_epoch"]:
            base_lr_post = setup["base_lr"] * setup["lr2_mult"]
            classifier_lr_post = setup["classifier_lr"] * setup["lr2_mult"]
            param_groups = _get_param_groups(model, base_lr_post, classifier_lr_post)
            opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            kl_lambda_real = setup["kl_lambda_start"]

        if epoch > setup["attention_epoch"]:
            kl_lambda_real += setup["kl_increment"]

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

            if epoch < setup["attention_epoch"]:
                loss, _ = compute_loss(outputs, labels, feats, gt_small, kl_lambda=333, only_ce=True)
            else:
                loss, _ = compute_loss(
                    outputs,
                    labels,
                    feats,
                    gt_small,
                    kl_lambda=kl_lambda_real,
                    only_ce=False,
                )

            loss.backward()
            opt.step()

        sch.step()

        # -------------------- val metrics --------------------
        model.eval()
        val_correct = 0
        val_total = 0

        sum_gxi_fwd = 0.0
        sum_gxi_rev = 0.0
        sum_ig_fwd = 0.0
        sum_ig_rev = 0.0
        sum_cam_fwd = 0.0
        sum_cam_rev = 0.0

        for batch_idx, batch in enumerate(dataloaders["val"]):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break

            inputs, labels, gt_masks, _ = batch
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            bsz = inputs.size(0)

            with torch.no_grad():
                outputs, feats = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += bsz

                cam_sal = cam_saliency_from_feats(model, feats, labels)
                cam_gt = nn.functional.interpolate(
                    gt_masks, size=cam_sal.shape[-2:], mode="nearest"
                ).squeeze(1)
                cam_fwd, cam_rev = compute_attn_losses(cam_sal, cam_gt)
                sum_cam_fwd += cam_fwd.item() * bsz
                sum_cam_rev += cam_rev.item() * bsz

            with torch.enable_grad():
                gxi_sal = gradient_x_input_saliency(model, inputs, labels)
                ig_sal = integrated_gradients_saliency(
                    model, inputs, labels, steps=args.ig_steps
                )

            gt_full = gt_masks.squeeze(1)
            gxi_fwd, gxi_rev = compute_attn_losses(gxi_sal, gt_full)
            ig_fwd, ig_rev = compute_attn_losses(ig_sal, gt_full)
            sum_gxi_fwd += gxi_fwd.item() * bsz
            sum_gxi_rev += gxi_rev.item() * bsz
            sum_ig_fwd += ig_fwd.item() * bsz
            sum_ig_rev += ig_rev.item() * bsz

        val_acc = val_correct / max(val_total, 1)
        gxi_fwd_kl = sum_gxi_fwd / max(val_total, 1)
        gxi_rev_kl = sum_gxi_rev / max(val_total, 1)
        ig_fwd_kl = sum_ig_fwd / max(val_total, 1)
        ig_rev_kl = sum_ig_rev / max(val_total, 1)
        cam_fwd_kl = sum_cam_fwd / max(val_total, 1)
        cam_rev_kl = sum_cam_rev / max(val_total, 1)

        test_results = evaluate_test(
            model, test_loaders, args.target, max_batches=args.max_test_batches
        )

        metrics = {
            "val_acc": float(val_acc),
            "gxi_rev_kl": float(gxi_rev_kl),
            "gxi_fwd_kl": float(gxi_fwd_kl),
            "ig_rev_kl": float(ig_rev_kl),
            "ig_fwd_kl": float(ig_fwd_kl),
            "cam_rev_kl": float(cam_rev_kl),
            "cam_fwd_kl": float(cam_fwd_kl),
        }
        _update_best(best, metrics, epoch, test_results)

        print(
            f"[{setup['name']}][Epoch {epoch}] val_acc={val_acc:.6f} "
            f"gxi_rev_kl={gxi_rev_kl:.6f} gxi_fwd_kl={gxi_fwd_kl:.6f} "
            f"ig_rev_kl={ig_rev_kl:.6f} ig_fwd_kl={ig_fwd_kl:.6f} "
            f"cam_rev_kl={cam_rev_kl:.6f} cam_fwd_kl={cam_fwd_kl:.6f} "
            f"test={test_results}",
            flush=True,
        )

    print(f"\n=== {setup['name']} best summary ===", flush=True)
    for key in [
        "val_acc",
        "gxi_rev_kl",
        "gxi_fwd_kl",
        "ig_rev_kl",
        "ig_fwd_kl",
        "cam_rev_kl",
        "cam_fwd_kl",
    ]:
        rec = best[key]
        print(
            f"[{setup['name']}][BEST {key}] value={rec['value']:.6f} "
            f"epoch={rec['epoch']} test={rec['test_results']}",
            flush=True,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{setup['name']}_best_metrics.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[{setup['name']}] saved: {out_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description="Guided NICO++ debug runner with per-epoch attribution KL metrics."
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
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--ig_steps", type=int, default=16)
    p.add_argument("--max_val_batches", type=int, default=None)
    p.add_argument("--max_test_batches", type=int, default=None)
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ryreu/guided_cnn/NICO_runs/output/debug_metric_autumn_rock",
    )
    args = p.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")
    args.target = targets

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

    print(
        f"Running held-out targets={args.target}, sources={sources}, "
        f"train={len(train_dataset)}, val={len(val_dataset)}, "
        f"ig_steps={args.ig_steps}, device={DEVICE}",
        flush=True,
    )

    for setup in TRIAL_SETUPS:
        run_single_setup(setup, args, train_dataset, val_dataset, test_datasets)

    print("\nAll debug runs completed.", flush=True)


if __name__ == "__main__":
    main()
