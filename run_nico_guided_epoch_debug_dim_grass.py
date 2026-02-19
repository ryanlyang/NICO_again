#!/usr/bin/env python3
"""
Single-run guided debug script for held-out dim+grass.

Uses fixed hyperparameters and prints per-epoch:
- validation accuracy
- reverse-KL (input gradients vs masks)
- optim_value
- test accuracy on dim/grass/overall
"""

import argparse
import copy
import math
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from run_guided_optuna_sgd import (
    ALL_DOMAINS,
    BATCH_SIZE,
    DEVICE,
    FIXED_NUM_EPOCHS,
    MOMENTUM,
    WEIGHT_DECAY,
    DEFAULT_TXTLIST_DIR,
    DEFAULT_MASK_ROOT,
    DEFAULT_IMAGE_ROOT,
    NICOWithMasks,
    _get_param_groups,
    build_train_val_datasets,
    compute_attn_losses,
    compute_loss,
    evaluate_test,
    input_grad_saliency,
    make_cam_model,
    seed_everything,
    seed_worker,
)


def train_debug(model, dataloaders, dataset_sizes, test_loaders, target_domains, args):
    best_wts_val = copy.deepcopy(model.state_dict())
    best_wts_optim = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_optim_value = -1.0

    param_groups = _get_param_groups(model, args.base_lr, args.classifier_lr)
    opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    kl_lambda_real = args.kl_lambda_start

    print("=== Debug run start ===", flush=True)
    print(
        f"targets={target_domains} base_lr={args.base_lr} classifier_lr={args.classifier_lr} "
        f"lr2_mult={args.lr2_mult} attention_epoch={args.attention_epoch} "
        f"kl_lambda_start={args.kl_lambda_start} kl_increment={args.kl_increment} beta={args.beta}",
        flush=True,
    )

    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            base_lr_post = args.base_lr * args.lr2_mult
            classifier_lr_post = args.classifier_lr * args.lr2_mult
            param_groups = _get_param_groups(model, base_lr_post, classifier_lr_post)
            opt = optim.SGD(param_groups, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            best_wts_val = copy.deepcopy(model.state_dict())
            best_wts_optim = copy.deepcopy(model.state_dict())
            best_val_acc = -1.0
            best_optim_value = -1.0
            print(
                f"[Epoch {epoch}] attention switch -> base_lr_post={base_lr_post:.8g} "
                f"classifier_lr_post={classifier_lr_post:.8g}",
                flush=True,
            )

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        epoch_val_acc = 0.0
        epoch_rev_kl = 0.0
        epoch_optim_value = 0.0

        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            running_corrects = 0
            running_attn_rev = 0.0
            total = 0

            for batch in dataloaders[phase]:
                inputs, labels, gt_masks, _ = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                gt_masks = gt_masks.to(DEVICE)

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, feats = model(inputs)
                    preds = outputs.argmax(dim=1)

                    gt_small = torch.nn.functional.interpolate(
                        gt_masks, size=feats.shape[-2:], mode="nearest"
                    ).squeeze(1)

                    if epoch < args.attention_epoch:
                        loss, _ = compute_loss(
                            outputs, labels, feats, gt_small, kl_lambda=333, only_ce=True
                        )
                    else:
                        loss, _ = compute_loss(
                            outputs, labels, feats, gt_small, kl_lambda=kl_lambda_real, only_ce=False
                        )

                    if is_train:
                        loss.backward()
                        opt.step()

                running_corrects += (preds == labels).sum().item()
                total += inputs.size(0)

                if not is_train:
                    with torch.enable_grad():
                        sal = input_grad_saliency(model, inputs, labels)
                    _, rev_kl = compute_attn_losses(sal, gt_masks.squeeze(1))
                    running_attn_rev += rev_kl.item() * inputs.size(0)

            if is_train:
                sch.step()

            if phase == "val":
                epoch_val_acc = running_corrects / max(total, 1)
                epoch_rev_kl = running_attn_rev / max(total, 1)
                epoch_optim_value = epoch_val_acc * math.exp(-float(args.beta) * epoch_rev_kl)

                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_wts_val = copy.deepcopy(model.state_dict())

                if epoch >= args.attention_epoch and epoch_optim_value > best_optim_value:
                    best_optim_value = epoch_optim_value
                    best_wts_optim = copy.deepcopy(model.state_dict())

        test_results = evaluate_test(model, test_loaders, target_domains)
        print(
            f"[Epoch {epoch}] val_acc={epoch_val_acc:.6f} rev_kl={epoch_rev_kl:.6f} "
            f"optim_value={epoch_optim_value:.6f} test={test_results}",
            flush=True,
        )

    if best_optim_value < 0.0:
        best_optim_value = float(best_val_acc)
        best_wts_optim = copy.deepcopy(best_wts_val)

    model.load_state_dict(best_wts_val)
    best_test_val = evaluate_test(model, test_loaders, target_domains)

    model.load_state_dict(best_wts_optim)
    best_test_optim = evaluate_test(model, test_loaders, target_domains)

    print("\n=== Final summary ===", flush=True)
    print(f"best_val_acc={best_val_acc:.6f} best_test_val={best_test_val}", flush=True)
    print(f"best_optim_value={best_optim_value:.6f} best_test_optim={best_test_optim}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Guided debug run for dim+grass held-out with per-epoch test metrics."
    )
    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    parser.add_argument("--dataset", type=str, default="NICO")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--mask_root", type=str, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--target", nargs="+", default=["dim", "grass"])
    parser.add_argument("--num_classes", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--num_epochs", type=int, default=FIXED_NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    # Fixed hyperparameters from requested Trial 26.
    parser.add_argument("--base_lr", type=float, default=4.388420270086194e-04)
    parser.add_argument("--classifier_lr", type=float, default=1.8305056881934558e-03)
    parser.add_argument("--lr2_mult", type=float, default=1.147852203655228e-01)
    parser.add_argument("--attention_epoch", type=int, default=11)
    parser.add_argument("--kl_lambda_start", type=float, default=1.0041163007349849e-01)
    parser.add_argument("--kl_increment", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

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

    train_dataset, val_dataset, has_val, _ = build_train_val_datasets(args, sources, data_transforms, g)
    if not has_val:
        print("Val split: 16% split from train (no *_val.txt found)", flush=True)

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
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    print(
        f"dataset_sizes train={dataset_sizes['train']} val={dataset_sizes['val']} targets={targets}",
        flush=True,
    )

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
        for domain in targets
    ]
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
    train_debug(model, dataloaders, dataset_sizes, test_loaders, targets, args)


if __name__ == "__main__":
    main()
