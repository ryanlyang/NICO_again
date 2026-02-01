#!/usr/bin/env python3
"""
Run guided CNN training/testing for 5 seeds per held-out domain using
best hyperparameters from run_nico_guided_optuna.py sweeps.
"""

import os
import csv
import math
import time
import copy
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

# =============================================================================
# BASE HYPERPARAMETERS (Table 9 protocol / DomainBed)
# =============================================================================
BATCH_SIZE = 32
TOTAL_STEPS = 10000
WEIGHT_DECAY = 0.0
VAL_SPLIT_RATIO = 0.16

DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_MASK_ROOT = "/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"

ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Best hyperparameters per domain (from Optuna sweeps)
DOMAIN_BEST = {
    "autumn": {
        "pre_lr": 0.00010211745586414122,
        "post_lr_ratio": 0.41964423769700504,
        "attention_epoch": 1,
        "kl_lambda_start": 5.084775691586139,
    },
    "rock": {
        "pre_lr": 7.378539137472923e-05,
        "post_lr_ratio": 0.40300482208586985,
        "attention_epoch": 1,
        "kl_lambda_start": 5.799062172239277,
    },
    "dim": {
        "pre_lr": 7.971735701006871e-05,
        "post_lr_ratio": 0.4711957049847522,
        "attention_epoch": 1,
        "kl_lambda_start": 5.554617439600434,
    },
    "grass": {
        "pre_lr": 3.982243424408718e-05,
        "post_lr_ratio": 0.4990670797677378,
        "attention_epoch": 1,
        "kl_lambda_start": 5.005260484234545,
    },
    "outdoor": {
        "pre_lr": 0.00011965280835287209,
        "post_lr_ratio": 0.4312803071612014,
        "attention_epoch": 2,
        "kl_lambda_start": 5.116880404002178,
    },
    "water": {
        "pre_lr": 8.32192057169152e-05,
        "post_lr_ratio": 0.4881046524030672,
        "attention_epoch": 1,
        "kl_lambda_start": 5.0209620480323744,
    },
}


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
# DATASET WITH MASKS
# =============================================================================

class NICOWithMasks(Dataset):
    """NICO++ dataset loader with ground truth masks."""
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
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        mask_path = self._get_mask_path(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)

        if self.image_transform is not None:
            if 'RandomResizedCrop' in str(self.image_transform):
                image, mask = self._apply_train_transforms(image, mask)
            else:
                image, mask = self._apply_eval_transforms(image, mask)

        return image, label, mask, img_path

    def _apply_train_transforms(self, image, mask):
        import torchvision.transforms.functional as TF
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3./4., 4./3.)
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
            mask_path = os.path.join(self.mask_root, mask_filename)
        else:
            basename = os.path.basename(img_path)
            basename = os.path.splitext(basename)[0] + '.png'
            mask_path = os.path.join(self.mask_root, basename)
        return mask_path


# =============================================================================
# MODEL + LOSS
# =============================================================================

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


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def train_one_run(model, dataloaders, dataset_sizes, args):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    opt = optim.Adam(model.parameters(), lr=args.pre_lr, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    kl_lambda_real = args.kl_lambda_start

    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            opt = optim.Adam(model.parameters(), lr=args.post_lr, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            best_wts = copy.deepcopy(model.state_dict())
            best_val_acc = -1.0

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_corrects = 0

            for batch in dataloaders[phase]:
                inputs, labels, gt_masks, _ = batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                gt_masks = gt_masks.to(DEVICE)

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, feats = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    weights = model.base.fc.weight[labels]
                    B, C, _, _ = feats.shape
                    _ = feats.view(B, C, -1).mean(dim=2)

                    cams = torch.einsum('bc,bchw->bhw', weights, feats)
                    cams = torch.relu(cams)

                    flat = cams.view(cams.size(0), -1)
                    mn, _ = flat.min(dim=1, keepdim=True)
                    mx, _ = flat.max(dim=1, keepdim=True)
                    sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

                    gt_small = nn.functional.interpolate(
                        gt_masks, size=sal_norm.shape[1:], mode='nearest'
                    ).squeeze(1)

                    if epoch < args.attention_epoch:
                        loss, _ = compute_loss(
                            outputs, labels,
                            feats,
                            gt_small,
                            kl_lambda=333,
                            only_ce=True
                        )
                    else:
                        loss, _ = compute_loss(
                            outputs, labels,
                            feats,
                            gt_small,
                            kl_lambda=kl_lambda_real,
                            only_ce=False
                        )

                    if is_train:
                        loss.backward()
                        opt.step()

                running_corrects += torch.sum(preds == labels.data)

            if is_train:
                sch.step()

            if phase == 'val':
                epoch_acc = running_corrects.double() / dataset_sizes['val']
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())

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
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs, _ = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
        acc = 100.0 * correct / max(total, 1)
        results[domain_name] = acc
        all_correct += correct
        all_total += total

    overall_acc = 100.0 * all_correct / max(all_total, 1)
    results['overall'] = overall_acc
    return results


# =============================================================================
# MAIN
# =============================================================================

def build_train_val_datasets(args, sources, data_transforms, generator):
    val_paths = [
        os.path.join(args.txtdir, args.dataset, f"{domain}_val.txt")
        for domain in sources
    ]
    has_val = all(os.path.exists(p) for p in val_paths)

    if has_val:
        train_dataset = NICOWithMasks(
            args.txtdir, args.dataset, sources, "train",
            args.mask_root,
            args.image_root,
            data_transforms['train'],
            None
        )
        val_dataset = NICOWithMasks(
            args.txtdir, args.dataset, sources, "val",
            args.mask_root,
            args.image_root,
            data_transforms['eval'],
            None
        )
        return train_dataset, val_dataset, has_val

    full_train_base = NICOWithMasks(
        args.txtdir, args.dataset, sources, "train",
        args.mask_root,
        args.image_root,
        None,
        None
    )
    n_total = len(full_train_base)
    n_val = max(1, int(VAL_SPLIT_RATIO * n_total))
    n_train = n_total - n_val

    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )
    train_idx_list = train_indices.indices
    val_idx_list = val_indices.indices

    train_dataset = NICOWithMasks(
        args.txtdir, args.dataset, sources, "train",
        args.mask_root,
        args.image_root,
        data_transforms['train'],
        None
    )
    val_dataset = NICOWithMasks(
        args.txtdir, args.dataset, sources, "train",
        args.mask_root,
        args.image_root,
        data_transforms['eval'],
        None
    )

    train_subset = Subset(train_dataset, train_idx_list)
    val_subset = Subset(val_dataset, val_idx_list)
    return train_subset, val_subset, has_val


def parse_seeds(args):
    if args.seeds:
        return [int(s) for s in args.seeds.split(',')]
    return list(range(args.seed_start, args.seed_start + args.num_seeds))


def main():
    parser = argparse.ArgumentParser(description='Run 5-seed guided CNN per domain (optuna best).')

    parser.add_argument('--txtdir', type=str, default=DEFAULT_TXTLIST_DIR, help='Path to txt lists')
    parser.add_argument('--dataset', type=str, default='NICO', help='Dataset name')
    parser.add_argument('--image_root', type=str, default=DEFAULT_IMAGE_ROOT, help='Root directory for image files')
    parser.add_argument('--mask_root', type=str, default=DEFAULT_MASK_ROOT, help='Path to mask directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')

    parser.add_argument('--domains', type=str, default=','.join(ALL_DOMAINS), help='Comma-separated target domains')
    parser.add_argument('--seeds', type=str, default='', help='Comma-separated seeds (overrides seed_start/num_seeds)')
    parser.add_argument('--seed_start', type=int, default=59, help='Start seed if --seeds not provided')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seeds if --seeds not provided')

    args = parser.parse_args()

    targets = [d.strip() for d in args.domains.split(',') if d.strip()]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    seeds = parse_seeds(args)
    os.makedirs(args.output_dir, exist_ok=True)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    summary_path = os.path.join(args.output_dir, "optuna_seeded_summary.csv")
    summary_header = [
        "domain", "seed", "pre_lr", "post_lr", "attention_epoch",
        "kl_lambda_start", "kl_increment", "best_val_acc", "test_acc"
    ]
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(summary_header)

    for domain in targets:
        if domain not in DOMAIN_BEST:
            raise ValueError(f"Missing hyperparameters for domain: {domain}")
        sources = [d for d in ALL_DOMAINS if d != domain]

        hp = DOMAIN_BEST[domain]
        pre_lr = hp["pre_lr"]
        post_lr = pre_lr * hp["post_lr_ratio"]
        attention_epoch = hp["attention_epoch"]
        kl_lambda_start = hp["kl_lambda_start"]
        kl_increment = kl_lambda_start / 10.0

        domain_dir = os.path.join(args.output_dir, f"target_{domain}")
        os.makedirs(domain_dir, exist_ok=True)

        print(f"\n=== Domain: {domain} ===")
        print(
            "Hyperparams: pre_lr={:.6g} post_lr={:.6g} att_epoch={} kl_start={:.6g}".format(
                pre_lr, post_lr, attention_epoch, kl_lambda_start
            )
        )

        for seed in seeds:
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            train_dataset, val_dataset, has_val = build_train_val_datasets(
                args, sources, data_transforms, g
            )
            dataset_sizes = {
                'train': len(train_dataset),
                'val': len(val_dataset),
            }
            steps_per_epoch = max(1, math.ceil(dataset_sizes['train'] / BATCH_SIZE))
            num_epochs = max(1, math.ceil(TOTAL_STEPS / steps_per_epoch))
            att_epoch = min(attention_epoch, max(1, num_epochs - 1))

            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
                'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
            }

            test_datasets = [
                NICOWithMasks(
                    args.txtdir, args.dataset, [domain], "test",
                    args.mask_root,
                    args.image_root,
                    data_transforms['eval'],
                    None
                )
            ]
            test_loaders = [
                DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
                for ds in test_datasets
            ]

            model = make_cam_model(60).to(DEVICE)

            run_args = argparse.Namespace(
                pre_lr=pre_lr,
                post_lr=post_lr,
                attention_epoch=att_epoch,
                kl_lambda_start=kl_lambda_start,
                kl_increment=kl_increment,
                num_epochs=num_epochs
            )

            start = time.time()
            best_val_acc = train_one_run(model, dataloaders, dataset_sizes, run_args)
            test_results = evaluate_test(model, test_loaders, [domain])
            elapsed = time.time() - start

            print(
                f"[Domain {domain} | Seed {seed}] "
                f"best_val_acc={best_val_acc:.6f} test={test_results} "
                f"time={elapsed/60:.1f}m"
            )

            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    domain,
                    seed,
                    pre_lr,
                    post_lr,
                    att_epoch,
                    kl_lambda_start,
                    kl_increment,
                    best_val_acc,
                    test_results.get(domain, None),
                ])


if __name__ == '__main__':
    main()
