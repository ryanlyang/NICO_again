#!/usr/bin/env python3
"""
Optuna hyperparameter search for guided CNN on NICO++ (official splits).
- Target domains are provided at runtime; source domains are remaining ones.
- Objective: maximize validation accuracy on official val split.
"""

import os
import time
import copy
import json
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

try:
    import optuna
except Exception as exc:
    raise SystemExit("optuna is required for this script. Install it in your env.") from exc

# =============================================================================
# BASE HYPERPARAMETERS (Table 9 protocol / DomainBed)
# =============================================================================
BATCH_SIZE = 32
TOTAL_STEPS = 10000
WEIGHT_DECAY = 0.0

DEFAULT_ATTENTION_EPOCH = 15  # will be clamped to [1, num_epochs-1]
DEFAULT_KL_LAMBDA_START = 15.0
DEFAULT_KL_INCREMENT = 1.5
VAL_SPLIT_RATIO = 0.16

SEED = 59
DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_MASK_ROOT = "/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
# DATASET WITH MASKS (official NICO++ splits)
# =============================================================================

class NICOWithMasks(Dataset):
    """
    NICO++ dataset loader with ground truth masks
    Returns: (image, label, mask, path)
    """
    def __init__(self, txtdir, dataset_name, domains, phase, mask_root,
                 image_root=None,
                 image_transform=None, mask_transform=None):
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

        # Load mask
        mask_path = self._get_mask_path(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', image.size, 0)

        # Apply synchronized transforms manually
        if self.image_transform is not None:
            if 'RandomResizedCrop' in str(self.image_transform):
                image, mask = self._apply_train_transforms(image, mask)
            else:
                image, mask = self._apply_eval_transforms(image, mask)

        return image, label, mask, img_path

    def _apply_train_transforms(self, image, mask):
        """Apply synchronized training transforms to image and mask"""
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
        """Apply evaluation transforms (no augmentation)"""
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


# =============================================================================
# MODEL: ResNet-50 with CAM
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


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_attn):
    ce_loss = nn.functional.cross_entropy(outputs, labels)

    B, C, Hf, Wf = cams.shape
    cam_avg = cams.mean(dim=1)

    cam_flat = cam_avg.view(B, -1)
    gt_flat = gt_masks.view(B, -1)

    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)

    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)

    if only_attn:
        return attn_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss


# =============================================================================
# TRAINING
# =============================================================================

def train_one_trial(model, dataloaders, dataset_sizes, args, trial=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    opt = optim.Adam(model.parameters(), lr=args.pre_lr, weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    kl_lambda_real = args.kl_lambda_start

    for epoch in range(args.num_epochs):
        if epoch == args.attention_epoch:
            opt = optim.Adam(model.parameters(), lr=args.post_lr, weight_decay=args.weight_decay)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)
            best_wts = copy.deepcopy(model.state_dict())
            best_val_acc = -1.0

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        for phase in ['train', 'val']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_attn_loss = 0.0

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
                    B, C, H, W = feats.shape
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
                        loss, attn_loss = compute_loss(
                            outputs, labels,
                            feats,
                            gt_small,
                            kl_lambda=333,
                            only_attn=True
                        )
                    else:
                        loss, attn_loss = compute_loss(
                            outputs, labels,
                            feats,
                            gt_small,
                            kl_lambda=kl_lambda_real,
                            only_attn=False
                        )

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_attn_loss += attn_loss.item() * inputs.size(0)

            if is_train:
                sch.step()

            if phase == 'val':
                epoch_acc = running_corrects.double() / dataset_sizes['val']
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())

        if trial is not None:
            trial.report(float(best_val_acc), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.load_state_dict(best_wts)
    return float(best_val_acc)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_test(model, test_loaders, target_domains):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    results = {}
    all_correct = 0
    all_total = 0

    for test_loader, domain_name in zip(test_loaders, target_domains):
        total, correct, total_loss = 0, 0, 0.0

        for batch in test_loader:
            if len(batch) == 4:
                images, labels, _, _ = batch
            else:
                images, labels = batch

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        _ = total_loss / max(total, 1)
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

def suggest_loguniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high, log=True)


def suggest_uniform(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_float(name, low, high)


def suggest_int(trial, name, low, high):
    if low == high:
        return low
    return trial.suggest_int(name, low, high)

def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(',') if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def main():
    parser = argparse.ArgumentParser(description='Optuna search for Guided CNN on NICO++ (official splits)')

    parser.add_argument('--txtdir', type=str, default=DEFAULT_TXTLIST_DIR, help='Path to txt lists')
    parser.add_argument('--dataset', type=str, default='NICO', help='Dataset name')
    parser.add_argument('--image_root', type=str, default=DEFAULT_IMAGE_ROOT, help='Root directory for image files')
    parser.add_argument('--target', nargs='+', required=True, help='Target domains')
    parser.add_argument('--num_classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--mask_root', type=str, default=DEFAULT_MASK_ROOT, help='Path to mask directory')

    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage (e.g., sqlite:///study.db)')
    parser.add_argument('--optuna_seed', type=int, default=SEED, help='Seed for Optuna sampler')
    parser.add_argument('--load_if_exists', action='store_true', help='Reuse study if it exists')

    # Search space (set min=max to fix a value)
    parser.add_argument('--pre_lr_low', type=float, default=1e-5)
    parser.add_argument('--pre_lr_high', type=float, default=3e-4)
    parser.add_argument('--post_lr_low', type=float, default=1e-5)
    parser.add_argument('--post_lr_high', type=float, default=3e-4)
    parser.add_argument('--att_epoch_min', type=int, default=1)
    parser.add_argument('--att_epoch_max', type=int, default=25)
    parser.add_argument('--kl_start_low', type=float, default=0.1)
    parser.add_argument('--kl_start_high', type=float, default=30.0)
    parser.add_argument('--kl_inc_low', type=float, default=0.0)
    parser.add_argument('--kl_inc_high', type=float, default=3.0)

    # After sweep: rerun the best hyperparameters for multiple seeds.
    parser.add_argument('--rerun_best', type=int, default=1, help='After sweep, rerun best params for multiple seeds (1/0).')
    parser.add_argument('--rerun_seed_start', type=int, default=59, help='Start seed for reruns (if --rerun_seeds not set).')
    parser.add_argument('--rerun_num_seeds', type=int, default=5, help='Number of rerun seeds (if --rerun_seeds not set).')
    parser.add_argument('--rerun_seeds', type=str, default='', help='Comma-separated explicit rerun seeds.')

    args = parser.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    args.target = targets

    run_name = f"target_{'-'.join(args.target)}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(args.seed)
    base_g = torch.Generator()
    base_g.manual_seed(args.seed)

    # Data transforms (DomainBed defaults)
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

    train_dataset, val_dataset, has_val = build_train_val_datasets(
        args, sources, data_transforms, base_g
    )

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
    }

    steps_per_epoch = max(1, math.ceil(dataset_sizes['train'] / BATCH_SIZE))
    num_epochs = max(1, math.ceil(TOTAL_STEPS / steps_per_epoch))
    args.num_epochs = num_epochs
    if args.att_epoch_max > args.num_epochs - 1:
        args.att_epoch_max = max(1, args.num_epochs - 1)

    if not has_val:
        print(f"Val split: {int(VAL_SPLIT_RATIO*100)}% split from train (no *_val.txt found)")

    test_datasets = [
        NICOWithMasks(
            args.txtdir, args.dataset, [domain], "test",
            args.mask_root,
            args.image_root,
            data_transforms['eval'],
            None
        )
        for domain in args.target
    ]

    test_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=args.num_workers, worker_init_fn=seed_worker, generator=base_g)
        for ds in test_datasets
    ]

    sampler = optuna.samplers.TPESampler(seed=args.optuna_seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists
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
        pre_lr = suggest_loguniform(trial, 'pre_lr', args.pre_lr_low, args.pre_lr_high)
        post_lr = suggest_loguniform(trial, 'post_lr', args.post_lr_low, args.post_lr_high)
        attention_epoch = suggest_int(trial, 'attention_epoch', args.att_epoch_min, args.att_epoch_max)
        kl_lambda_start = suggest_uniform(trial, 'kl_lambda_start', args.kl_start_low, args.kl_start_high)
        kl_increment = suggest_uniform(trial, 'kl_increment', args.kl_inc_low, args.kl_inc_high)
        attention_epoch = min(attention_epoch, max(1, args.num_epochs - 1))
        print(
            f"[TRIAL {trial.number}] start params={{'pre_lr': {pre_lr}, 'post_lr': {post_lr}, "
            f"'attention_epoch': {attention_epoch}, 'kl_lambda_start': {kl_lambda_start}, 'kl_increment': {kl_increment}}}",
            flush=True
        )

        trial_seed = args.seed + trial.number
        seed_everything(trial_seed)
        g = torch.Generator()
        g.manual_seed(trial_seed)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
            'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
        }

        model = make_cam_model(args.num_classes).to(DEVICE)

        trial_args = argparse.Namespace(
            pre_lr=pre_lr,
            post_lr=post_lr,
            weight_decay=WEIGHT_DECAY,
            attention_epoch=attention_epoch,
            kl_lambda_start=kl_lambda_start,
            kl_increment=kl_increment,
            num_epochs=args.num_epochs
        )

        try:
            best_val_acc = train_one_trial(model, dataloaders, dataset_sizes, trial_args, trial=trial)
            test_results = evaluate_test(model, test_loaders, args.target)
            if trial is not None:
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
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_test_results': best_trial.user_attrs.get("test_results", None),
        'n_trials': len(study.trials)
    }

    best_path = os.path.join(output_dir, 'optuna_best.json')
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)

    print("Best validation accuracy:", study.best_value)
    print("Best params:", study.best_params)
    if best["best_test_results"] is not None:
        print("Best trial test results:", best["best_test_results"])
    else:
        print("Best trial test results: not found in trial user_attrs")
    print("Saved:", best_path)

    if int(args.rerun_best) == 1:
        seeds = parse_seeds(args.rerun_seed_start, args.rerun_num_seeds, args.rerun_seeds)

        best_pre_lr = float(study.best_params['pre_lr'])
        best_post_lr = float(study.best_params.get('post_lr', best_pre_lr))
        best_post_lr_ratio = (best_post_lr / best_pre_lr) if best_pre_lr > 0 else float('nan')
        best_attention_epoch = int(study.best_params['attention_epoch'])
        best_kl_lambda_start = float(study.best_params['kl_lambda_start'])
        best_kl_increment = float(study.best_params.get('kl_increment', best_kl_lambda_start / 10.0))

        rerun_path = os.path.join(output_dir, "best_rerun_seeds.csv")
        rerun_header = [
            "seed", "pre_lr", "post_lr", "post_lr_ratio", "attention_epoch",
            "kl_lambda_start", "kl_increment", "best_val_acc", "test_results_json", "minutes"
        ]
        write_header = not os.path.exists(rerun_path)
        if write_header:
            with open(rerun_path, "w", newline="") as f:
                csv.writer(f).writerow(rerun_header)

        print(f"\n=== Rerun best params for {len(seeds)} seeds ===", flush=True)
        print(
            "Best params resolved: "
            f"pre_lr={best_pre_lr} post_lr={best_post_lr} att_epoch={best_attention_epoch} "
            f"kl_start={best_kl_lambda_start} kl_inc={best_kl_increment}",
            flush=True,
        )

        for seed in seeds:
            seed_everything(seed)
            g = torch.Generator()
            g.manual_seed(seed)

            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
                'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
            }

            model = make_cam_model(args.num_classes).to(DEVICE)

            run_args = argparse.Namespace(
                pre_lr=best_pre_lr,
                post_lr=best_post_lr,
                weight_decay=WEIGHT_DECAY,
                attention_epoch=min(best_attention_epoch, max(1, args.num_epochs - 1)),
                kl_lambda_start=best_kl_lambda_start,
                kl_increment=best_kl_increment,
                num_epochs=args.num_epochs
            )

            start = time.time()
            best_val_acc = train_one_trial(model, dataloaders, dataset_sizes, run_args, trial=None)
            test_results = evaluate_test(model, test_loaders, args.target)
            minutes = (time.time() - start) / 60.0

            print(
                f"[RERUN seed={seed}] best_val_acc={best_val_acc:.6f} test={test_results} time={minutes:.1f}m",
                flush=True,
            )

            with open(rerun_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    seed,
                    best_pre_lr,
                    best_post_lr,
                    best_post_lr_ratio,
                    best_attention_epoch,
                    best_kl_lambda_start,
                    best_kl_increment,
                    best_val_acc,
                    json.dumps(test_results),
                    minutes,
                ])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Saved reruns:", rerun_path, flush=True)


if __name__ == '__main__':
    main()
