#!/usr/bin/env python3
"""
Run one guided-CNN training run on NICO++ with official splits.
- Target domains are provided at runtime; source domains are the remaining ones.
- Hyperparameters follow the Table 2 setup used in custom_train_copy.py.
- Model selection is based on validation accuracy (official val split).
"""

import os
import time
import copy
import argparse
import random

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
import torchvision.models as models

# =============================================================================
# HYPERPARAMETERS (Table 2 values from custom_train_copy.py)
# =============================================================================
BATCH_SIZE = 192
NUM_EPOCHS = 30
PRE_ATTENTION_LR = 0.002
POST_ATTENTION_LR = 0.0002
MOMENTUM = 0.9
GAMMA = 0.1
WEIGHT_DECAY = 0.001

DEFAULT_ATTENTION_EPOCH = 15
DEFAULT_KL_LAMBDA_START = 15.0
DEFAULT_KL_INCREMENT = 1.5
VAL_SPLIT_RATIO = 0.16

SEED = 59
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
                 image_transform=None, mask_transform=None):
        self.mask_root = mask_root
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

        self.image_paths = all_names
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

        # RandomResizedCrop with same parameters (DomainBed scale 0.8-1.0)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(3./4., 4./3.)
        )
        image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.NEAREST)

        # RandomHorizontalFlip with same decision
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # RandomGrayscale (only on image)
        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # Convert to tensor and normalize
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

    def _get_mask_path(self, img_path):
        """
        Convert image path to mask path based on flat directory structure.

        Image path: NICO_dataset/autumn/airplane/autumn_0000001.jpg
        Mask path:  {mask_root}/autumn_airplane_autumn_0000001.png
        """
        if 'NICO_dataset' in img_path:
            rel_path = img_path.split('NICO_dataset/')[-1]
            parts = rel_path.split('/')

            if len(parts) == 3:
                domain = parts[0]
                class_name = parts[1]
                filename = parts[2]

                basename = os.path.splitext(filename)[0]
                mask_filename = f"{domain}_{class_name}_{basename}.png"
                mask_path = os.path.join(self.mask_root, mask_filename)
            else:
                basename = os.path.basename(img_path)
                basename = os.path.splitext(basename)[0] + '.png'
                mask_path = os.path.join(self.mask_root, basename)
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
            data_transforms['train'],
            None
        )
        val_dataset = NICOWithMasks(
            args.txtdir, args.dataset, sources, "val",
            args.mask_root,
            data_transforms['eval'],
            None
        )
        return train_dataset, val_dataset, has_val

    # Fallback: create validation split from train
    full_train_base = NICOWithMasks(
        args.txtdir, args.dataset, sources, "train",
        args.mask_root,
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
        data_transforms['train'],
        None
    )
    val_dataset = NICOWithMasks(
        args.txtdir, args.dataset, sources, "train",
        args.mask_root,
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
# LOSS FUNCTION (guided CNN loss)
# =============================================================================

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
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model, dataloaders, dataset_sizes, test_loaders, args):
    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    since = time.time()

    opt = optim.SGD(model.parameters(), lr=args.pre_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.StepLR(opt, step_size=args.attention_epoch, gamma=GAMMA)

    kl_lambda_real = args.kl_lambda_start

    for epoch in range(NUM_EPOCHS):
        if epoch == args.attention_epoch:
            print(f"\n*** ATTENTION EPOCH {epoch} REACHED: RESTARTING OPTIMIZER & SCHEDULER ***\n")
            opt = optim.SGD(model.parameters(), lr=args.post_lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.StepLR(opt, step_size=args.attention_epoch, gamma=GAMMA)
            best_wts = copy.deepcopy(model.state_dict())
            best_val_acc = -1.0

        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}, KL Lambda: {kl_lambda_real:.1f}")

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

                    # CAMs (class-specific weights)
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
                            only_ce=True
                        )
                    else:
                        loss, attn_loss = compute_loss(
                            outputs, labels,
                            feats,
                            gt_small,
                            kl_lambda=kl_lambda_real,
                            only_ce=False
                        )

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_attn_loss += attn_loss.item() * inputs.size(0)

            if is_train:
                sch.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_attn_loss = running_attn_loss / dataset_sizes[phase]

            print(f"{phase:8} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Attn_Loss: {epoch_attn_loss:.4f}")

            if phase == 'val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    print("         -> New best model saved based on val accuracy!")

        if epoch >= args.attention_epoch:
            print(f"\n*** Evaluating on test domains at epoch {epoch + 1} ***")
            test_results_epoch = evaluate_test(model, test_loaders, args.target)
            print(f"Epoch {epoch + 1} Test Results:")
            for domain, acc in test_results_epoch.items():
                print(f"  {domain}: {acc:.2f}%")
            print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val acc: {best_val_acc:.4f}")

    model.load_state_dict(best_wts)
    return model, best_val_acc


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

        print(f"Test {domain_name}: {acc:.2f}%")

    overall_acc = 100.0 * all_correct / max(all_total, 1)
    results['overall'] = overall_acc
    print(f"Test overall: {overall_acc:.2f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Guided CNN on NICO++ (official splits)')

    parser.add_argument('--txtdir', type=str, required=True, help='Path to txt lists')
    parser.add_argument('--dataset', type=str, default='NICO', help='Dataset name')
    parser.add_argument('--target', nargs='+', required=True, help='Target domains')
    parser.add_argument('--num_classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--mask_root', type=str, required=True, help='Path to mask directory')

    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    parser.add_argument('--attention_epoch', type=int, default=DEFAULT_ATTENTION_EPOCH, help='Epoch to start attention supervision')
    parser.add_argument('--pre_lr', type=float, default=PRE_ATTENTION_LR, help='Learning rate before attention epoch')
    parser.add_argument('--post_lr', type=float, default=POST_ATTENTION_LR, help='Learning rate after attention epoch')
    parser.add_argument('--kl_lambda_start', type=float, default=DEFAULT_KL_LAMBDA_START, help='Initial KL divergence weight')
    parser.add_argument('--kl_increment', type=float, default=DEFAULT_KL_INCREMENT, help='KL weight increment per epoch after attention')

    args = parser.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain to train on.")

    args.target = targets

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    run_name = f"target_{'-'.join(args.target)}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

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

    print("Loading data...")

    train_dataset, val_dataset, has_val = build_train_val_datasets(
        args, sources, data_transforms, g
    )

    test_datasets = [
        NICOWithMasks(
            args.txtdir, args.dataset, [domain], "test",
            args.mask_root,
            data_transforms['eval'],
            None
        )
        for domain in args.target
    ]

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
    }

    test_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                   num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        for ds in test_datasets
    ]

    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val']}")
    if not has_val:
        print(f"Val split: {int(VAL_SPLIT_RATIO*100)}% split from train (no *_val.txt found)")
    print(f"Test datasets: {[len(ds) for ds in test_datasets]}")
    print(f"Source domains: {sources}")
    print(f"Target domains: {args.target}")

    print("\nCreating ResNet-50 with CAM...")
    model = make_cam_model(args.num_classes).to(DEVICE)

    print(f"\n{'='*70}")
    print("Starting Training with Guided CNN")
    print(f"{'='*70}")
    print("Hyperparameters (Table 2):")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Pre-attention LR: {args.pre_lr}")
    print(f"  Post-attention LR: {args.post_lr}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Attention epoch: {args.attention_epoch}")
    print(f"  KL lambda start: {args.kl_lambda_start}")
    print(f"  KL increment: {args.kl_increment} (per epoch after attention)")
    print(f"{'='*70}\n")

    best_model, best_val_acc = train_model(model, dataloaders, dataset_sizes, test_loaders, args)

    print(f"\n{'='*70}")
    print("Evaluating on Test Domains")
    print(f"{'='*70}\n")

    test_results = evaluate_test(best_model, test_loaders, args.target)

    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("Guided CNN on NICO++\n")
        f.write(f"Source Domains: {sources}\n")
        f.write(f"Target Domains: {args.target}\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n\n")
        f.write("Test Results:\n")
        for domain, acc in test_results.items():
            f.write(f"  {domain}: {acc:.2f}%\n")

    log_file = os.path.join(output_dir, 'training.log')
    with open(log_file, 'w') as f:
        f.write("Training Complete!\n")
        f.write(f"Best val acc: {best_val_acc:.4f}\n\n")
        f.write("Test Results:\n")
        for domain, acc in test_results.items():
            f.write(f"Test {domain}: {acc:.2f}%\n")

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
