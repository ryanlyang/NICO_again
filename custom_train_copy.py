#!/usr/bin/env python3
"""
Guided CNN Training on NICO++ with Table 2 Hyperparameters
Keeps all guided CNN mechanisms (masks, CAMs, KL divergence) while using paper setup
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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import torchvision.models as models


# =============================================================================
# HYPERPARAMETERS (will be overridden by command-line args)
# =============================================================================
batch_size = 192        # Paper: 192
num_epochs = 30         # Shortened to 30 epochs
learning_rate = 0.002   # Paper: 2e-3
momentum = 0.9          # Paper: SGD momentum
gamma = 0.1             # Paper: decay to 2e-4 (0.1x)
weight_decay = 0.001    # Paper: 1e-3

# These will be set from command-line arguments
attention_epoch = 15    # Default: Middle of 30 epochs
kl_lambda_start = 15.0  # Default: Initial KL weight
kl_increment = 1.5      # Default: 10% of kl_lambda_start
step_size = 25          # Will be set to attention_epoch

checkpoint_dir = "guided_cnn_checkpoints_copy"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 59  # Match Table 2


# =============================================================================
# SEEDING & UTILS
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
# MASK TRANSFORMS
# =============================================================================

class ExpandWhite(object):
    def __init__(self, thr: int = 10, radius: int = 3):
        self.thr = thr
        self.radius = radius

    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1))
        dil = cv2.dilate(white, k, iterations=1)
        return Image.fromarray((dil * 255).astype(np.uint8))


class EdgeExtract(object):
    def __init__(self, thr: int = 10, edge_width: int = 1):
        self.thr = thr
        self.edge_width = edge_width

    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.edge_width + 1, 2 * self.edge_width + 1))
        edge = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, k)
        return Image.fromarray((edge * 255).astype(np.uint8))


class Brighten(object):
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(mask * self.factor, 0.0, 1.0)


# =============================================================================
# DATASET WITH MASKS
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

        # Load file names and labels from txt files
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
            # For training: apply synchronized random augmentations
            if 'RandomResizedCrop' in str(self.image_transform):
                # Training transforms - apply same augmentation to both
                image, mask = self._apply_train_transforms(image, mask)
            else:
                # Eval transforms - just resize and normalize
                image, mask = self._apply_eval_transforms(image, mask)

        return image, label, mask, img_path

    def _apply_train_transforms(self, image, mask):
        """Apply synchronized training transforms to image and mask"""
        import torchvision.transforms.functional as TF

        # Step 1: RandomResizedCrop with same parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.7, 1.0), ratio=(3./4., 4./3.)
        )
        image = TF.resized_crop(image, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, size=(224, 224), interpolation=TF.InterpolationMode.NEAREST)

        # Step 2: RandomHorizontalFlip with same decision
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Step 3: RandomGrayscale (only on image)
        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)

        # Step 4: Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = TF.to_tensor(mask)
        # Brighten mask
        mask = torch.clamp(mask * 8.0, 0.0, 1.0)

        return image, mask

    def _apply_eval_transforms(self, image, mask):
        """Apply evaluation transforms (no augmentation)"""
        import torchvision.transforms.functional as TF

        # Resize
        image = TF.resize(image, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (224, 224), interpolation=TF.InterpolationMode.NEAREST)

        # Convert to tensor and normalize
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
            # Extract: domain/class/filename from NICO_dataset/domain/class/filename.jpg
            rel_path = img_path.split('NICO_dataset/')[-1]
            parts = rel_path.split('/')

            if len(parts) == 3:
                domain = parts[0]      # e.g., 'autumn'
                class_name = parts[1]  # e.g., 'airplane'
                filename = parts[2]    # e.g., 'autumn_0000001.jpg'

                # Remove extension and build mask filename
                basename = os.path.splitext(filename)[0]  # 'autumn_0000001'
                mask_filename = f"{domain}_{class_name}_{basename}.png"
                mask_path = os.path.join(self.mask_root, mask_filename)
            else:
                # Fallback: just use basename
                basename = os.path.basename(img_path)
                basename = os.path.splitext(basename)[0] + '.png'
                mask_path = os.path.join(self.mask_root, basename)
        else:
            # Fallback: just use basename
            basename = os.path.basename(img_path)
            basename = os.path.splitext(basename)[0] + '.png'
            mask_path = os.path.join(self.mask_root, basename)

        return mask_path


# =============================================================================
# MODEL: ResNet-50 with CAM (like your LeNet with CAM)
# =============================================================================

def make_cam_model(num_classes):
    """
    ResNet-50 with CAM extraction hook
    Similar to your LeNet CAM wrapper but for ResNet
    """
    base = models.resnet50(pretrained=True)

    # Replace final FC layer
    num_features = base.fc.in_features
    base.fc = nn.Linear(num_features, num_classes)

    class CAMWrapResNet(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.features = None

            # Hook into layer4 (final conv layer before avgpool)
            self.base.layer4.register_forward_hook(self._hook_fn)

        def _hook_fn(self, module, inp, out):
            self.features = out

        def forward(self, x):
            out = self.base(x)
            return out, self.features

    return CAMWrapResNet(base)


# =============================================================================
# LOSS FUNCTION (Your Guided CNN Loss)
# =============================================================================

def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_ce):
    """
    Your guided CNN loss with KL divergence between CAMs and GT masks
    """
    ce_loss = nn.functional.cross_entropy(outputs, labels)

    # Flatten CAMs and GT masks
    B, C, Hf, Wf = cams.shape
    # Average over channels to get single attention map
    cam_avg = cams.mean(dim=1)  # (B, Hf, Wf)

    cam_flat = cam_avg.view(B, -1)
    gt_flat = gt_masks.view(B, -1)

    # Softmax for CAM, normalize GT
    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)

    # KL divergence
    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)

    if only_ce:
        return ce_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss


# =============================================================================
# TRAINING FUNCTION (Your train_model with Table 2 hyperparameters)
# =============================================================================

def train_model(model, dataloaders, dataset_sizes, test_loaders, args):
    """
    Train with guided CNN approach but using Table 2 hyperparameters

    Key features:
    - Attention epoch at middle (epoch 30 of 60)
    - Optimizer/scheduler RESET at attention epoch
    - KL lambda increases by 10 per epoch after attention epoch
    - Before attention epoch: only attention loss
    - After attention epoch: CE + KL loss
    - Evaluates on test domains after each epoch past attention_epoch
    """
    best_wts = copy.deepcopy(model.state_dict())
    best_optim = -100.0
    since = time.time()

    # Initial optimizer & scheduler
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    sch = optim.lr_scheduler.StepLR(opt, step_size=args.attention_epoch, gamma=gamma)

    kl_lambda_real = args.kl_lambda_start

    for epoch in range(num_epochs):
        # =====================================================================
        # RESTART at attention_epoch (VERY IMPORTANT!)
        # =====================================================================
        if epoch == args.attention_epoch:
            print(f"\n*** ATTENTION EPOCH {epoch} REACHED: RESTARTING OPTIMIZER & SCHEDULER ***\n")
            # Reset to initial learning rate (2e-3)
            opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            sch = optim.lr_scheduler.StepLR(opt, step_size=args.attention_epoch, gamma=gamma)
            # Reset best weights
            best_wts = copy.deepcopy(model.state_dict())
            best_optim = -100.0

        # Increase KL lambda after attention epoch (10% of initial = 10.0)
        if epoch > args.attention_epoch:
            kl_lambda_real += args.kl_increment

        print(f"\nEpoch {epoch + 1}/{num_epochs}, KL Lambda: {kl_lambda_real:.1f}")

        for phase in ['train', 'val_in']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_attn_loss = 0.0

            for batch in dataloaders[phase]:
                inputs, labels, gt_masks, paths = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                gt_masks = gt_masks.to(device)

                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, feats = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # =========================================================
                    # COMPUTE CAMs (using class-specific weights)
                    # =========================================================
                    # Get weights for the predicted classes
                    weights = model.base.fc.weight[labels]  # (B, num_features)

                    # Global Average Pool features
                    B, C, H, W = feats.shape
                    feats_gap = feats.view(B, C, -1).mean(dim=2)  # (B, C)

                    # Compute CAMs: weighted combination of feature maps
                    cams = torch.einsum('bc,bchw->bhw', weights, feats)  # (B, H, W)
                    cams = torch.relu(cams)

                    # Normalize CAMs to [0, 1]
                    flat = cams.view(cams.size(0), -1)
                    mn, _ = flat.min(dim=1, keepdim=True)
                    mx, _ = flat.max(dim=1, keepdim=True)
                    sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

                    # Resize GT masks to match CAM size
                    gt_small = nn.functional.interpolate(
                        gt_masks, size=sal_norm.shape[1:], mode='nearest'
                    ).squeeze(1)

                    # =========================================================
                    # COMPUTE LOSS (your guided CNN approach)
                    # =========================================================
                    if epoch < args.attention_epoch:
                        # Before attention epoch: ONLY attention loss
                        loss, attn_loss = compute_loss(
                            outputs, labels,
                            feats,  # Pass full features
                            gt_small,
                            kl_lambda=333,  # High value to force attention learning
                            only_ce=True
                        )
                    else:
                        # After attention epoch: CE + KL loss
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

            # Step scheduler at end of each training phase
            if is_train:
                sch.step()

            # Epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_attn_loss = running_attn_loss / dataset_sizes[phase]

            print(f"{phase:8} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Attn_Loss: {epoch_attn_loss:.4f}")

            # Save best model based on validation
            if phase == 'val_in':
                # Optimization metric: accuracy * (1 - attn_loss)
                optim_num = epoch_acc * (1 - epoch_attn_loss)
                print(f"{phase:8} Optim Num: {optim_num:.4f}")

                if (epoch >= args.attention_epoch) and (optim_num > best_optim):
                    best_optim = optim_num
                    best_wts = copy.deepcopy(model.state_dict())
                    print(f"         -> New best model saved!")

        # =====================================================================
        # EVALUATE ON TEST after attention epoch
        # =====================================================================
        if epoch >= args.attention_epoch:
            print(f"\n*** Evaluating on test domains at epoch {epoch + 1} ***")
            from torch.utils.data import Subset
            test_results_epoch = evaluate_test(model, test_loaders, args.target)
            print(f"Epoch {epoch + 1} Test Results:")
            for domain, acc in test_results_epoch.items():
                print(f"  {domain}: {acc:.2f}%")
            print()

    print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val optim metric: {best_optim:.4f}")

    # Load best weights
    model.load_state_dict(best_wts)
    return model, best_optim


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_test(model, test_loaders, target_domains):
    """Evaluate on test domains"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    results = {}
    all_correct = 0
    all_total = 0

    for test_loader, domain_name in zip(test_loaders, target_domains):
        total, correct, total_loss = 0, 0, 0.0

        for batch in test_loader:
            if len(batch) == 4:  # (image, label, mask, path)
                images, labels, _, _ = batch
            else:  # (image, label)
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)

        results[domain_name] = acc
        all_correct += correct
        all_total += total

        print(f"Test {domain_name}: {acc:.2f}%")

    # Overall accuracy
    overall_acc = 100.0 * all_correct / max(all_total, 1)
    results['overall'] = overall_acc
    print(f"Test overall: {overall_acc:.2f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Guided CNN on NICO++ with Table 2 Hyperparameters')

    # Data arguments
    parser.add_argument('--txtdir', type=str, default='txtlist', help='Path to txt lists')
    parser.add_argument('--dataset', type=str, default='NICO', help='Dataset name')
    parser.add_argument('--source', nargs='+', required=True, help='Source domains')
    parser.add_argument('--target', nargs='+', required=True, help='Target domains')
    parser.add_argument('--domain_split', type=str, required=True, help='Domain split name')
    parser.add_argument('--num_classes', type=int, default=60, help='Number of classes')
    parser.add_argument('--mask_root', type=str, required=True, help='Path to mask directory')

    # Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    # Hyperparameters for visualization sweep
    parser.add_argument('--attention_epoch', type=int, default=30, help='Epoch to start attention supervision')
    parser.add_argument('--kl_lambda_start', type=float, default=15.0, help='Initial KL divergence weight')
    parser.add_argument('--kl_increment', type=float, default=1.5, help='KL weight increment per epoch')

    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # =========================================================================
    # DATA TRANSFORMS (Paper: 224x224, ImageNet normalization)
    # =========================================================================
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
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

    mask_transforms = {
        'train': transforms.Compose([
            # ExpandWhite(thr=10, radius=3),
            # EdgeExtract(thr=10, edge_width=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Brighten(8.0),
        ]),
        'eval': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print("Loading data...")

    # Create dataset WITHOUT transforms to get the split indices
    full_train_base = NICOWithMasks(
        args.txtdir, args.dataset, args.source, "train",
        args.mask_root,
        None,  # No transforms yet
        None
    )

    # Split into train/val indices (16% validation)
    n_total = len(full_train_base)
    n_val_in = max(1, int(0.16 * n_total))
    n_train = n_total - n_val_in

    # Get the indices from random_split
    train_indices, val_indices = random_split(range(n_total), [n_train, n_val_in], generator=g)
    train_idx_list = train_indices.indices
    val_idx_list = val_indices.indices

    # Create TWO separate datasets: one with train transforms, one with eval transforms
    train_dataset = NICOWithMasks(
        args.txtdir, args.dataset, args.source, "train",
        args.mask_root,
        data_transforms['train'],  # Augmented transforms
        mask_transforms['train']
    )

    val_dataset = NICOWithMasks(
        args.txtdir, args.dataset, args.source, "train",
        args.mask_root,
        data_transforms['eval'],  # NO augmentation for validation
        mask_transforms['eval']
    )

    # Create subsets using the same indices but different underlying datasets
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_idx_list)
    val_in_subset = Subset(val_dataset, val_idx_list)

    # Test datasets (one per target domain)
    test_datasets = [
        NICOWithMasks(
            args.txtdir, args.dataset, [domain], "test",
            args.mask_root,
            data_transforms['eval'],
            mask_transforms['eval']
        )
        for domain in args.target
    ]

    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                           num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
        'val_in': DataLoader(val_in_subset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g),
    }

    dataset_sizes = {
        'train': len(train_subset),
        'val_in': len(val_in_subset),
    }

    test_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=False,
                  num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        for ds in test_datasets
    ]

    print(f"Train dataset size: {dataset_sizes['train']}")
    print(f"Val dataset size: {dataset_sizes['val_in']}")
    print(f"Test datasets: {[len(ds) for ds in test_datasets]}")

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("\nCreating ResNet-50 with CAM...")
    model = make_cam_model(args.num_classes).to(device)

    # =========================================================================
    # TRAIN
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Starting Training with Guided CNN")
    print(f"{'='*70}")
    print(f"Hyperparameters (Table 2):")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Attention epoch: {args.attention_epoch}")
    print(f"  KL lambda start: {args.kl_lambda_start}")
    print(f"  KL increment: {args.kl_increment} (per epoch after attention)")
    print(f"{'='*70}\n")

    best_model, best_score = train_model(model, dataloaders, dataset_sizes, test_loaders, args)

    # =========================================================================
    # EVALUATE ON TEST
    # =========================================================================
    print(f"\n{'='*70}")
    print("Evaluating on Test Domains")
    print(f"{'='*70}\n")

    test_results = evaluate_test(best_model, test_loaders, args.target)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    # Skip model saving for test_copy (only need test accuracies per epoch)
    model_path = "(not saved - test_copy version)"

    # Save results
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Guided CNN on NICO++\n")
        f.write(f"Domain Split: {args.domain_split}\n")
        f.write(f"Source Domains: {args.source}\n")
        f.write(f"Target Domains: {args.target}\n\n")
        f.write("Test Results:\n")
        for domain, acc in test_results.items():
            f.write(f"  {domain}: {acc:.2f}%\n")

    # Save training log
    log_file = os.path.join(args.output_dir, 'training.log')
    with open(log_file, 'w') as f:
        f.write("Training Complete!\n")
        f.write(f"Best val optim metric: {best_score:.4f}\n\n")
        f.write("Test Results:\n")
        for domain, acc in test_results.items():
            f.write(f"Test {domain}: {acc:.2f}%\n")

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
