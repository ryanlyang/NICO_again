#!/usr/bin/env python3
"""
Evaluate CLIP ViT-B/32 zero-shot accuracy on NICO++ using official txtlist splits.

Default behavior evaluates TEST accuracy on the target domains.
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


SEED = 59
DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
DEFAULT_OUTPUT_DIR = "/home/ryreu/guided_cnn/NICO_runs/output"
ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

# Canonical NICO++ class ordering used in your runs.
NICO_CLASS_NAMES = [
    "airplane", "butterfly", "clock", "dog", "football", "gun", "kangaroo", "monkey", "pumpkin", "seal", "squirrel", "train",
    "bear", "cactus", "corn", "dolphin", "fox", "hat", "lifeboat", "motorcycle", "rabbit", "sheep", "sunflower", "truck",
    "bicycle", "car", "cow", "elephant", "frog", "helicopter", "lion", "ostrich", "racket", "ship", "tent", "umbrella",
    "bird", "cat", "crab", "fishing rod", "giraffe", "horse", "lizard", "owl", "sailboat", "shrimp", "tiger", "wheat",
    "bus", "chair", "crocodile", "flower", "goose", "hot air balloon", "mailbox", "pineapple", "scooter", "spider", "tortoise", "wolf",
]


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


class NICOTxtDataset(Dataset):
    def __init__(self, txtdir: str, dataset: str, domain: str, phase: str, image_root: str, transform):
        from domainbed.datasets import _dataset_info

        txt_file = os.path.join(txtdir, dataset, f"{domain}_{phase}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Missing split file: {txt_file}")
        names, labels = _dataset_info(txt_file)
        self.paths = [_resolve_path(p, image_root) for p in names]
        self.labels = [int(y) for y in labels]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx], path


def infer_class_names_from_paths(txtdir: str, dataset: str, domains: List[str], phase: str, image_root: str) -> List[str]:
    from domainbed.datasets import _dataset_info

    label_to_name: Dict[int, str] = {}
    for domain in domains:
        txt_file = os.path.join(txtdir, dataset, f"{domain}_{phase}.txt")
        if not os.path.exists(txt_file):
            continue
        names, labels = _dataset_info(txt_file)
        for rel_path, label in zip(names, labels):
            path = _resolve_path(rel_path, image_root)
            class_key = os.path.normpath(path).split(os.sep)[-2]
            class_name = class_key.replace("_", " ")
            label_int = int(label)
            if label_int not in label_to_name:
                label_to_name[label_int] = class_name

    if not label_to_name:
        raise RuntimeError("Could not infer class names from txtlist paths.")
    max_label = max(label_to_name.keys())
    inferred = []
    for label in range(max_label + 1):
        if label not in label_to_name:
            raise RuntimeError(f"Missing class name for label {label} when inferring from paths.")
        inferred.append(label_to_name[label])
    return inferred


def get_prompt_templates(prompt_set: str) -> List[str]:
    key = prompt_set.lower()
    if key == "gals":
        return [
            "an image of a {}",
            "a photo of a {}",
        ]
    if key == "photo":
        return ["a photo of a {}"]
    if key == "imagenet":
        return [
            "a photo of a {}",
            "a blurry photo of a {}",
            "a close-up photo of a {}",
            "a bright photo of a {}",
            "a dark photo of a {}",
        ]
    raise ValueError(f"Unknown prompt_set: {prompt_set}")


def _ensure_clip_bpe(repo_root: str) -> str:
    bpe_path = os.path.join(repo_root, "GALS", "CLIP", "clip", "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(bpe_path):
        return bpe_path

    os.makedirs(os.path.dirname(bpe_path), exist_ok=True)
    urls = [
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
        "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz",
    ]
    last_err = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, bpe_path)
            break
        except Exception as err:  # pragma: no cover
            last_err = err
    if not os.path.exists(bpe_path):
        raise FileNotFoundError(
            f"Missing CLIP tokenizer vocab at {bpe_path}, and auto-download failed: {last_err}"
        )
    return bpe_path


def load_clip(model_name: str, device: str):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    _ensure_clip_bpe(repo_root)
    clip_root = os.path.join(repo_root, "GALS", "CLIP")
    if clip_root not in sys.path:
        sys.path.insert(0, clip_root)
    from clip import clip  # pylint: disable=import-error

    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess, clip


@torch.no_grad()
def build_zeroshot_text_features(model, clip_pkg, class_names: List[str], templates: List[str], device: str):
    class_features = []
    for class_name in class_names:
        prompts = [tpl.format(class_name) for tpl in templates]
        tokenized = clip_pkg.tokenize(prompts).to(device)
        text_feats = model.encode_text(tokenized)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feat = text_feats.mean(dim=0)
        text_feat = text_feat / text_feat.norm()
        class_features.append(text_feat)
    return torch.stack(class_features, dim=0)  # [C, D]


@torch.no_grad()
def evaluate_domain(model, text_features, loader, device: str):
    total, correct = 0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        image_feats = model.encode_image(images)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        logits = image_feats @ text_features.T
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct / max(total, 1), correct, total


def main():
    p = argparse.ArgumentParser(description="CLIP ViT zero-shot evaluation on NICO++")
    p.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target", nargs="+", default=["autumn", "rock", "dim", "grass", "outdoor", "water"],
                   help="Domains to evaluate (typically held-out test domains).")
    p.add_argument("--phase", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--model_name", type=str, default="ViT-B/32")
    p.add_argument("--prompt_set", type=str, default="gals", choices=["gals", "photo", "imagenet"])
    p.add_argument("--class_name_source", type=str, default="canonical", choices=["canonical", "paths"],
                   help="Use canonical NICO class names or infer class names from txtlist paths by label.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"clip_zeroshot_{args.model_name.replace('/', '-')}_{args.prompt_set}_{args.phase}_{'-'.join(targets)}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, clip_pkg = load_clip(args.model_name, device)

    templates = get_prompt_templates(args.prompt_set)
    if args.class_name_source == "paths":
        class_names = infer_class_names_from_paths(
            txtdir=args.txtdir,
            dataset=args.dataset,
            domains=targets,
            phase=args.phase,
            image_root=args.image_root,
        )
    else:
        class_names = NICO_CLASS_NAMES

    text_features = build_zeroshot_text_features(
        model=model,
        clip_pkg=clip_pkg,
        class_names=class_names,
        templates=templates,
        device=device,
    )

    results: Dict[str, float] = {}
    all_correct = 0
    all_total = 0
    for domain in targets:
        ds = NICOTxtDataset(
            txtdir=args.txtdir,
            dataset=args.dataset,
            domain=domain,
            phase=args.phase,
            image_root=args.image_root,
            transform=preprocess,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=torch.cuda.is_available(),
        )
        acc, correct, total = evaluate_domain(model, text_features, loader, device)
        results[domain] = acc
        all_correct += correct
        all_total += total
        print(f"[{domain}] {args.phase}_acc={acc:.4f}% ({correct}/{total})", flush=True)

    results["overall"] = 100.0 * all_correct / max(all_total, 1)
    print(f"[overall] {args.phase}_acc={results['overall']:.4f}% ({all_correct}/{all_total})", flush=True)

    payload = {
        "model_name": args.model_name,
        "prompt_set": args.prompt_set,
        "prompt_templates": templates,
        "class_name_source": args.class_name_source,
        "num_classes": len(class_names),
        "phase": args.phase,
        "targets": targets,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "results": results,
    }
    out_path = os.path.join(out_dir, "clip_zeroshot_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
