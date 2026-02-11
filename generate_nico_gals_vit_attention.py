#!/usr/bin/env python3
"""
Precompute GALS-style CLIP ViT attention maps for NICO++.

This follows the GALS prompt style:
  - "an image of a <class>"
  - "a photo of a <class>"

Attention extraction uses CLIP ViT-B/32 transformer attention.
Outputs are saved as .pth files that mirror image paths under --output_root.
"""

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


SEED = 59
DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
DEFAULT_OUTPUT_ROOT = "/home/ryreu/guided_cnn/NICO_runs/attention_maps/nico_gals_vit_b32"
ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]

_all_new_class_names = [
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


def _relative_from_image_root(path: str, image_root: str) -> str:
    root_norm = os.path.normpath(image_root)
    path_norm = os.path.normpath(path)
    try:
        rel = os.path.relpath(path_norm, root_norm)
        if not rel.startswith(".."):
            return rel
    except ValueError:
        pass
    parts = path_norm.split(os.sep)
    # Fallback: assume .../<domain>/<class>/<file>
    if len(parts) >= 3:
        return os.path.join(parts[-3], parts[-2], parts[-1])
    return os.path.basename(path_norm)


def _class_from_path(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 2:
        return "object"
    class_key = parts[-2].lower()
    return class_key


def _build_prompt_map() -> Dict[str, str]:
    prompt_map = {}
    for name in _all_new_class_names:
        key = name.lower().replace(" ", "_")
        prompt_map[key] = name.lower()
    return prompt_map


def _build_prompts_for_class(class_key: str, prompt_map: Dict[str, str]) -> List[str]:
    if class_key in prompt_map:
        class_text = prompt_map[class_key]
    else:
        class_text = class_key.replace("_", " ")
    return [
        f"an image of a {class_text}",
        f"a photo of a {class_text}",
    ]


def _normalize_attention(att: torch.Tensor) -> torch.Tensor:
    # att: (1,1,H,W)
    b, c, h, w = att.shape
    flat = att.view(b, -1)
    maxv = flat.max(dim=1, keepdim=True)[0]
    maxv[maxv == 0] = 1.0
    flat = flat - flat.min(dim=1, keepdim=True)[0]
    flat = flat / maxv
    return flat.view(b, c, h, w)


def _compute_vit_transformer_attention(model, image, tokenized_text, device):
    """
    Adapted from GALS/utils/attention_utils.py transformer_attention()
    without optional plotting and extra dependencies.
    """
    logits_per_image, _ = model(image, tokenized_text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu()

    attentions = []
    unnormalized_attentions = []

    for idx in range(tokenized_text.shape[0]):
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot[0, idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits_per_image)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        rel = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype, device=device)

        for blk in image_attn_blocks:
            grad = blk.attn_grad.reshape(-1, blk.attn_grad.shape[-1], blk.attn_grad.shape[-1])
            cam = blk.attn_probs.reshape(-1, blk.attn_probs.shape[-1], blk.attn_probs.shape[-1])
            cam = (grad * cam).clamp(min=0).mean(dim=0)
            rel = rel + torch.matmul(cam, rel)

        rel[0, 0] = 0
        image_relevance = rel[0, 1:]
        side = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, side, side).detach().float().cpu()

        unnormalized_attentions.append(image_relevance)
        attentions.append(_normalize_attention(image_relevance))

    return {
        "unnormalized_attentions": torch.cat(unnormalized_attentions, dim=0),
        "attentions": torch.cat(attentions, dim=0),
        "probs": probs,
    }


def _load_clip_vit(device):
    gals_clip_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GALS", "CLIP")
    if gals_clip_root not in sys.path:
        sys.path.insert(0, gals_clip_root)
    from clip import clip  # pylint: disable=import-error

    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Match GALS behavior: resize to 224x224 and skip center crop.
    preprocess_no_crop = []
    for t in preprocess.transforms:
        if isinstance(t, transforms.Resize):
            preprocess_no_crop.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        elif isinstance(t, transforms.CenterCrop):
            continue
        else:
            preprocess_no_crop.append(t)
    preprocess_no_crop = transforms.Compose(preprocess_no_crop)
    return model, preprocess_no_crop, clip


def _collect_records(txtdir: str, dataset: str, domains: List[str], image_root: str) -> List[Tuple[str, int]]:
    from domainbed.datasets import _dataset_info

    records = {}
    for domain in domains:
        for phase in ("train", "val", "test"):
            txt_file = os.path.join(txtdir, dataset, f"{domain}_{phase}.txt")
            if not os.path.exists(txt_file):
                continue
            names, labels = _dataset_info(txt_file)
            for path, label in zip(names, labels):
                abs_path = _resolve_path(path, image_root)
                records[abs_path] = int(label)
    out = [(k, v) for k, v in records.items()]
    out.sort(key=lambda x: x[0])
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate GALS ViT attention maps for NICO++")
    parser.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    parser.add_argument("--dataset", type=str, default="NICO")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--domains", type=str, default=",".join(ALL_DOMAINS),
                        help="Comma-separated domains to include.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", type=int, default=0, help="1 to overwrite existing .pth maps")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index into sorted unique image list.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index (exclusive). -1 means process until the end.")
    args = parser.parse_args()

    domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in domains if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}. Valid: {ALL_DOMAINS}")

    seed_everything(args.seed)
    os.makedirs(args.output_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = _collect_records(args.txtdir, args.dataset, domains, args.image_root)
    if len(records) == 0:
        raise RuntimeError("No image records found from txtlist files.")

    start_idx = max(0, int(args.start_idx))
    end_idx = len(records) if int(args.end_idx) < 0 else min(len(records), int(args.end_idx))
    if start_idx >= end_idx:
        raise ValueError(
            f"Invalid range start_idx={start_idx}, end_idx={end_idx}, total={len(records)}."
        )

    records = records[start_idx:end_idx]

    model, preprocess, clip_pkg = _load_clip_vit(device)
    prompt_map = _build_prompt_map()
    token_cache: Dict[str, torch.Tensor] = {}
    prompt_cache: Dict[str, List[str]] = {}

    total = len(records)
    written = 0
    skipped = 0
    missing = 0

    for abs_path, _ in tqdm(records, desc="Generating ViT attention"):
        if not os.path.exists(abs_path):
            missing += 1
            continue

        rel = _relative_from_image_root(abs_path, args.image_root)
        out_path = os.path.join(args.output_root, rel)
        out_path = os.path.splitext(out_path)[0] + ".pth"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (args.overwrite == 0) and os.path.exists(out_path):
            skipped += 1
            continue

        class_key = _class_from_path(abs_path)
        if class_key not in token_cache:
            prompts = _build_prompts_for_class(class_key, prompt_map)
            tokens = clip_pkg.tokenize(prompts).to(device)
            token_cache[class_key] = tokens
            prompt_cache[class_key] = prompts

        image = preprocess(Image.open(abs_path).convert("RGB")).unsqueeze(0).to(device)
        att = _compute_vit_transformer_attention(model, image, token_cache[class_key], device)
        att["text_list"] = prompt_cache[class_key]
        att["class_key"] = class_key
        att["image_path"] = abs_path
        torch.save(att, out_path)
        written += 1

    print("\n=== DONE ===")
    print(f"Output root: {args.output_root}")
    print(f"Selected range: [{start_idx}, {end_idx})")
    print(f"Total records: {total}")
    print(f"Written: {written}")
    print(f"Skipped existing: {skipped}")
    print(f"Missing images: {missing}")


if __name__ == "__main__":
    main()
