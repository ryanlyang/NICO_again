#!/usr/bin/env python3
"""
CLIP + Logistic Regression hyperparameter sweep for NICO++.

Pipeline:
1) Extract fixed CLIP image embeddings once for train/val/test.
2) Sweep sklearn LogisticRegression hyperparameters on top.
3) Optimize validation objective (default: val_avg_group_acc).
4) Re-run best hyperparameters for multiple seeds.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from threadpoolctl import threadpool_limits

try:
    from sklearn.linear_model import LogisticRegression
except Exception as exc:
    raise SystemExit("scikit-learn is required (pip install scikit-learn).") from exc

try:
    import optuna  # type: ignore
except Exception:
    optuna = None


SEED = 59
DEFAULT_TXTLIST_DIR = "/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist"
DEFAULT_IMAGE_ROOT = "/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG"
DEFAULT_OUTPUT_DIR = "/home/ryreu/guided_cnn/NICO_runs/output"
ALL_DOMAINS = ["autumn", "rock", "dim", "grass", "outdoor", "water"]


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


@dataclass
class Record:
    path: str
    y: int
    domain_idx: int
    domain_name: str


class NICORecordsDataset(Dataset):
    def __init__(self, records: Sequence[Record], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec.path).convert("RGB")
        image = self.transform(image)
        return image, rec.y, rec.domain_idx, rec.domain_name


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
    return model, preprocess


def load_records_from_txtlist(
    txtdir: str,
    dataset: str,
    image_root: str,
    domains: Sequence[str],
    phase: str,
) -> List[Record]:
    from domainbed.datasets import _dataset_info

    records: List[Record] = []
    for d in domains:
        txt_file = os.path.join(txtdir, dataset, f"{d}_{phase}.txt")
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Missing split file: {txt_file}")
        names, labels = _dataset_info(txt_file)
        domain_idx = ALL_DOMAINS.index(d)
        for p, y in zip(names, labels):
            records.append(Record(_resolve_path(p, image_root), int(y), domain_idx, d))
    return records


def build_train_val_test_records(
    txtdir: str,
    dataset: str,
    image_root: str,
    targets: Sequence[str],
    seed: int,
    val_split_ratio: float,
) -> Tuple[List[Record], List[Record], List[Record], bool]:
    sources = [d for d in ALL_DOMAINS if d not in targets]

    # Check if explicit val files exist for all source domains.
    has_val = all(os.path.exists(os.path.join(txtdir, dataset, f"{d}_val.txt")) for d in sources)
    if has_val:
        train_records = load_records_from_txtlist(txtdir, dataset, image_root, sources, "train")
        val_records = load_records_from_txtlist(txtdir, dataset, image_root, sources, "val")
    else:
        rng = np.random.default_rng(seed)
        train_records = []
        val_records = []
        for d in sources:
            domain_train = load_records_from_txtlist(txtdir, dataset, image_root, [d], "train")
            idx = np.arange(len(domain_train))
            rng.shuffle(idx)
            n_val = max(1, int(val_split_ratio * len(domain_train)))
            val_idx = set(idx[:n_val].tolist())
            for i, rec in enumerate(domain_train):
                if i in val_idx:
                    val_records.append(rec)
                else:
                    train_records.append(rec)

    test_records = load_records_from_txtlist(txtdir, dataset, image_root, targets, "test")
    return train_records, val_records, test_records, has_val


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


@torch.no_grad()
def extract_features(model, preprocess, records: Sequence[Record], batch_size: int, num_workers: int, device: str):
    ds = NICORecordsDataset(records, preprocess)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )

    feat_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    d_idx_list: List[np.ndarray] = []
    d_name_list: List[str] = []

    for images, ys, d_idxs, d_names in loader:
        images = images.to(device, non_blocking=True)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feat_list.append(feats.float().cpu().numpy())
        y_list.append(ys.numpy())
        d_idx_list.append(d_idxs.numpy())
        d_name_list.extend(list(d_names))

    X = np.concatenate(feat_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.int64, copy=False)
    d_idx = np.concatenate(d_idx_list, axis=0).astype(np.int64, copy=False)
    d_name = np.array(d_name_list)

    # Hardening for sklearn solvers.
    X = l2_normalize_rows(X)
    X = np.ascontiguousarray(X.astype(np.float64, copy=False))
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, y, d_idx, d_name


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, d_names: np.ndarray):
    acc = 100.0 * float((y_true == y_pred).mean())

    group_total: Dict[str, int] = {}
    group_correct: Dict[str, int] = {}
    for yt, yp, dn in zip(y_true.tolist(), y_pred.tolist(), d_names.tolist()):
        key = f"class{yt}@{dn}"
        group_total[key] = group_total.get(key, 0) + 1
        group_correct[key] = group_correct.get(key, 0) + int(yt == yp)

    group_items = []
    group_accs = []
    for key in sorted(group_total.keys()):
        total = group_total[key]
        correct = group_correct[key]
        gacc = correct / max(total, 1)
        group_accs.append(gacc)
        group_items.append({"group": key, "correct": correct, "total": total, "acc": 100.0 * gacc})

    if len(group_accs) == 0:
        avg_group = 0.0
        worst_group = 0.0
    else:
        avg_group = 100.0 * float(np.mean(group_accs))
        worst_group = 100.0 * float(np.min(group_accs))

    return {
        "acc": acc,
        "avg_group_acc": avg_group,
        "worst_group_acc": worst_group,
        "group_details": group_items,
    }


def parse_penalty_solver_choices(text: str) -> List[str]:
    allowed = {
        "l2:lbfgs",
        "l2:liblinear",
        "l2:saga",
        "l1:liblinear",
        "l1:saga",
        "elasticnet:saga",
    }
    out = [x.strip() for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("No penalty:solver choices provided.")
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"Invalid penalty:solver choices: {bad}. Allowed: {sorted(allowed)}")
    return out


def parse_seeds(seed_start: int, num_seeds: int, seeds_csv: str):
    if seeds_csv:
        return [int(s) for s in seeds_csv.split(",") if s.strip()]
    return list(range(seed_start, seed_start + num_seeds))


def fit_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_val_name: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    d_test_name: np.ndarray,
    params: Dict,
    max_iter: int,
    seed: int,
):
    penalty = params["penalty"]
    solver = params["solver"]
    fit_intercept = bool(params["fit_intercept"])
    C = float(params["C"])
    l1_ratio = params.get("l1_ratio", None)

    clf = LogisticRegression(
        C=C,
        fit_intercept=fit_intercept,
        penalty=penalty,
        solver=solver,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        max_iter=max_iter,
        random_state=seed,
        n_jobs=1,
        multi_class="auto",
    )

    with threadpool_limits(limits=1):
        clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    val_metrics = compute_metrics(y_val, y_val_pred, d_val_name)
    test_metrics = compute_metrics(y_test, y_test_pred, d_test_name)
    return val_metrics, test_metrics


def objective_from_metrics(metrics: Dict, objective_name: str) -> float:
    if objective_name == "val_avg_group_acc":
        return float(metrics["avg_group_acc"])
    if objective_name == "val_worst_group_acc":
        return float(metrics["worst_group_acc"])
    if objective_name == "val_acc":
        return float(metrics["acc"])
    raise ValueError(f"Unknown objective: {objective_name}")


def write_csv_row(path: str, row: Dict, fieldnames: List[str]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def sample_random_params(rng: random.Random, choices: List[str], c_min: float, c_max: float, l1_min: float, l1_max: float):
    ps = rng.choice(choices)
    penalty, solver = ps.split(":")
    C = math.exp(rng.uniform(math.log(c_min), math.log(c_max)))
    fit_intercept = rng.choice([True, False])
    params = {
        "penalty_solver": ps,
        "penalty": penalty,
        "solver": solver,
        "C": C,
        "fit_intercept": fit_intercept,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = rng.uniform(l1_min, l1_max)
    else:
        params["l1_ratio"] = None
    return params


def main():
    p = argparse.ArgumentParser(description="CLIP + LR sweep for NICO++")
    p.add_argument("--txtdir", type=str, default=DEFAULT_TXTLIST_DIR)
    p.add_argument("--dataset", type=str, default="NICO")
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target", nargs="+", required=True, help="Held-out domains (e.g., autumn rock)")

    p.add_argument("--clip_model", type=str, default="ViT-B/32")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    p.add_argument("--study_name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--load_if_exists", action="store_true")

    p.add_argument("--c_min", type=float, default=1e-2)
    p.add_argument("--c_max", type=float, default=1e2)
    p.add_argument("--max_iter", type=int, default=5000)
    p.add_argument(
        "--penalty_solvers",
        type=str,
        default="l2:lbfgs,l2:liblinear,l2:saga,l1:liblinear,l1:saga,elasticnet:saga",
    )
    p.add_argument("--l1_ratio_min", type=float, default=0.05)
    p.add_argument("--l1_ratio_max", type=float, default=0.95)
    p.add_argument("--objective", type=str, default="val_avg_group_acc",
                   choices=["val_avg_group_acc", "val_worst_group_acc", "val_acc"])
    p.add_argument("--val_split_ratio", type=float, default=0.16)

    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--post_output_csv", type=str, default="")
    p.add_argument("--post_seeds", type=int, default=5)
    p.add_argument("--post_seed_start", type=int, default=59)
    p.add_argument("--post_seeds_list", type=str, default="")
    args = p.parse_args()

    targets = [d.lower() for d in args.target]
    unknown = [d for d in targets if d not in ALL_DOMAINS]
    if unknown:
        raise ValueError(f"Unknown target domains: {unknown}. Valid: {ALL_DOMAINS}")
    sources = [d for d in ALL_DOMAINS if d not in targets]
    if not sources:
        raise ValueError("Target domains cover all domains; no source domains remain.")

    seed_everything(args.seed)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    target_tag = "-".join(targets)
    run_dir = os.path.join(args.output_dir, f"clip_lr_target_{target_tag}")
    os.makedirs(run_dir, exist_ok=True)
    trial_csv = args.output_csv if args.output_csv else os.path.join(run_dir, "clip_lr_trials.csv")
    post_csv = args.post_output_csv if args.post_output_csv else os.path.join(run_dir, "clip_lr_best_post_seeds.csv")

    penalty_solver_choices = parse_penalty_solver_choices(args.penalty_solvers)

    print(
        f"CLIP+LR NICO++ | targets={targets} sources={sources} "
        f"sampler={args.sampler} trials={args.n_trials} objective={args.objective} device={device}",
        flush=True,
    )

    train_records, val_records, test_records, has_val = build_train_val_test_records(
        txtdir=args.txtdir,
        dataset=args.dataset,
        image_root=args.image_root,
        targets=targets,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
    )
    if not has_val:
        print(f"Val split: {int(args.val_split_ratio * 100)}% split from source train (no *_val.txt).", flush=True)
    print(
        f"Record counts | train={len(train_records)} val={len(val_records)} test={len(test_records)}",
        flush=True,
    )

    model, preprocess = load_clip(args.clip_model, device)
    print("Extracting CLIP features once per split...", flush=True)
    X_train, y_train, _, _ = extract_features(model, preprocess, train_records, args.batch_size, args.num_workers, device)
    X_val, y_val, _, d_val_name = extract_features(model, preprocess, val_records, args.batch_size, args.num_workers, device)
    X_test, y_test, _, d_test_name = extract_features(model, preprocess, test_records, args.batch_size, args.num_workers, device)
    print(
        f"Feature shapes | train={X_train.shape} val={X_val.shape} test={X_test.shape}",
        flush=True,
    )

    fieldnames = [
        "run_id", "trial", "status", "sampler", "objective_name", "objective",
        "C", "fit_intercept", "penalty", "solver", "l1_ratio", "max_iter",
        "val_acc", "val_avg_group_acc", "val_worst_group_acc",
        "test_acc", "test_avg_group_acc", "test_worst_group_acc",
        "val_group_details_json", "test_group_details_json", "seconds", "error",
    ]

    def run_trial_with_params(params: Dict, trial_id: int, fit_seed: int):
        t0 = time.time()
        row = {
            "run_id": target_tag,
            "trial": trial_id,
            "status": "ok",
            "sampler": args.sampler,
            "objective_name": args.objective,
            "objective": "",
            "C": params["C"],
            "fit_intercept": params["fit_intercept"],
            "penalty": params["penalty"],
            "solver": params["solver"],
            "l1_ratio": params.get("l1_ratio", None),
            "max_iter": args.max_iter,
            "val_acc": "",
            "val_avg_group_acc": "",
            "val_worst_group_acc": "",
            "test_acc": "",
            "test_avg_group_acc": "",
            "test_worst_group_acc": "",
            "val_group_details_json": "",
            "test_group_details_json": "",
            "seconds": "",
            "error": "",
        }
        try:
            val_metrics, test_metrics = fit_and_eval(
                X_train, y_train,
                X_val, y_val, d_val_name,
                X_test, y_test, d_test_name,
                params=params,
                max_iter=args.max_iter,
                seed=fit_seed,
            )
            objective = objective_from_metrics(val_metrics, args.objective)
            row["objective"] = objective
            row["val_acc"] = val_metrics["acc"]
            row["val_avg_group_acc"] = val_metrics["avg_group_acc"]
            row["val_worst_group_acc"] = val_metrics["worst_group_acc"]
            row["test_acc"] = test_metrics["acc"]
            row["test_avg_group_acc"] = test_metrics["avg_group_acc"]
            row["test_worst_group_acc"] = test_metrics["worst_group_acc"]
            row["val_group_details_json"] = json.dumps(val_metrics["group_details"])
            row["test_group_details_json"] = json.dumps(test_metrics["group_details"])
        except Exception as exc:
            objective = -1e18
            row["status"] = "fail"
            row["objective"] = objective
            row["error"] = repr(exc)
        row["seconds"] = time.time() - t0
        write_csv_row(trial_csv, row, fieldnames)
        return objective, row

    best_objective = -1e18
    best_params: Optional[Dict] = None
    best_trial_id = -1

    if args.sampler == "tpe" and optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    elif optuna is not None:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = None
        if args.sampler == "tpe":
            print("Optuna unavailable; falling back to random sampling.", flush=True)

    if sampler is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=args.load_if_exists,
        )

        def objective_fn(trial):
            ps = trial.suggest_categorical("penalty_solver", penalty_solver_choices)
            penalty, solver = ps.split(":")
            params = {
                "penalty": penalty,
                "solver": solver,
                "C": trial.suggest_float("C", args.c_min, args.c_max, log=True),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            }
            if penalty == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", args.l1_ratio_min, args.l1_ratio_max)
            else:
                params["l1_ratio"] = None
            value, _ = run_trial_with_params(params, trial.number, args.seed + trial.number)
            return float(value)

        study.optimize(objective_fn, n_trials=args.n_trials, timeout=args.timeout)
        best_objective = float(study.best_value)
        best_trial_id = int(study.best_trial.number)
        best_params = dict(study.best_params)
        if "penalty_solver" in best_params:
            penalty, solver = str(best_params["penalty_solver"]).split(":")
            best_params["penalty"] = penalty
            best_params["solver"] = solver
            best_params.pop("penalty_solver", None)
        if "l1_ratio" not in best_params:
            best_params["l1_ratio"] = None
    else:
        rng = random.Random(args.seed)
        t0_all = time.time()
        for trial_id in range(args.n_trials):
            if args.timeout is not None and (time.time() - t0_all) >= args.timeout:
                print(f"Timeout reached after {trial_id} random trials.", flush=True)
                break
            params = sample_random_params(
                rng=rng,
                choices=penalty_solver_choices,
                c_min=args.c_min,
                c_max=args.c_max,
                l1_min=args.l1_ratio_min,
                l1_max=args.l1_ratio_max,
            )
            value, _ = run_trial_with_params(params, trial_id, args.seed + trial_id)
            if value > best_objective:
                best_objective = value
                best_params = dict(params)
                best_trial_id = trial_id

    if best_params is None:
        raise RuntimeError("No successful trial completed.")

    print(f"Best objective ({args.objective}): {best_objective:.6f}", flush=True)
    print(f"Best trial: {best_trial_id}", flush=True)
    print(f"Best params: {best_params}", flush=True)

    best_json = {
        "run_id": target_tag,
        "objective_name": args.objective,
        "best_objective": best_objective,
        "best_trial": best_trial_id,
        "best_params": best_params,
        "n_trials_requested": args.n_trials,
    }
    with open(os.path.join(run_dir, "clip_lr_best.json"), "w") as f:
        json.dump(best_json, f, indent=2)

    seeds = parse_seeds(args.post_seed_start, args.post_seeds, args.post_seeds_list)
    post_fields = [
        "run_id", "seed", "objective_name", "objective",
        "C", "fit_intercept", "penalty", "solver", "l1_ratio", "max_iter",
        "val_acc", "val_avg_group_acc", "val_worst_group_acc",
        "test_acc", "test_avg_group_acc", "test_worst_group_acc",
        "val_group_details_json", "test_group_details_json", "seconds", "error",
    ]
    print(f"Re-running best hyperparameters for {len(seeds)} seeds...", flush=True)
    for seed in seeds:
        t0 = time.time()
        row = {
            "run_id": target_tag,
            "seed": seed,
            "objective_name": args.objective,
            "objective": "",
            "C": best_params["C"],
            "fit_intercept": best_params["fit_intercept"],
            "penalty": best_params["penalty"],
            "solver": best_params["solver"],
            "l1_ratio": best_params.get("l1_ratio", None),
            "max_iter": args.max_iter,
            "val_acc": "",
            "val_avg_group_acc": "",
            "val_worst_group_acc": "",
            "test_acc": "",
            "test_avg_group_acc": "",
            "test_worst_group_acc": "",
            "val_group_details_json": "",
            "test_group_details_json": "",
            "seconds": "",
            "error": "",
        }
        try:
            val_metrics, test_metrics = fit_and_eval(
                X_train, y_train,
                X_val, y_val, d_val_name,
                X_test, y_test, d_test_name,
                params=best_params,
                max_iter=args.max_iter,
                seed=seed,
            )
            objective = objective_from_metrics(val_metrics, args.objective)
            row["objective"] = objective
            row["val_acc"] = val_metrics["acc"]
            row["val_avg_group_acc"] = val_metrics["avg_group_acc"]
            row["val_worst_group_acc"] = val_metrics["worst_group_acc"]
            row["test_acc"] = test_metrics["acc"]
            row["test_avg_group_acc"] = test_metrics["avg_group_acc"]
            row["test_worst_group_acc"] = test_metrics["worst_group_acc"]
            row["val_group_details_json"] = json.dumps(val_metrics["group_details"])
            row["test_group_details_json"] = json.dumps(test_metrics["group_details"])
            print(
                f"[POST seed={seed}] objective={objective:.6f} "
                f"val_acc={val_metrics['acc']:.4f} test_acc={test_metrics['acc']:.4f}",
                flush=True,
            )
        except Exception as exc:
            row["objective"] = -1e18
            row["error"] = repr(exc)
            print(f"[POST seed={seed}] failed: {exc}", flush=True)
        row["seconds"] = time.time() - t0
        write_csv_row(post_csv, row, post_fields)

    print(f"Saved trials CSV: {trial_csv}", flush=True)
    print(f"Saved post-seeds CSV: {post_csv}", flush=True)


if __name__ == "__main__":
    main()

