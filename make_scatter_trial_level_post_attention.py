import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_log_text_from_source(path: str) -> str:
    src = Path(path).read_text()
    m = re.search(r'LOG_TEXT\s*=\s*r?"""(.*)"""', src, re.S)
    if not m:
        raise ValueError(f"Could not find LOG_TEXT in {path}")
    return m.group(1)


def parse_attention_epochs(log_text: str):
    """
    Build trial -> attention_epoch map from both formats:
      - === trial_36 start params={...'attention_epoch': 11,...} ===
      - [TRIAL 0] start params={...'attention_epoch': 3,...}
    """
    out = {}
    p1 = re.compile(
        r"===\s*trial_(\d+)\s+start params=\{.*?'attention_epoch':\s*(\d+).*?\}\s*===",
        re.IGNORECASE,
    )
    p2 = re.compile(
        r"\[TRIAL\s+(\d+)\]\s+start params=\{.*?'attention_epoch':\s*(\d+).*?\}",
        re.IGNORECASE,
    )
    for m in p1.finditer(log_text):
        out[int(m.group(1))] = int(m.group(2))
    for m in p2.finditer(log_text):
        out[int(m.group(1))] = int(m.group(2))
    return out


def parse_epoch_rows(log_text: str) -> pd.DataFrame:
    """
    Parse rows from both formats:
      - [trial_27][Epoch 10] ...
      - [TRIAL 1][Epoch 10] ...
    """
    pattern = re.compile(
        r"\[(?:trial_(?P<trial_u>\d+)|trial\s+(?P<trial_s>\d+))\]"
        r"\[Epoch\s+(?P<epoch>\d+)\]\s+val_acc=(?P<val_acc>[0-9.]+).*?"
        r"ig_fwd_kl=(?P<ig_fwd_kl>[0-9.]+).*?"
        r"test=\{.*?'overall':\s*(?P<test_overall>[0-9.]+)\}",
        re.IGNORECASE,
    )
    rows = []
    for m in pattern.finditer(log_text):
        trial = int(m.group("trial_u") or m.group("trial_s"))
        epoch = int(m.group("epoch"))
        val_acc = float(m.group("val_acc"))
        ig_fwd_kl = float(m.group("ig_fwd_kl"))
        test_overall = float(m.group("test_overall"))
        rows.append((trial, epoch, val_acc, ig_fwd_kl, test_overall))
    return pd.DataFrame(
        rows,
        columns=["trial", "epoch", "val_acc", "ig_fwd_kl", "test_overall"],
    )


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pick_trial_points(df_post: pd.DataFrame, beta: float) -> pd.DataFrame:
    rows = []
    for trial_id, grp in df_post.groupby("trial"):
        grp = grp.sort_values("epoch").copy()
        grp["log_optim_num"] = np.log(np.maximum(grp["val_acc"].to_numpy(), 1e-12)) - beta * grp["ig_fwd_kl"].to_numpy()
        grp["optim_num"] = np.exp(grp["log_optim_num"])

        idx_val = grp["val_acc"].idxmax()
        idx_opt = grp["log_optim_num"].idxmax()

        r_val = grp.loc[idx_val]
        r_opt = grp.loc[idx_opt]
        rows.append(
            {
                "trial": int(trial_id),
                "attention_epoch": int(grp["attention_epoch"].iloc[0]),
                "n_post_epochs": int(len(grp)),
                "best_val_epoch": int(r_val["epoch"]),
                "best_val_acc": float(r_val["val_acc"]),
                "test_at_best_val": float(r_val["test_overall"]),
                "best_optim_epoch": int(r_opt["epoch"]),
                "best_optim_num": float(r_opt["optim_num"]),
                "best_log_optim_num": float(r_opt["log_optim_num"]),
                "test_at_best_optim": float(r_opt["test_overall"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("trial").reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Trial-level post-attention scatter: one point per trial."
    )
    ap.add_argument("--source_file", type=str, default="make_scatter.py")
    ap.add_argument(
        "--beta",
        type=float,
        default=None,
        help="If set, use this fixed beta. If omitted, sweep beta_min..beta_max and pick best.",
    )
    ap.add_argument("--beta_min", type=int, default=10)
    ap.add_argument("--beta_max", type=int, default=50)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument(
        "--save_plots",
        action="store_true",
        help="Also save PNG plots to output_dir. Default behavior is interactive pop-up only.",
    )
    ap.add_argument(
        "--include_attention_epoch",
        action="store_true",
        help="If set, keep epoch >= attention_epoch. Default is strict post only: epoch > attention_epoch.",
    )
    args = ap.parse_args()

    log_text = load_log_text_from_source(args.source_file)
    attention_map = parse_attention_epochs(log_text)
    df = parse_epoch_rows(log_text)

    if df.empty:
        raise ValueError("No epoch rows parsed from LOG_TEXT.")

    df["attention_epoch"] = df["trial"].map(attention_map)
    df = df[df["attention_epoch"].notna()].copy()

    if args.include_attention_epoch:
        df_post = df[df["epoch"] >= df["attention_epoch"]].copy()
    else:
        df_post = df[df["epoch"] > df["attention_epoch"]].copy()

    if df_post.empty:
        raise ValueError("No rows remain after post-attention filtering.")

    if args.beta is None:
        if args.beta_min > args.beta_max:
            raise ValueError("beta_min must be <= beta_max")
        beta_rows = []
        for b in range(args.beta_min, args.beta_max + 1):
            trial_df_b = pick_trial_points(df_post, beta=float(b))
            r_opt_b = safe_pearson(
                trial_df_b["best_optim_num"].to_numpy(),
                trial_df_b["test_at_best_optim"].to_numpy(),
            )
            beta_rows.append((b, r_opt_b))
        beta_df = pd.DataFrame(beta_rows, columns=["beta", "r_opt"])
        beta_df_non_nan = beta_df.dropna(subset=["r_opt"])
        if beta_df_non_nan.empty:
            raise ValueError("No valid beta produced a finite correlation.")
        best_beta = int(beta_df_non_nan.loc[beta_df_non_nan["r_opt"].idxmax(), "beta"])
        print(f"Best beta in [{args.beta_min}, {args.beta_max}] = {best_beta}")
        print("\nTop beta values:")
        print(beta_df.sort_values("r_opt", ascending=False).head(10).to_string(index=False))
    else:
        best_beta = float(args.beta)

    trial_df = pick_trial_points(df_post, beta=float(best_beta))
    if trial_df.empty:
        raise ValueError("No trial-level points computed.")

    r_val = safe_pearson(trial_df["best_val_acc"].to_numpy(), trial_df["test_at_best_val"].to_numpy())
    r_opt = safe_pearson(trial_df["best_optim_num"].to_numpy(), trial_df["test_at_best_optim"].to_numpy())

    print(f"Trials used: {len(trial_df)}")
    print(
        f"Filter: epoch {'>=' if args.include_attention_epoch else '>'} attention_epoch, "
        f"beta={best_beta:g}"
    )
    print(f"Pearson (best val_acc per trial vs test_at_best_val): {r_val:.6f}")
    print(f"Pearson (best optim_num per trial vs test_at_best_optim): {r_opt:.6f}")
    print("Note: log-space optim metric shown as natural log (best_log_optim_num).")
    print("\nPer-trial selected points:")
    print(
        trial_df[
            [
                "trial",
                "attention_epoch",
                "n_post_epochs",
                "best_val_epoch",
                "best_val_acc",
                "test_at_best_val",
                "best_optim_epoch",
                "best_log_optim_num",
                "test_at_best_optim",
            ]
        ].to_string(index=False)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "trial_level_post_attention_points.csv"
    trial_df.to_csv(csv_path, index=False)

    # Plot 1: val-based selection (one dot per trial)
    plt.figure(figsize=(7, 5))
    plt.scatter(trial_df["best_val_acc"], trial_df["test_at_best_val"], alpha=0.85)
    plt.xlabel("best val_acc (post-attention, one point per trial)")
    plt.ylabel("test_overall at that epoch (%)")
    plt.title(f"Val-selected epoch per trial | Pearson r={r_val:.4f}")
    plt.tight_layout()
    if args.save_plots:
        val_png = out_dir / "trial_level_valacc_vs_test.png"
        plt.savefig(val_png, dpi=180)
        print(f"Saved: {val_png}")
    plt.show()

    # Plot 2: optim-based selection (one dot per trial)
    plt.figure(figsize=(7, 5))
    plt.scatter(trial_df["best_optim_num"], trial_df["test_at_best_optim"], alpha=0.85)
    plt.xscale("log")
    plt.xlabel(f"best optim_num per trial (beta={best_beta:g}, post-attention)")
    plt.ylabel("test_overall at that epoch (%)")
    plt.title(f"Optim-selected epoch per trial | Pearson r={r_opt:.4f}")
    plt.tight_layout()
    if args.save_plots:
        opt_png = out_dir / "trial_level_optimnum_vs_test.png"
        plt.savefig(opt_png, dpi=180)
        print(f"Saved: {opt_png}")
    plt.show()

    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
