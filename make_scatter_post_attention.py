import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Sweep beta over this range and choose the best pooled Pearson on filtered rows.
BETA_MIN = 1
BETA_MAX = 50
BETA_REPORT = 10

# Reads LOG_TEXT from make_scatter.py so you do not need to paste twice.
SOURCE_SCATTER_FILE = "make_scatter.py"


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


def parse_epoch_rows(log_text: str):
    """
    Parse epoch rows from both formats:
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


def main():
    log_text = load_log_text_from_source(SOURCE_SCATTER_FILE)
    attention_epochs = parse_attention_epochs(log_text)
    df = parse_epoch_rows(log_text)

    if df.empty:
        raise ValueError("No epoch rows parsed. Check LOG_TEXT content and format.")

    df["attention_epoch"] = df["trial"].map(attention_epochs)
    total_rows = len(df)
    no_attn_rows = int(df["attention_epoch"].isna().sum())
    if no_attn_rows > 0:
        print(f"Rows missing attention_epoch mapping: {no_attn_rows}")

    # "After attention epoch" = strictly epoch > attention_epoch.
    df = df[df["attention_epoch"].notna()].copy()
    df_post = df[df["epoch"] > df["attention_epoch"]].copy()

    if df_post.empty:
        raise ValueError("No rows remain after filtering epoch > attention_epoch.")

    print(
        f"Parsed rows: {total_rows}, with attention mapping: {len(df)}, "
        f"post-attention rows: {len(df_post)}"
    )
    for trial_id, grp in df_post.groupby("trial"):
        epochs = sorted(grp["epoch"].tolist())
        print(
            f"trial_{int(trial_id)} post-attn epochs: {len(epochs)} "
            f"(min={epochs[0]}, max={epochs[-1]}), attention_epoch={int(grp['attention_epoch'].iloc[0])}"
        )

    r_val = float(np.corrcoef(df_post["val_acc"].to_numpy(), df_post["test_overall"].to_numpy())[0, 1])

    beta_results = []
    for beta in range(BETA_MIN, BETA_MAX + 1):
        optim_num = df_post["val_acc"].to_numpy() * np.exp(-beta * df_post["ig_fwd_kl"].to_numpy())
        if np.std(optim_num) == 0 or np.std(df_post["test_overall"].to_numpy()) == 0:
            r_optim = float("nan")
        else:
            r_optim = float(np.corrcoef(optim_num, df_post["test_overall"].to_numpy())[0, 1])
        beta_results.append((beta, r_optim))

    beta_df = pd.DataFrame(beta_results, columns=["beta", "r_optim"])
    best_row = beta_df.loc[beta_df["r_optim"].idxmax()]
    best_beta = int(best_row["beta"])
    best_r = float(best_row["r_optim"])

    df_post["optim_num"] = df_post["val_acc"] * np.exp(-best_beta * df_post["ig_fwd_kl"])

    print(f"Pooled Pearson r(val_acc, test_overall) [post-attn only] = {r_val:.6f}")
    print(f"Best beta in [{BETA_MIN}, {BETA_MAX}] [post-attn only]: beta={best_beta}, r={best_r:.6f}")
    print("\nTop beta values (post-attn only):")
    print(beta_df.sort_values("r_optim", ascending=False).head(10).to_string(index=False))

    # Also report best rows by optim_num at a fixed beta (requested: beta=10).
    df_beta = df_post.copy()
    df_beta["optim_num_beta"] = df_beta["val_acc"] * np.exp(-BETA_REPORT * df_beta["ig_fwd_kl"])
    top_beta = df_beta.sort_values("optim_num_beta", ascending=False).head(15)
    print(f"\nTop 15 rows by optim_num at beta={BETA_REPORT} (post-attn only):")
    print(
        top_beta[
            ["trial", "epoch", "attention_epoch", "val_acc", "ig_fwd_kl", "optim_num_beta", "test_overall"]
        ].to_string(index=False)
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(df_post["val_acc"], df_post["test_overall"], alpha=0.8)
    plt.xlabel("val_acc (post-attention epochs only)")
    plt.ylabel("test_overall (%)")
    plt.title(f"val_acc vs test_overall (post-attn) | Pearson r={r_val:.4f}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(df_post["optim_num"], df_post["test_overall"], alpha=0.8)
    plt.xscale("log")
    plt.xlabel(f"optim_num = val_acc * exp(-{best_beta} * ig_fwd_kl) (post-attn)")
    plt.ylabel("test_overall (%)")
    plt.title(f"optim_num vs test_overall (post-attn) | Pearson r={best_r:.4f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
