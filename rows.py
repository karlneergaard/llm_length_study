#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rows.py

Builds per-model master CSVs and produces plots from:
  runs/{phi3_*|phi4_*|deepseek7b_*}/detailed_rows.csv

"""

from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

FINAL_JSONL = DATA_DIR / "boolq_final.jsonl"

# PHI-3 conditions (no memory)
PHI3_FOLDERS = {
    "baseline":       RUNS_DIR / "phi3_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "phi3_meta" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "phi3_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "phi3_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "phi3_misleading" / "detailed_rows.csv",
}

# PHI-4 conditions (no memory)
PHI4_FOLDERS = {
    "baseline":       RUNS_DIR / "phi4_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "phi4_meta" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "phi4_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "phi4_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "phi4_misleading" / "detailed_rows.csv",
}

# DeepSeek-7B conditions (no memory)
DEEPSEEK_FOLDERS = {
    "baseline":       RUNS_DIR / "deepseek7b_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "deepseek7b_meta" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "deepseek7b_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "deepseek7b_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "deepseek7b_misleading" / "detailed_rows.csv",
}

QWEN_FOLDERS = {
    "baseline":       RUNS_DIR / "qwen7b_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "qwen7b_meta" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "qwen7b_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "qwen7b_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "qwen7b_misleading" / "detailed_rows.csv",
}

OUT_PHI3_MASTER = RUNS_DIR / "phi3_rows.csv"
OUT_PHI4_MASTER = RUNS_DIR / "phi4_rows.csv"
OUT_DEEPSEEK_MASTER = RUNS_DIR / "deepseek7b_rows.csv"
OUT_QWEN_MASTER = RUNS_DIR / "qwen7b_rows.csv"

OUT_PHI3_PLOT = RUNS_DIR / "phi3_acc_combined.png"
OUT_PHI4_PLOT = RUNS_DIR / "phi4_acc_combined.png"
OUT_DEEPSEEK_PLOT = RUNS_DIR / "deepseek7b_acc_combined.png"
OUT_QWEN_PLOT = RUNS_DIR / "qwen7b_acc_combined.png"

OUT_PHI3_INCORRECT = RUNS_DIR / "phi3_incorrect_by_condition.png"
OUT_PHI4_INCORRECT = RUNS_DIR / "phi4_incorrect_by_condition.png"
OUT_DEEPSEEK_INCORRECT = RUNS_DIR / "deepseek7b_incorrect_by_condition.png"
OUT_QWEN_INCORRECT = RUNS_DIR / "qwen7b_incorrect_by_condition.png"

# ---------- Helpers ----------
def load_final_ids(path: Path) -> set:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if "id" in obj:
                    ids.add(str(obj["id"]))
    return ids

def truthy(x) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

def load_detail_csv(path: Path, condition: str, keep_ids: set) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    key_col = "boolq" if "boolq" in df.columns else "boolq_id"
    df["boolq_id"] = df[key_col].astype(str)
    df = df[df["boolq_id"].isin(keep_ids)].copy()
    df["condition"] = condition
    return df

def compute_acc_type(row) -> str:
    pred_label = str(row.get("pred_label", "")).strip().lower()
    if pred_label == "idk" or truthy(row.get("is_idk", False)):
        return "IDK"
    ac = str(row.get("answered_correct", "")).strip().upper()
    if ac == "TRUE":
        return "Correct"
    elif ac == "FALSE":
        return "Incorrect"
    return "NC"

def percent_table(grouped_counts: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    df = grouped_counts.copy()
    totals = df.groupby(group_cols, as_index=False)["n"].sum().rename(columns={"n": "total"})
    df = df.merge(totals, on=group_cols, how="left")
    df["percent"] = (df["n"] / df["total"]).fillna(0.0)
    return df

def build_master(folders: dict, keep_ids: set, label: str, out_csv: Path) -> pd.DataFrame:
    parts = []
    for cond, path in folders.items():
        df = load_detail_csv(path, cond, keep_ids)
        if not df.empty:
            parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    merged["acc_type"] = merged.apply(compute_acc_type, axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[MASTER] {label}: wrote {len(merged)} rows -> {out_csv}")
    return merged

# ---------- Plotters ----------
def plot_by_length(merged: pd.DataFrame, label: str, out_path: Path):
    if merged.empty:
        return
    keep_conds = ["meta", "semantic", "underspecified", "misleading"]
    len_df = merged[merged["condition"].isin(keep_conds)].copy()
    len_df = len_df.dropna(subset=["L"])
    len_df["L"] = len_df["L"].astype(int)

    grouped = len_df.groupby(["condition", "L", "acc_type"]).size().reset_index(name="n")
    pct = percent_table(grouped, ["condition", "L"])

    color_map = {"Incorrect": "tab:red", "Correct": "tab:green", "IDK": "tab:orange"}
    markers = {"meta": "D", "semantic": "o", "underspecified": "s", "misleading": "X"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for acc, color in color_map.items():
        for cond, marker in markers.items():
            sub = pct[(pct["acc_type"] == acc) & (pct["condition"] == cond)]
            if not sub.empty:
                ax.plot(sub["L"], sub["percent"], marker=marker, color=color, label=f"{cond}-{acc}")

    ax.set_xlabel("Length (L)")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Wrote {out_path}")

def plot_incorrect_by_condition(merged: pd.DataFrame, label: str, out_path: Path):
    if merged.empty:
        return
    len_df = merged[merged["condition"] != "baseline"].copy()
    len_df = len_df.dropna(subset=["L"])
    len_df["L"] = len_df["L"].astype(int)

    grouped = len_df.groupby(["condition", "L", "acc_type"]).size().reset_index(name="n")
    pct = percent_table(grouped, ["condition", "L"])
    inc = pct[pct["acc_type"] == "Incorrect"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"meta": "blue", "semantic": "green", "underspecified": "red", "misleading": "purple"}
    for cond, color in colors.items():
        sub = inc[inc["condition"] == cond]
        if not sub.empty:
            ax.plot(sub["L"], sub["percent"], marker="o", color=color, label=cond)

    ax.set_xlabel("Length (L)")
    ax.set_ylabel("Percent Incorrect")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Wrote {out_path}")

# ---------- Main ----------
def main():
    keep_ids = load_final_ids(FINAL_JSONL)
    phi3 = build_master(PHI3_FOLDERS, keep_ids, "phi3", OUT_PHI3_MASTER)
    phi4 = build_master(PHI4_FOLDERS, keep_ids, "phi4", OUT_PHI4_MASTER)
    deepseek = build_master(DEEPSEEK_FOLDERS, keep_ids, "deepseek7b", OUT_DEEPSEEK_MASTER)
    qwen = build_master(QWEN_FOLDERS, keep_ids, "qwen7b", OUT_QWEN_MASTER)

    plot_by_length(phi3, "phi3", OUT_PHI3_PLOT)
    plot_by_length(phi4, "phi4", OUT_PHI4_PLOT)
    plot_by_length(deepseek, "deepseek7b", OUT_DEEPSEEK_PLOT)
    plot_by_length(qwen, "qwen7b", OUT_QWEN_PLOT)

    plot_incorrect_by_condition(phi3, "phi3", OUT_PHI3_INCORRECT)
    plot_incorrect_by_condition(phi4, "phi4", OUT_PHI4_INCORRECT)
    plot_incorrect_by_condition(deepseek, "deepseek7b", OUT_DEEPSEEK_INCORRECT)
    plot_incorrect_by_condition(qwen, "qwen7b", OUT_QWEN_INCORRECT)

if __name__ == "__main__":
    main()
