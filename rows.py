#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rows.py

Builds per-model master CSVs and produces plots from:
  runs/{phi3_*|phi4_*}/detailed_rows.csv

Filters to ids in data/boolq_final.jsonl (1002 items), derives acc_type,
writes master CSVs, and produces three plots total:

1) runs/phi3_acc_combined.png  (phi3: acc types by length & condition; color = acc_type, style = condition)
2) runs/phi4_acc_combined.png  (phi4: same)
3) runs/phi3_incorrect_by_condition.png (phi3: % Incorrect across lengths; one colored line per condition)
4) runs/phi4_incorrect_by_condition.png (phi4: same)

Notes:
- Memory condition included (same structure as semantic).
- Misleading may have only some lengths; plotting tolerates gaps.
"""

from pathlib import Path
import json
import pandas as pd
import math

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

PHI3_FOLDERS = {
    "baseline":       RUNS_DIR / "phi3_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "phi3_meta" / "detailed_rows.csv",
    "memory":         RUNS_DIR / "phi3_memory" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "phi3_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "phi3_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "phi3_misleading" / "detailed_rows.csv",
}

PHI4_FOLDERS = {
    "baseline":       RUNS_DIR / "phi4_baseline" / "detailed_rows.csv",
    "meta":           RUNS_DIR / "phi4_meta" / "detailed_rows.csv",
    "memory":         RUNS_DIR / "phi4_memory" / "detailed_rows.csv",
    "semantic":       RUNS_DIR / "phi4_semantic" / "detailed_rows.csv",
    "underspecified": RUNS_DIR / "phi4_underspecified" / "detailed_rows.csv",
    "misleading":     RUNS_DIR / "phi4_misleading" / "detailed_rows.csv",
}

OUT_PHI3_MASTER = RUNS_DIR / "phi3_rows.csv"
OUT_PHI4_MASTER = RUNS_DIR / "phi4_rows.csv"

OUT_PHI3_PLOT = RUNS_DIR / "phi3_acc_combined.png"
OUT_PHI4_PLOT = RUNS_DIR / "phi4_acc_combined.png"

OUT_PHI3_INCORRECT = RUNS_DIR / "phi3_incorrect_by_condition.png"
OUT_PHI4_INCORRECT = RUNS_DIR / "phi4_incorrect_by_condition.png"


# ---------- Helpers ----------
def load_final_ids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            _id = obj.get("id")
            if _id is not None:
                ids.add(str(_id))
    if not ids:
        raise ValueError(f"No 'id' values found in {path}")
    return ids

def truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}

def load_detail_csv(path: Path, condition: str, keep_ids: set) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    key_col = None
    for cand in ("boolq", "boolq_id"):
        if cand in df.columns:
            key_col = cand
            break
    if key_col is None:
        raise ValueError(f"{path} must have 'boolq' or 'boolq_id'. Found: {list(df.columns)}")
    df["boolq_id"] = df[key_col].astype(str)
    df = df[df["boolq_id"].isin(keep_ids)].copy()
    df["condition"] = condition
    return df

def compute_acc_type(row) -> str:
    pred_label = str(row.get("pred_label", row.get("pred_lable", ""))).strip().lower()
    if truthy(row.get("is_idk")) or pred_label == "idk":
        return "IDK"
    if pred_label == "other":
        return "NC"
    ac = row.get("answered_correct")
    if ac is not None:
        s = str(ac).strip().upper()
        if s in {"TRUE", "FALSE", "IDK", "NC"}:
            return s
    return "NA"

ACC_MAP = {"FALSE": "Incorrect", "TRUE": "Correct", "IDK": "IDK", "NC": "NC", "NA": "NA"}
ACC_LEVELS = ["Incorrect", "Correct", "IDK", "NC"]

def percent_table(grouped_counts: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    df = grouped_counts.copy()
    if df.empty:
        df = pd.DataFrame(columns=group_cols + ["acc_type", "n"])
    all_groups = df[group_cols].drop_duplicates()
    scaffold = (
        all_groups.assign(key=1)
        .merge(pd.DataFrame({"acc_type": ACC_LEVELS, "key": 1}), on="key")
        .drop(columns="key")
    )
    df = scaffold.merge(df, on=group_cols + ["acc_type"], how="left").fillna({"n": 0})
    totals = df.groupby(group_cols, as_index=False)["n"].sum().rename(columns={"n": "total"})
    df = df.merge(totals, on=group_cols, how="left")
    df["percent"] = (df["n"] / df["total"]).where(df["total"] > 0, 0.0)
    df["acc_type"] = pd.Categorical(df["acc_type"], categories=ACC_LEVELS, ordered=True)
    return df.sort_values(group_cols + ["acc_type"]).reset_index(drop=True)

def build_master(folders: dict, keep_ids: set, label: str, out_csv: Path) -> pd.DataFrame:
    parts = []
    for cond, path in folders.items():
        df = load_detail_csv(path, cond, keep_ids)
        if not df.empty:
            parts.append(df)
    if not parts:
        raise RuntimeError(f"No data found for {label}.")
    merged = pd.concat(parts, ignore_index=True)

    for col in ("answered_correct", "pred_label", "pred_lable"):
        if col in merged.columns:
            merged[col] = merged[col].astype(str)

    merged["_acc_upper"] = merged.apply(compute_acc_type, axis=1)
    merged["acc_type"] = merged["_acc_upper"].map(ACC_MAP).fillna("NA")

    if "L" not in merged.columns:
        merged["L"] = pd.NA
    else:
        with pd.option_context("mode.chained_assignment", None):
            try:
                merged["L"] = merged["L"].astype("Int64")
            except Exception:
                pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"[MASTER] {label}: wrote {len(merged)} rows -> {out_csv}")
    return merged


# ---------- Plotters (combined acc_type view stays) ----------
def plot_by_length(merged: pd.DataFrame, label: str, out_path: Path):
    """
    Baseline as a dot at L=1 (no baseline legend entry).
    Non-baseline conditions as lines across available L.

    Legend 1 (left): colors = acc types
    Legend 2 (right): marker shapes = conditions (no baseline)
    """
    if merged.empty:
        print(f"[PLOT] {label}: no data.")
        return

    # Baseline selection (prefer condition==baseline; fall back to L==1)
    base_mask = (merged["condition"] == "baseline") | (merged.get("L").fillna(-1).astype("Int64") == 1)
    baseline_df = merged.loc[base_mask].copy()

    counts_baseline = baseline_df.groupby(["acc_type"], dropna=False).size().reset_index(name="n")
    counts_baseline["condition"] = "baseline"
    pct_baseline = percent_table(counts_baseline, ["condition"])

    # Non-baseline by length
    keep_conds = ["meta", "memory", "semantic", "underspecified", "misleading"]
    len_df = merged[merged["condition"].isin(keep_conds)].copy()
    len_df = len_df[~len_df["L"].isna()]
    try:
        len_df["L"] = len_df["L"].astype(int)
    except Exception:
        pass
    counts_len = (
        len_df.groupby(["condition", "L", "acc_type"], dropna=False)
              .size().reset_index(name="n")
    )
    pct_len = percent_table(counts_len, ["condition", "L"])

    # Build plotting frame
    base = pct_baseline.loc[pct_baseline["condition"] == "baseline", ["acc_type", "percent"]].copy()
    base["condition"] = "baseline"
    base["L"] = 1
    plot_df = pd.concat([
        base[["condition", "L", "acc_type", "percent"]],
        pct_len[["condition", "L", "acc_type", "percent"]],
    ], ignore_index=True)

    # Colors encode acc_type (unchanged)
    color_map = {"Incorrect": "tab:red", "Correct": "tab:green", "IDK": "tab:orange", "NC": "tab:blue"}

    # NEW: markers encode condition
    cond_markers = {
        "meta":           "D",   # diamond
        "memory":         "*",   # star
        "semantic":       "o",   # circle
        "underspecified": "s",   # square
        "misleading":     "X",   # X
    }

    all_L = sorted(set(plot_df["L"].dropna().astype(int).tolist()) | {1})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline dots at L=1 only (NO baseline entry in condition legend)
    base_sub = plot_df[(plot_df["condition"] == "baseline") & (plot_df["L"] == 1)]
    for acc, color in color_map.items():
        row = base_sub[base_sub["acc_type"] == acc]
        if not row.empty:
            ax.scatter([1], [float(row["percent"].iloc[0])], color=color, s=55, zorder=3)

    # Condition lines: solid; markers vary by condition
    for acc, color in color_map.items():
        for cond, marker in cond_markers.items():
            sub = plot_df[(plot_df["condition"] == cond) & (plot_df["acc_type"] == acc)].copy()
            if not sub.empty:
                sub = sub.sort_values("L")
                ax.plot(
                    sub["L"].astype(int),
                    sub["percent"].astype(float),
                    marker=marker,
                    linestyle="-",
                    color=color,
                    linewidth=2.2,
                    markersize=6,
                )

    ax.set_xlabel("Length (L)")
    ax.set_ylabel("Percent")
    ax.set_xticks(all_L)
    ax.set_ylim(0, 1)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{label}: Percent by acc_type across length and condition")

    # Legend A: acc types (color)
    from matplotlib.lines import Line2D
    leg_acc = [Line2D([0], [0], color=color_map[a], lw=3, label=a) for a in ["Incorrect", "IDK", "Correct", "NC"]]
    legA = ax.legend(handles=leg_acc, title="acc_type (color)", loc="upper left", handlelength=3.2)
    ax.add_artist(legA)

    # Legend B: conditions (marker shapes; neutral color; no line)
    cond_handles = []
    for cond, marker in cond_markers.items():
        h = Line2D([0], [0], color="black", lw=0, marker=marker, markersize=7, label=cond,
                   markerfacecolor="none", markeredgewidth=1.5)
        cond_handles.append(h)
    ax.legend(handles=cond_handles, title="condition (marker)", loc="upper right", ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Wrote {out_path.resolve()}")
    
    return pct_baseline, pct_len

# ---------- New: Incorrect-only per condition/length (color = condition) ----------
def plot_incorrect_by_condition(merged: pd.DataFrame, label: str, out_path: Path):
    """
    One plot per model: y = percent Incorrect, x = length (L),
    one colored line per condition. Baseline is a single dot at L=1.
    """
    if merged.empty:
        print(f"[PLOT] {label} incorrect: no data.")
        return

    # Compute % tables
    base_mask = (merged["condition"] == "baseline") | (merged.get("L").fillna(-1).astype("Int64") == 1)
    baseline_df = merged.loc[base_mask].copy()
    counts_baseline = baseline_df.groupby(["acc_type"], dropna=False).size().reset_index(name="n")
    counts_baseline["condition"] = "baseline"
    pct_baseline = percent_table(counts_baseline, ["condition"])

    keep_conds = ["meta", "memory", "semantic", "underspecified", "misleading"]
    len_df = merged[merged["condition"].isin(keep_conds)].copy()
    len_df = len_df[~len_df["L"].isna()]
    try:
        len_df["L"] = len_df["L"].astype(int)
    except Exception:
        pass
    counts_len = (
        len_df.groupby(["condition", "L", "acc_type"], dropna=False)
              .size().reset_index(name="n")
    )
    pct_len = percent_table(counts_len, ["condition", "L"])

    # Filter to Incorrect only
    base_inc = pct_baseline[(pct_baseline["condition"] == "baseline") & (pct_baseline["acc_type"] == "Incorrect")].copy()
    base_inc["L"] = 1

    len_inc = pct_len[pct_len["acc_type"] == "Incorrect"].copy()

    # Colors per condition (solid lines)
    cond_colors = {
        "baseline":       "black",
        "meta":           "#1f77b4",  # blue
        "memory":         "#ff7f0e",  # orange
        "semantic":       "#2ca02c",  # green
        "underspecified": "#d62728",  # red
        "misleading":     "#9467bd",  # purple
    }

    all_L = sorted(set([1] + len_inc["L"].astype(int).tolist()))
    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline dot at L=1
    if not base_inc.empty:
        ax.scatter([1], [float(base_inc["percent"].iloc[0])], color=cond_colors["baseline"], s=65, label="baseline", zorder=3)

    # Non-baseline lines
    for cond in ["meta", "memory", "semantic", "underspecified", "misleading"]:
        sub = len_inc[len_inc["condition"] == cond].copy()
        if not sub.empty:
            sub = sub.sort_values("L")
            ax.plot(
                sub["L"].astype(int),
                sub["percent"].astype(float),
                marker="o",
                linestyle="-",
                color=cond_colors[cond],
                linewidth=2.5,
                markersize=5,
                label=cond
            )

    ax.set_xlabel("Length (L)")
    ax.set_ylabel("Percent Incorrect")
    ax.set_xticks(all_L)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{label}: Incorrect (%) across length by condition")
    ax.legend(loc="upper right", ncol=2, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PLOT] Wrote {out_path.resolve()}")


# ---------- Main ----------
def main():
    keep_ids = load_final_ids(FINAL_JSONL)

    # Build masters
    phi3 = build_master(PHI3_FOLDERS, keep_ids, "phi3", OUT_PHI3_MASTER)
    phi4 = build_master(PHI4_FOLDERS, keep_ids, "phi4", OUT_PHI4_MASTER)

    # Per-model combined plots (acc types view kept as-is)
    phi3_base_pct, phi3_len_pct = plot_by_length(phi3, "phi3", OUT_PHI3_PLOT)
    phi4_base_pct, phi4_len_pct = plot_by_length(phi4, "phi4", OUT_PHI4_PLOT)

    # New: Incorrect-only, across lengths, per condition (one plot per model)
    plot_incorrect_by_condition(phi3, "phi3", OUT_PHI3_INCORRECT)
    plot_incorrect_by_condition(phi4, "phi4", OUT_PHI4_INCORRECT)

if __name__ == "__main__":
    main()