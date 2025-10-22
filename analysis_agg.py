# phi4_rows.py
import json
import pandas as pd
from pathlib import Path

import argparse

# Headless backend so PNG saving works without a GUI
import matplotlib
matplotlib.use("Agg")

from os import getenv

# ---------- Paths ----------
# BASELINE_CSV = RUNS_DIR / "baseline" / "detailed_rows.csv"
# META_CSV     = RUNS_DIR / "meta"     / "detailed_rows.csv"
# SEM_CSV      = RUNS_DIR / "semantic" / "detailed_rows.csv"

SCAFFOLDS = ["baseline", "meta", "semantic", "misleading", "underspecified"]
LINESTY = {"meta": "-", "semantic": "--", "misleading": "-.", "underspecified": ":"}

# ---------- Helpers ----------
def load_final_ids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {ln} in {path}: {e}") from e
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
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    # Accept 'boolq' or 'boolq_id' as id column
    key_col = None
    for cand in ("boolq", "boolq_id"):
        if cand in df.columns:
            key_col = cand
            break
    if key_col is None:
        raise ValueError(f"{path} must have a 'boolq' or 'boolq_id' column. Found: {list(df.columns)}")

    df["boolq_id"] = df[key_col].astype(str)
    df = df[df["boolq_id"].isin(keep_ids)].copy()
    df["condition"] = condition
    return df

def compute_acc_type(row) -> str:
    """
    Priority:
      1) IDK if is_idk is truthy OR pred_label == 'idk'
      2) NC  if pred_label == 'other'
      3) Otherwise from answered_correct (TRUE/FALSE/IDK/NC if present)
      4) Else 'NA'
    """
    pred_label = str(row.get("pred_label", row.get("pred_lable", ""))).strip().lower()
    if truthy(row.get("is_idk")) or pred_label == "idk":
        return "IDK"
    if pred_label == "other":
        return "NC"

    ac = row.get("answered_correct")
    if ac is not None:
        ac_s = str(ac).strip().upper()
        if ac_s in {"TRUE", "FALSE", "IDK", "NC"}:
            return ac_s

    return "NA"

ACC_MAP = {"FALSE": "Incorrect", "TRUE": "Correct", "IDK": "IDK", "NC": "NC", "NA": "NA"}
ACC_LEVELS = ["Incorrect", "Correct", "IDK", "NC"]  # desired order everywhere

def percent_table(count_df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Given counts with columns: group_cols + ['acc_type','n'],
    return percentages per group (denominator = sum of n per group).
    Assumes acc_type ∈ ACC_LEVELS only (we’ll scaffold missing with n=0).
    """
    df = count_df.copy()

    # Ensure all acc_type levels exist per group (fill missing with n=0)
    all_groups = df[group_cols].drop_duplicates()
    scaffold = (
        all_groups.assign(key=1)
        .merge(pd.DataFrame({"acc_type": ACC_LEVELS, "key": 1}), on="key")
        .drop(columns="key")
    )
    df = scaffold.merge(df, on=group_cols + ["acc_type"], how="left")
    df["n"] = df["n"].fillna(0)

    # Denominator per group
    totals = df.groupby(group_cols, as_index=False)["n"].sum().rename(columns={"n": "total"})
    df = df.merge(totals, on=group_cols, how="left")

    # Percentages
    df["percent"] = (df["n"] / df["total"]).where(df["total"] > 0, 0.0)

    # Ordering
    df["acc_type"] = pd.Categorical(df["acc_type"], categories=ACC_LEVELS, ordered=True)
    return df.sort_values(group_cols + ["acc_type"]).reset_index(drop=True)

def plot_combined_lengths(pct_baseline: pd.DataFrame, pct_len: pd.DataFrame, out_path: Path):
    """
    Single plot: y=percent, x=Length (L).
    - Baseline: 4 dots at L=1 (no lines)
    - Meta: 4 series (one per acc_type), solid lines/dots
    - Semantic: same colors per acc_type, dashed lines/dots
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    # Build tidy plotting frame: condition, L, acc_type, percent
    base = pct_baseline.copy()
    base = base.loc[base["condition"] == "baseline", ["acc_type", "percent"]].copy()
    base["condition"] = "baseline"
    base["L"] = 1

    bylen = pct_len.copy()  # has: condition, L, acc_type, percent
    df_plot = pd.concat([
        base[["condition","L","acc_type","percent"]],
        bylen[["condition","L","acc_type","percent"]],
    ], ignore_index=True)

    if df_plot.empty:
        print("[PLOT] No data to plot for combined lengths.")
        return

    # Colors per new acc_type labels
    color_map = {
        "Incorrect": "tab:red",
        "Correct":   "tab:green",
        "IDK":       "tab:orange",
        "NC":        "tab:blue",
    }

    # X ticks from observed L (ensure 1 is present for baseline)
    all_L = sorted(set(df_plot["L"].dropna().astype(int).tolist()))
    if 1 not in all_L:
        all_L = [1] + all_L

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot baseline dots only (L=1)
    base_sub = df_plot[(df_plot["condition"] == "baseline") & (df_plot["L"] == 1)]
    for acc in ACC_LEVELS:
        row = base_sub[base_sub["acc_type"] == acc]
        if not row.empty:
            ax.scatter([1], [float(row["percent"].iloc[0])],
                       color=color_map[acc], s=50, zorder=3)

    # Plot others
    for acc in ACC_LEVELS:
        for this_scaffold in SCAFFOLDS[1:]:
            this_df = df_plot[(df_plot["condition"] == this_scaffold) & (df_plot["acc_type"] == acc)].copy()
            if not this_df.empty:
                this_df = this_df.sort_values("L")

                ax.plot(this_df["L"].astype(int), this_df["percent"].astype(float),
                    marker="o", linestyle=LINESTY[this_scaffold], color=color_map[acc], linewidth=2)
                
        # # meta series
        # m = df_plot[(df_plot["condition"] == "meta") & (df_plot["acc_type"] == acc)].copy()
        # if not m.empty:
        #     m = m.sort_values("L")
        #     ax.plot(m["L"].astype(int), m["percent"].astype(float),
        #             marker="o", linestyle="-", color=color_map[acc], linewidth=2)

        # # semantic series
        # s = df_plot[(df_plot["condition"] == "semantic") & (df_plot["acc_type"] == acc)].copy()
        # if not s.empty:
        #     s = s.sort_values("L")
        #     ax.plot(s["L"].astype(int), s["percent"].astype(float),
        #             marker="o", linestyle="--", color=color_map[acc], linewidth=2)

    # Axes & formatting
    ax.set_xlabel("Length (L)")
    ax.set_ylabel("Percent")
    ax.set_xticks(all_L)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    ax.grid(True, alpha=0.25)
    ax.set_title("Percent by acc_type across length and condition")

    # Legends
    color_handles = [Line2D([0], [0], color=color_map[a], lw=3, label=a) for a in ACC_LEVELS]
    style_handles = [Line2D([0], [0], color="black", lw=3, linestyle=LINESTY[s],  label=s) for s in SCAFFOLDS[1:]]

    leg1 = ax.legend(handles=color_handles, title="acc_type", loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="condition", loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Wrote {out_path.resolve()}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_final", required=True, type=str,
                    help="Path to data/boolq_final.jsonl")
    ap.add_argument("--in_dir", required=True, type=str,
                    help="directory for csv file")
    ap.add_argument("--out_dir", default=None,
                    help="directory to save file")

    args = ap.parse_args()

    # paths
    final_jsonl = Path(args.in_final)
    in_dir = Path(args.in_dir)
    RUNS_DIR = Path(args.out_dir) if args.out_dir else in_dir.joinpath("aggregated")
    
    out_path = RUNS_DIR.joinpath("rows.csv")
    out_path.parent.mkdir(exist_ok=True)

    file_paths, file_scaffolds = [], []
    for scd in SCAFFOLDS:
        pth = in_dir.joinpath("{}/detailed_rows.csv".format(scd))
        if not pth.exists():
            continue

        file_paths.append(pth)
        file_scaffolds.append(scd) 

    if not file_paths:
        print("!! No csv files found")
        exit()
    
    # Merge phase
    keep_ids = load_final_ids(final_jsonl)

    parts = [load_detail_csv(fpth, fsca, keep_ids) for (fpth, fsca) in zip(file_paths, file_scaffolds)]

    merged = pd.concat(parts, ignore_index=True)

    # Normalize columns we inspect
    for col in ("answered_correct", "pred_label", "pred_lable"):
        if col in merged.columns:
            merged[col] = merged[col].astype(str)

    # Compute original acc_type (uppercase set), then map to new labels
    merged["_acc_upper"] = merged.apply(compute_acc_type, axis=1)
    merged["acc_type"] = merged["_acc_upper"].map(ACC_MAP).fillna("NA")

    # Ensure L exists; we’ll treat baseline separately
    if "L" not in merged.columns:
        merged["L"] = pd.NA

    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")

    # Analysis phase
    # Make L numeric where possible
    if "L" in merged.columns:
        with pd.option_context("mode.chained_assignment", None):
            try:
                merged["L"] = merged["L"].astype("Int64")
            except Exception:
                pass

    # (A) Baseline only (condition=='baseline' OR L==1)
    baseline_mask = (merged["condition"] == "baseline") | (merged.get("L").fillna(-1).astype("Int64") == 1)
    baseline_df = merged.loc[baseline_mask].copy()

    counts_baseline = (
        baseline_df.groupby(["acc_type"], dropna=False)
                   .size().reset_index(name="n")
    )
    counts_baseline["condition"] = "baseline"
    pct_baseline = percent_table(counts_baseline, ["condition"])
    (RUNS_DIR / "acc_baseline.csv").parent.mkdir(parents=True, exist_ok=True)
    pct_baseline.to_csv(RUNS_DIR / "acc_baseline.csv", index=False)
    print(f"[CSV] Wrote {RUNS_DIR / 'phi4_acc_baseline.csv'}")

    # (B) other conditions by length
    len_df = merged[~merged["condition"].isin(["baseline"])].copy()
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

    pct_len.to_csv(RUNS_DIR / "acc_by_length.csv", index=False)
    print(f"[CSV] Wrote {RUNS_DIR / 'phi4_acc_by_length.csv'}")

    # Combined plot
    plot_combined_lengths(pct_baseline, pct_len, RUNS_DIR / "acc_combined.png")

    # Quick peek
    print("\n[Baseline %]")
    print(pct_baseline[["condition","acc_type","percent"]])
    print("\n[By length %] (head)")
    print(pct_len.sort_values(["condition","L","acc_type"]).head(12))

if __name__ == "__main__":
    main()
