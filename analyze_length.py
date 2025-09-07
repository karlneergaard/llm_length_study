#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_length.py — scoring & plots for the Length vs Veracity study

- Scores final-turn YES/NO (English only) -> strict correctness
- Summaries by (L, scaffold)
- McNemar tests:
    (A) Adjacent lengths within each scaffold (e.g., 3→6, 6→9, 9→12)
    (B) Versus baseline (L=1, scaffold='baseline') for each scaffold & length L>1
- Plot: accuracy vs L by scaffold with annotations for (A) and (B)

python analyze_length.py --in-root runs \
  --make-plots
# (point --in-root at a parent containing runs/baseline_*, runs/light_*, runs/rich_*)
"""

import argparse, json, re, math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Basic helpers
# -----------------------
def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

_word_re = re.compile(r"^\W*([A-Za-z]+)")

def first_token_english_yes_no(text: str) -> Tuple[str, bool]:
    """
    Return (first_token_lower, compliant_flag).
    Compliant iff token is 'yes' or 'no' (English only).
    """
    s = text.strip()
    m = _word_re.search(s)
    if not m:
        return ("", False)
    tok = m.group(1).lower()
    if tok in ("yes", "no"):
        return (tok, True)
    return (tok, False)

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    halfw = (z * ((p*(1-p)/n + (z**2)/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


# -----------------------
# Extract final reply & score
# -----------------------
def extract_final_reply(d: dict) -> str:
    L = int(d.get("length", 0))
    tr = d.get("transcript", [])
    final = None
    for row in tr:
        if row.get("role") == "model" and int(row.get("turn", -1)) == L:
            final = row.get("reply", "")
    if final is None:
        for row in reversed(tr):
            if row.get("role") == "model":
                final = row.get("reply", "")
                break
    return final or ""

def degenerate_reply(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if all(ch in "!?.;:-_=+*/\\|~`'^\"”“’’,<>()[]{} \t" for ch in s):
        return True
    if len(set(s)) == 1 and len(s) >= 3:
        return True
    return False

def score_file(path: Path) -> Dict:
    d = read_json(path)
    scaffold = d.get("scaffold")
    L = int(d.get("length", 0))
    boolq_id = d.get("boolq_id") or ""
    gold = d.get("gold_answer")
    question = d.get("question", "")

    reply = extract_final_reply(d)
    first_tok, compliant = first_token_english_yes_no(reply)
    pred = None
    if compliant:
        pred = True if first_tok == "yes" else False
    correct_strict = int(compliant and pred == gold)
    correct_comp_only = int(compliant and pred == gold)

    has_expl = False
    if compliant:
        rest = reply.strip()[len(first_tok):].strip()
        has_expl = len(rest) > 0

    words = len(reply.strip().split())
    chars = len(reply)

    return {
        "path": str(path),
        "file": path.name,
        "boolq_id": boolq_id,
        "L": L,
        "scaffold": scaffold,
        "gold": gold,
        "question": question,
        "first_token": first_tok,
        "compliant": int(compliant),
        "pred": pred if pred is not None else "",
        "correct_strict": correct_strict,
        "correct_comp_only": correct_comp_only,
        "reply_len_words": words,
        "reply_len_chars": chars,
        "has_explanation": int(has_expl),
        "degenerate_flag": int(degenerate_reply(reply)),
        "timing_s": float(d.get("timing_s", np.nan)),
    }


# -----------------------
# McNemar utilities
# -----------------------
def mcnemar(b: int, c: int) -> Tuple[float, float]:
    """
    Continuity-corrected McNemar chi-square and p-value (df=1).
    If b+c==0 => chi2=0, p=1.
    """
    if (b + c) == 0:
        return 0.0, 1.0
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    # Chi-square df=1 => p = erfc(sqrt(chi2/2))
    pval = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, pval

def p_to_stars(p: float) -> str:
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"

def mcnemar_adjacent_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjacent lengths within each scaffold using strict correctness.
    Returns tidy rows per (scaffold, L1->L2).
    """
    records = []
    for sc in sorted(df["scaffold"].dropna().unique()):
        dsc = df[df["scaffold"] == sc].copy()
        piv = dsc.pivot_table(index="boolq_id", columns="L",
                              values="correct_strict", aggfunc="first")
        lens = sorted([c for c in piv.columns if pd.notnull(c)])
        for i in range(len(lens) - 1):
            L1, L2 = int(lens[i]), int(lens[i+1])
            sub = piv[[L1, L2]].dropna()
            if sub.empty:
                b = c = n_pair = 0
                a11 = a00 = 0
                acc_L1 = acc_L2 = float("nan")
            else:
                a11 = int(((sub[L1] == 1) & (sub[L2] == 1)).sum())
                a00 = int(((sub[L1] == 0) & (sub[L2] == 0)).sum())
                b = int(((sub[L1] == 1) & (sub[L2] == 0)).sum())
                c = int(((sub[L1] == 0) & (sub[L2] == 1)).sum())
                n_pair = len(sub)
                acc_L1 = float((sub[L1] == 1).mean())
                acc_L2 = float((sub[L2] == 1).mean())
            chi2, pval = mcnemar(b, c)
            records.append({
                "type": "adjacent",
                "scaffold": sc,
                "L1": L1,
                "L2": L2,
                "n_pairs": int(n_pair),
                "both_correct": int(a11),
                "both_incorrect": int(a00),
                "b_L1_correct_L2_incorrect": int(b),
                "c_L1_incorrect_L2_correct": int(c),
                "chi2_cc": float(chi2),
                "p_value": float(pval),
                "acc_L1": acc_L1,
                "acc_L2": acc_L2,
                "acc_diff": (acc_L2 - acc_L1) if (n_pair > 0 and not np.isnan(acc_L1) and not np.isnan(acc_L2)) else float("nan"),
            })
    return pd.DataFrame.from_records(records)

def mcnemar_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each scaffold & length L>1 against baseline (scaffold='baseline', L=1)
    using strict correctness on paired items.
    """
    # Baseline correctness per id
    base = df[(df["scaffold"] == "baseline") & (df["L"] == 1)][["boolq_id", "correct_strict"]].copy()
    base = base.rename(columns={"correct_strict": "corr_base"})
    base.set_index("boolq_id", inplace=True)

    records = []
    if base.empty:
        return pd.DataFrame(columns=[
            "type","scaffold","L_base","L_target","n_pairs","both_correct","both_incorrect",
            "b_base_correct_target_incorrect","c_base_incorrect_target_correct",
            "chi2_cc","p_value","acc_base","acc_target","acc_diff"
        ])

    for sc in sorted(df["scaffold"].dropna().unique()):
        if sc == "baseline":
            continue
        dsc = df[df["scaffold"] == sc].copy()
        piv = dsc.pivot_table(index="boolq_id", columns="L",
                              values="correct_strict", aggfunc="first")
        for L in sorted([c for c in piv.columns if pd.notnull(c) and c != 1]):
            target = piv[[L]].dropna().rename(columns={L: "corr_target"})
            # align with baseline
            merged = base.join(target, how="inner")
            if merged.empty:
                b = c = n_pair = 0
                a11 = a00 = 0
                acc_b = acc_t = float("nan")
            else:
                a11 = int(((merged["corr_base"] == 1) & (merged["corr_target"] == 1)).sum())
                a00 = int(((merged["corr_base"] == 0) & (merged["corr_target"] == 0)).sum())
                b = int(((merged["corr_base"] == 1) & (merged["corr_target"] == 0)).sum())
                c = int(((merged["corr_base"] == 0) & (merged["corr_target"] == 1)).sum())
                n_pair = len(merged)
                acc_b = float((merged["corr_base"] == 1).mean())
                acc_t = float((merged["corr_target"] == 1).mean())
            chi2, pval = mcnemar(b, c)
            records.append({
                "type": "vs_baseline",
                "scaffold": sc,
                "L_base": 1,
                "L_target": int(L),
                "n_pairs": int(n_pair),
                "both_correct": int(a11),
                "both_incorrect": int(a00),
                "b_base_correct_target_incorrect": int(b),
                "c_base_incorrect_target_correct": int(c),
                "chi2_cc": float(chi2),
                "p_value": float(pval),
                "acc_base": acc_b,
                "acc_target": acc_t,
                "acc_diff": (acc_t - acc_b) if (not np.isnan(acc_b) and not np.isnan(acc_t)) else float("nan"),
            })
    return pd.DataFrame.from_records(records)


# -----------------------
# Main analysis
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze Length vs Veracity outputs and plot curves.")
    ap.add_argument("--in-root", required=True, help="Directory containing JSON outputs from run_length.py (you can point to a parent folder containing baseline/light/rich).")
    ap.add_argument("--out-dir", default=None, help="Where to write CSVs/plots (default: <in-root>/analysis)")
    ap.add_argument("--make-plots", action="store_true", help="Save accuracy/compliance plots as PNGs")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    if not in_root.exists():
        raise SystemExit(f"Not found: {in_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (in_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect JSONs
    files = [p for p in in_root.rglob("*.json") if p.name != "_manifest.json"]
    if not files:
        raise SystemExit("No JSON files found under --in-root.")

    # Score
    rows = [score_file(p) for p in files]
    df = pd.DataFrame(rows)

    # Detailed results
    detailed_path = out_dir / "results_detailed.csv"
    df.to_csv(detailed_path, index=False)
    print(f"[WRITE] {detailed_path}")

    # Summary by (L, scaffold)
    grp = df.groupby(["L", "scaffold"], as_index=False)
    def summarize(group: pd.DataFrame) -> Dict:
        n = len(group)
        comp = int(group["compliant"].sum())
        acc_strict = int(group["correct_strict"].sum())
        comp_rows = group[group["compliant"] == 1]
        acc_comp = int(comp_rows["correct_comp_only"].sum())
        n_comp = len(comp_rows)

        c_lo, c_hi = wilson_ci(comp, n)
        a_lo, a_hi = wilson_ci(acc_strict, n)
        ac_lo, ac_hi = wilson_ci(acc_comp, n_comp) if n_comp > 0 else (float("nan"), float("nan"))

        return {
            "N": n,
            "compliance_rate": comp / n if n else float("nan"),
            "compliance_ci_low": c_lo,
            "compliance_ci_high": c_hi,
            "acc_strict": acc_strict / n if n else float("nan"),
            "acc_strict_ci_low": a_lo,
            "acc_strict_ci_high": a_hi,
            "acc_comp_only": acc_comp / n_comp if n_comp else float("nan"),
            "acc_comp_only_ci_low": ac_lo,
            "acc_comp_only_ci_high": ac_hi,
            "mean_time_s": group["timing_s"].mean(),
            "mean_words": group["reply_len_words"].mean(),
        }

    summary_records = []
    for (L, sc), g in grp:
        rec = {"L": int(L), "scaffold": sc}
        rec.update(summarize(g))
        summary_records.append(rec)
    df_sum = pd.DataFrame(summary_records).sort_values(["scaffold", "L"])
    sum_path = out_dir / "summary_by_L_scaffold.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"[WRITE] {sum_path}")

    # Overall summary
    grp2 = df.groupby("scaffold", as_index=False)
    overall_records = []
    for sc, g in grp2:
        n = len(g)
        comp = int(g["compliant"].sum())
        acc_strict = int(g["correct_strict"].sum())
        c_lo, c_hi = wilson_ci(comp, n)
        a_lo, a_hi = wilson_ci(acc_strict, n)
        overall_records.append({
            "scaffold": sc,
            "N": n,
            "compliance_rate": comp / n if n else float("nan"),
            "compliance_ci_low": c_lo,
            "compliance_ci_high": c_hi,
            "acc_strict": acc_strict / n if n else float("nan"),
            "acc_strict_ci_low": a_lo,
            "acc_strict_ci_high": a_hi,
            "mean_time_s": g["timing_s"].mean(),
            "mean_words": g["reply_len_words"].mean(),
        })
    df_overall = pd.DataFrame(overall_records).sort_values("scaffold")
    overall_path = out_dir / "summary_overall.csv"
    df_overall.to_csv(overall_path, index=False)
    print(f"[WRITE] {overall_path}")

    # Paired tables (per scaffold), useful for sanity checks
    for sc in sorted(df["scaffold"].dropna().unique()):
        dsc = df[df["scaffold"] == sc].copy()
        piv_pred = dsc.pivot_table(index="boolq_id", columns="L", values="pred", aggfunc="first")
        piv_comp = dsc.pivot_table(index="boolq_id", columns="L", values="compliant", aggfunc="first")
        piv_corr = dsc.pivot_table(index="boolq_id", columns="L", values="correct_strict", aggfunc="first")
        piv_pred.columns = [f"pred_L{int(c)}" for c in piv_pred.columns]
        piv_comp.columns = [f"comp_L{int(c)}" for c in piv_comp.columns]
        piv_corr.columns = [f"corr_L{int(c)}" for c in piv_corr.columns]
        paired = pd.concat([piv_pred, piv_comp, piv_corr], axis=1).reset_index()
        paired_path = out_dir / f"paired_by_id_{sc}.csv"
        paired.to_csv(paired_path, index=False)
        print(f"[WRITE] {paired_path}")

    # Significance: adjacent within-scaffold
    df_sig_adj = mcnemar_adjacent_pairs(df)
    sig_adj_path = out_dir / "significance_adjacent_by_scaffold.csv"
    df_sig_adj.to_csv(sig_adj_path, index=False)
    print(f"[WRITE] {sig_adj_path}")

    # Significance: vs baseline (per scaffold & L>1)
    df_sig_base = mcnemar_vs_baseline(df)
    sig_base_path = out_dir / "significance_vs_baseline_by_scaffold.csv"
    df_sig_base.to_csv(sig_base_path, index=False)
    print(f"[WRITE] {sig_base_path}")

    # Plots
    if args.make_plots:
        # Accuracy vs L (strict), annotated
        plt.figure()
        ymin, ymax = 1.0, 0.0
        scaffolds = list(sorted(df_sum["scaffold"].unique()))
        for sc in scaffolds:
            dsc = df_sum[df_sum["scaffold"] == sc].sort_values("L")
            plt.plot(dsc["L"], dsc["acc_strict"], marker="o", label=sc)
            ymin = min(ymin, dsc["acc_strict"].min())
            ymax = max(ymax, dsc["acc_strict"].max())

        # Adjacent annotations (per scaffold)
        for sc in scaffolds:
            if sc == "baseline":
                continue
            dsc = df_sum[df_sum["scaffold"] == sc].sort_values("L")
            sig_sc = df_sig_adj[df_sig_adj["scaffold"] == sc].sort_values(["L1", "L2"])
            for _, row in sig_sc.iterrows():
                L1, L2 = int(row["L1"]), int(row["L2"])
                p = float(row["p_value"])
                stars = p_to_stars(p)
                # y at a bit above the higher of the two points
                y1 = float(dsc.loc[dsc["L"] == L1, "acc_strict"].values[0])
                y2 = float(dsc.loc[dsc["L"] == L2, "acc_strict"].values[0])
                y_annot = max(y1, y2) + 0.02
                x_mid = (L1 + L2) / 2.0
                txt = f"{stars}\n(p={p:.3g})" if stars != "ns" else "ns"
                plt.text(x_mid, y_annot, txt, ha="center", va="bottom", fontsize=9)
                ymax = max(ymax, y_annot)

        # Baseline-vs-L annotations (per scaffold & L>1)
        if not df_sig_base.empty:
            for sc in sorted(df_sig_base["scaffold"].unique()):
                sig_sc = df_sig_base[df_sig_base["scaffold"] == sc]
                dsc = df_sum[(df_sum["scaffold"] == sc)].sort_values("L")
                for _, row in sig_sc.iterrows():
                    L = int(row["L_target"])
                    p = float(row["p_value"])
                    stars = p_to_stars(p)
                    # point coordinates
                    if (dsc["L"] == L).any():
                        y = float(dsc.loc[dsc["L"] == L, "acc_strict"].values[0])
                        y_annot = y + 0.04  # slightly higher than adjacent labels
                        txt = f"{stars} vs L1\n(p={p:.3g})" if stars != "ns" else "ns vs L1"
                        plt.text(L, y_annot, txt, ha="center", va="bottom", fontsize=9)
                        ymax = max(ymax, y_annot)

        plt.xlabel("Length (L)")
        plt.ylabel("Accuracy (strict)")
        plt.title("BoolQ Accuracy vs Length (by scaffold)\nAdjacency and vs-baseline McNemar significance")
        plt.legend()
        plt.ylim(max(0.0, ymin - 0.05), min(1.0, ymax + 0.03))
        plt.grid(True, linestyle=":")
        acc_plot = out_dir / "accuracy_vs_L_by_scaffold.png"
        plt.savefig(acc_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {acc_plot}")

        # Compliance vs L (no stats)
        plt.figure()
        for sc in scaffolds:
            dsc = df_sum[df_sum["scaffold"] == sc].sort_values("L")
            plt.plot(dsc["L"], dsc["compliance_rate"], marker="o", label=sc)
        plt.xlabel("Length (L)")
        plt.ylabel("Compliance (first token YES/NO)")
        plt.title("Compliance vs Length (by scaffold)")
        plt.legend()
        plt.grid(True, linestyle=":")
        comp_plot = out_dir / "compliance_vs_L_by_scaffold.png"
        plt.savefig(comp_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] {comp_plot}")

    print("[DONE] Analysis complete.")


if __name__ == "__main__":
    main()
