#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_length.py — scoring & plots for the Length vs Veracity study

Now supports:
- Strict Y/N compliance & accuracy (unchanged from before).
- Lenient (abstention-aware) parsing with labels: {yes,no,idk,other}.
- Coverage (commit rate), IDK rate, answered-only accuracy.
- Strict McNemar:
    (A) Adjacent lengths within each scaffold (e.g., 3→6, 6→9, ...)
    (B) Versus baseline (L=1, scaffold='baseline') for each scaffold & length L>1
- Answered-only McNemar (lenient): adjacent lengths, restricted to rows that answered in both.

Also writes paired-by-id tables with extended columns for inspection.

Usage
-----
python analyze_length.py --in-root runs \
  --make-plots \
  --yn-map extended

Notes
-----
- This script is a drop-in replacement; outputs are written to --out-dir
  (default: <in-root>/_analysis) and include both strict and lenient views.
- If you prefer your old-only layout, the "strict" CSVs/plots here are
  directly comparable to your previous ones.



  
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------
# Utility
# -----------------------

def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_manifests(root: Path) -> List[Path]:
    return list(root.rglob("_manifest.json"))

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    halfw = (z * ((p*(1-p)/n + (z**2)/(4*n*n)) ** 0.5)) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))

def mcnemar_cc(b: int, c: int) -> Tuple[float, float]:
    """Continuity-corrected McNemar (chi2, p) with 1 dof; p approx via exp(-x/2)."""
    if b + c == 0:
        return (0.0, 1.0)
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = math.exp(-chi2 / 2.0)
    return (chi2, p)

def p_to_stars(p: float) -> str:
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return ""

# -----------------------
# Extract final reply & parsers
# -----------------------

_WORD_RE = re.compile(r"[A-Za-z]+")
_IDK_PATTERNS = [
    r"\bi\s*(?:do\s*not|don'?t)\s*know\b",
    r"\bnot\s*sure\b",
    r"\bunsure\b",
    r"\bunknown\b",
    r"\bcannot\s*(?:tell|determine)\b",
    r"\bno\s*idea\b",
]
_YES_PATTERNS_EXT = [r"^yes\b", r"^\by\b", r"\btrue\b", r"\bcorrect\b"]
_NO_PATTERNS_EXT  = [r"^no\b",  r"^\bn\b", r"\bfalse\b", r"\bincorrect\b"]

def extract_final_reply(d: dict) -> str:
    tr = d.get("transcript", [])
    best_turn = -1
    final_reply = ""
    for row in tr:
        if row.get("role") == "model":
            t = int(row.get("turn", -1))
            if t >= best_turn:
                best_turn = t
                final_reply = row.get("reply", "") or ""
    return final_reply

def first_token_english_yes_no(text: str) -> Tuple[str, bool]:
    """Strict: only accept first word yes/no as compliant."""
    s = (text or "").lstrip()
    m = _WORD_RE.search(s)
    if not m:
        return ("", False)
    tok = m.group(0).lower()
    if tok in ("yes", "no"):
        return (tok, True)
    return (tok, False)

def extract_decision_lenient(text: str, yn_map: str) -> Tuple[str, str]:
    """
    Lenient: return (pred_label, pred_raw), pred_label ∈ {'yes','no','idk','other'}.
    pred_raw is the matched phrase (for auditing). Earliest decisive cue wins.
    """
    s = (text or "").strip().lower()
    # Remove leading 'Answer:' labels if present
    s = re.sub(r"^\s*(?:answer|final answer)\s*[:\-–]\s*", "", s)

    def _search_any(patterns: List[str]) -> Optional[re.Match]:
        first = None
        for pat in patterns:
            m = re.search(pat, s, flags=re.I)
            if m and (first is None or m.start() < first.start()):
                first = m
        return first

    idk_m = _search_any(_IDK_PATTERNS)
    yes_pats = [r"^yes\b"] if yn_map == "plain" else _YES_PATTERNS_EXT
    no_pats  = [r"^no\b"]  if yn_map == "plain" else _NO_PATTERNS_EXT
    yes_m = _search_any(yes_pats)
    no_m  = _search_any(no_pats)

    candidates = []
    if idk_m: candidates.append(("idk", idk_m))
    if yes_m: candidates.append(("yes", yes_m))
    if no_m:  candidates.append(("no",  no_m))
    if not candidates:
        return ("other", "")
    label, match = min(candidates, key=lambda kv: kv[1].start())
    return (label, s[match.start():match.end()])

def degenerate_reply(text: str) -> bool:
    if not text or not text.strip():
        return True
    s = text.strip()
    if len(s) <= 1:
        return True
    if re.fullmatch(r"[\.\,\!\?]+", s):
        return True
    return False

# -----------------------
# Rows & scoring
# -----------------------

@dataclass
class Row:
    scaffold: str
    L: int
    boolq_id: str
    gold: int  # 1 if True, 0 if False
    reply: str
    timing_s: float

    # strict
    first_token: str
    compliant: bool
    pred_strict: Optional[int]  # 1 yes, 0 no, None if non-compliant
    correct_strict: Optional[bool]

    # lenient
    pred_label: str            # 'yes','no','idk','other'
    pred_raw: str
    is_answered: bool
    is_idk: bool
    answered_correct: Optional[bool]

def score_record(d: dict, yn_map: str) -> Row:
    L = int(d.get("length", 0))
    sc = str(d.get("scaffold", "") or "")
    bid = str(d.get("boolq_id", "") or "")
    gold = 1 if int(d.get("gold_answer", 0)) == 1 else 0
    reply = extract_final_reply(d)
    first_tok, compliant = first_token_english_yes_no(reply)
    pred_strict = None
    correct_strict = None
    if compliant:
        pred_strict = 1 if first_tok == "yes" else 0
        correct_strict = (pred_strict == gold)

    pred_label, pred_raw = extract_decision_lenient(reply, yn_map)
    is_answered = (pred_label in ("yes", "no"))
    is_idk = (pred_label == "idk")
    answered_correct = None
    if is_answered:
        answered_correct = ((pred_label == "yes") == (gold == 1))

    return Row(
        scaffold=sc, L=L, boolq_id=bid, gold=gold, reply=reply,
        timing_s=float(d.get("timing_s", float("nan"))),
        first_token=first_tok, compliant=compliant,
        pred_strict=pred_strict, correct_strict=correct_strict,
        pred_label=pred_label, pred_raw=pred_raw,
        is_answered=is_answered, is_idk=is_idk,
        answered_correct=answered_correct,
    )

def load_rows(in_root: Path, yn_map: str) -> List[Row]:
    rows: List[Row] = []
    for mani in find_manifests(in_root):
        m = read_json(mani)
        items = m.get("items", [])
        base = mani.parent
        for it in items:
            p = base / it["path"]
            try:
                d = read_json(p)
            except Exception:
                continue
            rows.append(score_record(d, yn_map))
    return rows

# -----------------------
# Summaries
# -----------------------

def summarize_strict(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (sc, L), g in df.groupby(["scaffold","L"], as_index=False):
        n = len(g)
        comp = int(g["compliant"].sum())
        acc_n = int(g["correct_strict"].fillna(False).sum())
        c_lo, c_hi = wilson_ci(comp, n)
        a_lo, a_hi = wilson_ci(acc_n, n)
        recs.append({
            "scaffold": sc, "L": L, "N": n,
            "compliance_rate": comp / n if n else float("nan"),
            "compliance_ci_low": c_lo,
            "compliance_ci_high": c_hi,
            "acc_strict": acc_n / n if n else float("nan"),
            "acc_strict_ci_low": a_lo,
            "acc_strict_ci_high": a_hi,
        })
    return pd.DataFrame(recs).sort_values(["scaffold","L"])

def summarize_lenient(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (sc, L), g in df.groupby(["scaffold","L"], as_index=False):
        n = len(g)
        idk = int((g["pred_label"] == "idk").sum())
        answered = int(g["is_answered"].sum())
        ans_corr = int(g["answered_correct"].fillna(False).sum())
        # overall acc if IDK counted wrong (for easy comparability)
        overall_acc_wrong = (
            int(((g["pred_label"] == "yes") & (g["gold"] == 1)).sum()) +
            int(((g["pred_label"] == "no")  & (g["gold"] == 0)).sum())
        ) / n if n else float("nan")
        answered_acc = (ans_corr / answered) if answered else float("nan")
        recs.append({
            "scaffold": sc, "L": L, "N": n,
            "coverage": answered / n if n else float("nan"),
            "idk_rate": idk / n if n else float("nan"),
            "answered_acc": answered_acc,
            "overall_acc_idk_wrong": overall_acc_wrong,
        })
    return pd.DataFrame(recs).sort_values(["scaffold","L"])

def write_paired_tables(df: pd.DataFrame, out_dir: Path):
    for sc in sorted(df["scaffold"].dropna().unique()):
        dsc = df[df["scaffold"] == sc].copy()
        piv_pred = dsc.pivot_table(index="boolq_id", columns="L", values="first_token", aggfunc="first")
        piv_comp = dsc.pivot_table(index="boolq_id", columns="L", values="compliant", aggfunc="first")
        piv_corr = dsc.pivot_table(index="boolq_id", columns="L", values="correct_strict", aggfunc="first")
        piv_label = dsc.pivot_table(index="boolq_id", columns="L", values="pred_label", aggfunc="first")
        piv_ans = dsc.pivot_table(index="boolq_id", columns="L", values="is_answered", aggfunc="first")
        piv_idk = dsc.pivot_table(index="boolq_id", columns="L", values="is_idk", aggfunc="first")
        piv_anscorr = dsc.pivot_table(index="boolq_id", columns="L", values="answered_correct", aggfunc="first")

        def _rename(prefix, df_):
            df_ = df_.copy()
            df_.columns = [f"{prefix}_L{int(c)}" for c in df_.columns]
            return df_

        paired = pd.concat([
            _rename("pred", piv_pred),
            _rename("comp", piv_comp),
            _rename("corr", piv_corr),
            _rename("label", piv_label),
            _rename("answered", piv_ans),
            _rename("idk", piv_idk),
            _rename("anscorr", piv_anscorr),
        ], axis=1).reset_index()
        paired_path = out_dir / f"paired_by_id_{sc}_extended.csv"
        paired.to_csv(paired_path, index=False)

# -----------------------
# Significance tests
# -----------------------

def significance_adjacent_strict(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for sc in sorted(df["scaffold"].dropna().unique()):
        dsc = df[df["scaffold"] == sc].copy()
        Ls = sorted(dsc["L"].unique())
        for a, b in zip(Ls, Ls[1:]):
            da = dsc[dsc["L"] == a][["boolq_id","correct_strict"]].copy()
            db = dsc[dsc["L"] == b][["boolq_id","correct_strict"]].copy()
            merged = pd.merge(da, db, on="boolq_id", suffixes=(f"_{a}", f"_{b}"))
            a_corr = merged[f"correct_strict_{a}"].fillna(False)
            b_corr = merged[f"correct_strict_{b}"].fillna(False)
            b_cnt = int((a_corr & ~b_corr).sum())  # a correct, b wrong
            c_cnt = int((~a_corr & b_corr).sum())  # a wrong, b correct
            chi2, p = mcnemar_cc(b_cnt, c_cnt)
            recs.append({
                "scaffold": sc, "L_a": a, "L_b": b,
                "n_paired": int(len(merged)),
                "b_a_correct_b_incorrect": b_cnt,
                "c_a_incorrect_b_correct": c_cnt,
                "chi2_cc": chi2, "p_value": p, "stars": p_to_stars(p),
            })
    return pd.DataFrame(recs)

def significance_vs_baseline_strict(df: pd.DataFrame, baseline_scaffold: str = "baseline", baseline_L: int = 1) -> pd.DataFrame:
    recs = []
    dbase = df[(df["scaffold"] == baseline_scaffold) & (df["L"] == baseline_L)][["boolq_id","correct_strict"]].copy()
    if dbase.empty:
        return pd.DataFrame([])
    for sc in sorted(df["scaffold"].dropna().unique()):
        Ls = sorted(df[df["scaffold"] == sc]["L"].unique())
        for L in Ls:
            if sc == baseline_scaffold and L == baseline_L:
                continue
            dcur = df[(df["scaffold"] == sc) & (df["L"] == L)][["boolq_id","correct_strict"]].copy()
            merged = pd.merge(dbase, dcur, on="boolq_id", suffixes=("_base", "_cur"))
            if merged.empty:
                continue
            a_corr = merged["correct_strict_base"].fillna(False)
            b_corr = merged["correct_strict_cur"].fillna(False)
            b_cnt = int((a_corr & ~b_corr).sum())
            c_cnt = int((~a_corr & b_corr).sum())
            chi2, p = mcnemar_cc(b_cnt, c_cnt)
            recs.append({
                "scaffold": sc, "L": L,
                "n_paired": int(len(merged)),
                "b_base_correct_cur_incorrect": b_cnt,
                "c_base_incorrect_cur_correct": c_cnt,
                "chi2_cc": chi2, "p_value": p, "stars": p_to_stars(p),
            })
    return pd.DataFrame(recs)

def significance_adjacent_answered_only(df: pd.DataFrame) -> pd.DataFrame:
    """Lenient: adjacent lengths, rows that answered (Yes/No) in both."""
    recs = []
    for sc in sorted(df["scaffold"].dropna().unique()):
        dsc = df[df["scaffold"] == sc].copy()
        Ls = sorted(dsc["L"].unique())
        for a, b in zip(Ls, Ls[1:]):
            da = dsc[dsc["L"] == a][["boolq_id","is_answered","answered_correct"]].copy()
            db = dsc[dsc["L"] == b][["boolq_id","is_answered","answered_correct"]].copy()
            merged = pd.merge(da, db, on="boolq_id", suffixes=(f"_{a}", f"_{b}"))
            both = merged[merged[f"is_answered_{a}"] & merged[f"is_answered_{b}"]]
            if both.empty:
                recs.append({
                    "scaffold": sc, "L_a": a, "L_b": b,
                    "n_both_answered": 0, "b_a_correct_b_incorrect": 0, "c_a_incorrect_b_correct": 0,
                    "chi2_cc": 0.0, "p_value": 1.0, "stars": ""
                })
                continue
            a_corr = both[f"answered_correct_{a}"].fillna(False)
            b_corr = both[f"answered_correct_{b}"].fillna(False)
            b_cnt = int((a_corr & ~b_corr).sum())
            c_cnt = int((~a_corr & b_corr).sum())
            chi2, p = mcnemar_cc(b_cnt, c_cnt)
            recs.append({
                "scaffold": sc, "L_a": a, "L_b": b,
                "n_both_answered": int(len(both)),
                "b_a_correct_b_incorrect": b_cnt,
                "c_a_incorrect_b_correct": c_cnt,
                "chi2_cc": chi2, "p_value": p, "stars": p_to_stars(p),
            })
    return pd.DataFrame(recs)

# -----------------------
# Plots
# -----------------------

def plot_strict(acc_df: pd.DataFrame, adj_sig: pd.DataFrame, base_sig: pd.DataFrame, out_dir: Path):
    if plt is None or acc_df.empty:
        return
    fig = plt.figure()
    for sc, g in acc_df.groupby("scaffold"):
        plt.plot(g["L"], g["acc_strict"], marker="o", label=sc)
    # annotate adjacent stars at point L_b
    for _, r in adj_sig.iterrows():
        if r["p_value"] < 0.05:
            Lb = r["L_b"]
            sc = r["scaffold"]
            g = acc_df[(acc_df["scaffold"] == sc) & (acc_df["L"] == Lb)]
            if not g.empty:
                y = float(g["acc_strict"].iloc[0])
                plt.text(Lb, y + 0.02, f"A{r['stars']}", ha="center", fontsize=8)
    # annotate baseline stars at point L
    for _, r in base_sig.iterrows():
        if r["p_value"] < 0.05:
            L = r["L"]
            sc = r["scaffold"]
            g = acc_df[(acc_df["scaffold"] == sc) & (acc_df["L"] == L)]
            if not g.empty:
                y = float(g["acc_strict"].iloc[0])
                plt.text(L, y + 0.05, f"B{r['stars']}", ha="center", fontsize=8)
    plt.xlabel("Length (L)")
    plt.ylabel("Strict accuracy")
    plt.title("Strict accuracy vs Length (A = adjacent, B = vs baseline)")
    plt.grid(True, linestyle=":")
    plt.legend()
    outp = out_dir / "acc_strict_vs_L_by_scaffold.png"
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()

def plot_compliance(comp_df: pd.DataFrame, out_dir: Path):
    if plt is None or comp_df.empty:
        return
    fig = plt.figure()
    for sc, g in comp_df.groupby("scaffold"):
        plt.plot(g["L"], g["compliance_rate"], marker="o", label=sc)
    plt.xlabel("Length (L)")
    plt.ylabel("Compliance (first token YES/NO)")
    plt.title("Compliance vs Length (by scaffold)")
    plt.grid(True, linestyle=":")
    plt.legend()
    outp = out_dir / "compliance_vs_L_by_scaffold.png"
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()

def plot_lenient(len_df: pd.DataFrame, out_dir: Path):
    if plt is None or len_df.empty:
        return
    # Coverage + IDK
    fig = plt.figure()
    for sc, g in len_df.groupby("scaffold"):
        plt.plot(g["L"], g["coverage"], marker="o", label=f"{sc} coverage")
        plt.plot(g["L"], g["idk_rate"], marker="x", label=f"{sc} idk")
    plt.xlabel("Length (L)")
    plt.ylabel("Rate")
    plt.title("Coverage & IDK rate vs Length (lenient)")
    plt.grid(True, linestyle=":")
    plt.legend(ncol=2)
    outp = out_dir / "coverage_idk_vs_L_by_scaffold.png"
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()

    # Answered accuracy
    fig = plt.figure()
    for sc, g in len_df.groupby("scaffold"):
        plt.plot(g["L"], g["answered_acc"], marker="o", label=sc)
    plt.xlabel("Length (L)")
    plt.ylabel("Answered accuracy")
    plt.title("Answered accuracy vs Length (lenient)")
    plt.grid(True, linestyle=":")
    plt.legend()
    outp = out_dir / "answered_acc_vs_L_by_scaffold.png"
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, type=Path,
                    help="Root directory containing run folders with _manifest.json files.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write outputs (default: <in-root>/_analysis).")
    ap.add_argument("--yn-map", choices=["plain","extended"], default="plain",
                    help="Lenient parser: map only yes/no (plain) or include true/false, correct/incorrect (extended).")
    ap.add_argument("--make-plots", action="store_true",
                    help="Emit PNG plots (requires matplotlib).")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.in_root / "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.in_root, yn_map=args.yn_map)
    if not rows:
        print(f"[WARN] No rows loaded from {args.in_root}. Did you point --in-root at the runs parent?")
        return
    df = pd.DataFrame([r.__dict__ for r in rows])
    det_path = out_dir / "detailed_rows.csv"
    df.to_csv(det_path, index=False)
    print(f"[WRITE] {det_path}  (N={len(df)})")

    # Summaries
    df_strict = summarize_strict(df)
    df_strict.to_csv(out_dir / "summary_by_scaffold_L_strict.csv", index=False)
    df_len = summarize_lenient(df)
    df_len.to_csv(out_dir / "summary_by_scaffold_L_lenient.csv", index=False)

    # Paired tables
    write_paired_tables(df, out_dir)

    # Significance (strict)
    df_adj = significance_adjacent_strict(df)
    df_adj.to_csv(out_dir / "significance_adjacent_strict.csv", index=False)
    df_vsbase = significance_vs_baseline_strict(df)
    if not df_vsbase.empty:
        df_vsbase.to_csv(out_dir / "significance_vs_baseline_strict.csv", index=False)

    # Lenient answered-only significance
    df_ansadj = significance_adjacent_answered_only(df)
    df_ansadj.to_csv(out_dir / "significance_adjacent_answered_only.csv", index=False)

    # Plots
    if args.make_plots:
        plot_strict(df_strict, df_adj, df_vsbase if not df_vsbase.empty else pd.DataFrame([]), out_dir)
        plot_compliance(df_strict, out_dir)
        plot_lenient(df_len, out_dir)

    print("[DONE] Analysis complete.")

if __name__ == "__main__":
    main()