#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
from patsy import dmatrix
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
INPUTS = {
    "phi3": RUNS_DIR / "phi3_rows.csv",
    "phi4": RUNS_DIR / "phi4_rows.csv",
    "deepseek7b": RUNS_DIR / "deepseek7b_rows.csv",
    "qwen7b": RUNS_DIR / "qwen7b_rows.csv",
}
OUT_DIR = RUNS_DIR / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_SCAFFOLDS = ["meta", "semantic", "underspecified", "misleading"]
VALID_L = [6, 11, 16, 21]
ACC_ALL = ["Correct", "Incorrect", "IDK"]  # we drop anything else (e.g., NC)

COLORS = {
    "meta":           "#1f77b4",
    "semantic":       "#2ca02c",
    "underspecified": "#d62728",
    "misleading":     "#9467bd",
}

def prep_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for need in ("condition", "L", "acc_type"):
        if need not in df.columns:
            raise ValueError(f"{path} missing required column: {need}")
    df["scaffold"] = df["condition"].astype(str).str.strip().str.lower()
    df["L"] = pd.to_numeric(df["L"], errors="coerce").astype("Int64")
    df["acc_type"] = df["acc_type"].astype(str).str.strip().str.title()

    df = df[
        df["scaffold"].isin(VALID_SCAFFOLDS)
        & df["L"].isin(VALID_L)
        & df["acc_type"].isin(ACC_ALL)
    ].copy()

    # Drop unused scaffold levels but preserve order
    df["scaffold"] = pd.Categorical(df["scaffold"], categories=VALID_SCAFFOLDS, ordered=False)
    df["L"] = df["L"].astype(int)
    # Keep only present classes (important for clean coding)
    present_acc = [c for c in ACC_ALL if c in df["acc_type"].unique()]
    df["acc_type"] = pd.Categorical(df["acc_type"], categories=present_acc, ordered=True)
    return df

def build_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Design matrix for predictors only, with an intercept.
    Reference scaffold = semantic (treatment coding).
    """
    df_local = df.copy()
    # Make 'semantic' the reference by reordering so semantic is first in Treatment
    # Patsy's Treatment(default reference=first) -> put 'semantic' first
    present_scaffolds = [s for s in VALID_SCAFFOLDS if s in df_local["scaffold"].unique()]
    if "semantic" in present_scaffolds:
        present_scaffolds = ["semantic"] + [s for s in present_scaffolds if s != "semantic"]
    df_local["scaffold"] = pd.Categorical(df_local["scaffold"], categories=present_scaffolds, ordered=False)

    # Build design: intercept + treatment-coded scaffold + L + interaction
    X = dmatrix("1 + C(scaffold, Treatment(reference='semantic')) * L",
                df_local, return_type="dataframe")
    return X, df_local

def fit_model(df: pd.DataFrame, label: str):
    """
    If 3 classes present -> MNLogit; if 2 classes present -> binary Logit.
    Returns (result, classes, is_multinomial, X, y).
    """
    classes = list(df["acc_type"].cat.categories)
    X, df_local = build_X(df)

    if len(classes) == 3:
        # Multinomial: map classes to codes 0..K-1 in the order of categories
        code_map = {c: i for i, c in enumerate(classes)}
        y = df_local["acc_type"].map(code_map).astype(int).to_numpy()
        model = sm.MNLogit(y, X)
        result = model.fit(method="newton", maxiter=200, disp=False)
        summ_path = OUT_DIR / f"{label}_acc_mnlogit_summary.txt"
        with summ_path.open("w", encoding="utf-8") as f:
            f.write(result.summary().as_text())
        print(f"[MNLogit] {label}: classes={classes}  wrote -> {summ_path}")
        return result, classes, True, X, y

    elif len(classes) == 2:
        # Binary Logit: pick positive class = "Incorrect" if present, else the second
        pos = "Incorrect" if "Incorrect" in classes else classes[-1]
        y = (df_local["acc_type"] == pos).astype(int).to_numpy()
        model = sm.Logit(y, X)
        result = model.fit(method="newton", maxiter=200, disp=False)
        summ_path = OUT_DIR / f"{label}_acc_logit_summary.txt"
        with summ_path.open("w", encoding="utf-8") as f:
            f.write(result.summary().as_text())
        print(f"[Logit ] {label}: classes={classes} (positive='{pos}')  wrote -> {summ_path}")
        return result, classes, False, X, y

    else:
        raise RuntimeError(f"{label}: only one class present {classes}; cannot fit a model.")

def predict_grid(result, classes, is_multinomial: bool, df_ref: pd.DataFrame, label: str) -> pd.DataFrame:
    # Use only scaffolds that were present in the data (so the design matches coefficients)
    pres_scaff = [s for s in VALID_SCAFFOLDS if s in df_ref["scaffold"].unique()]
    grid = pd.DataFrame([(s, l) for s in pres_scaff for l in VALID_L], columns=["scaffold", "L"])
    grid["scaffold"] = pd.Categorical(grid["scaffold"], categories=pres_scaff, ordered=False)

    Xg = dmatrix("1 + C(scaffold, Treatment(reference='semantic')) * L",
                 grid, return_type="dataframe")

    if is_multinomial:
        probs = result.predict(Xg)  # shape (n, K)
        probs = pd.DataFrame(probs, columns=classes)
    else:
        # Binary: result.predict gives P(y=1) for the positive class we used.
        p1 = result.predict(Xg)
        pos = "Incorrect" if "Incorrect" in classes else classes[-1]
        neg = [c for c in classes if c != pos][0]
        probs = pd.DataFrame({
            pos: p1,
            neg: 1.0 - p1
        })
        # If one of (Correct/IDK/Incorrect) is missing entirely, keep only present cols.
        probs = probs[[c for c in ["Correct", "Incorrect", "IDK"] if c in probs.columns]]

    out = pd.concat([grid.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    csv_path = OUT_DIR / f"{label}_acc_pred_probs.csv"
    out.to_csv(csv_path, index=False)
    print(f"[PRED ] {label}: wrote predicted probabilities -> {csv_path}")
    return out

def plot_pred(pred: pd.DataFrame, label: str):
    for outcome in ["Correct", "Incorrect", "IDK"]:
        if outcome not in pred.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        for s in sorted(pred["scaffold"].unique(), key=lambda x: VALID_SCAFFOLDS.index(x)):
            sub = pred[pred["scaffold"] == s].sort_values("L")
            ax.plot(sub["L"], sub[outcome], marker="o", linestyle="-",
                    color=COLORS.get(s, "gray"), label=s, linewidth=2.0)
        ax.set_title(f"{label}: Predicted P({outcome}) by length and scaffold")
        ax.set_xlabel("Length (L)")
        ax.set_ylabel(f"P({outcome})")
        ax.set_xticks(VALID_L)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, frameon=True, loc="upper right")
        out_path = OUT_DIR / f"{label}_acc_pred_{outcome}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[PLOT ] {label}: wrote {out_path}")

def main():
    for label, path in INPUTS.items():
        if not path.exists():
            print(f"[WARN] Missing: {path} — skipping.")
            continue
        df = prep_data(path)
        counts = df["acc_type"].value_counts(dropna=False).to_dict()
        scaffolds = df["scaffold"].unique().tolist()
        Ls = sorted(df["L"].unique().tolist())
        print(f"[DATA ] {label}: n={len(df)}  scaffolds={scaffolds}  L={Ls}  acc={counts}")

        try:
            result, classes, is_mn, X, y = fit_model(df, label)
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            continue

        pred = predict_grid(result, classes, is_mn, df, label)
        plot_pred(pred, label)

if __name__ == "__main__":
    main()