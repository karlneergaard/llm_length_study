# stims_length.py
import argparse
import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # folder containing this script

def load_df(path: Path) -> pd.DataFrame:
    """Robust JSONL loader: line-by-line to catch any malformed lines."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                snippet = (s[:160] + "...") if len(s) > 160 else s
                raise ValueError(
                    f"Malformed JSON on line {i} in {path}:\n  {snippet}\n  ^ {e}"
                ) from e
    df = pd.DataFrame(rows)
    # Validate required columns
    for col in ("answer", "domain"):
        if col not in df.columns:
            raise ValueError(f"{path} must contain '{col}' column. Got: {list(df.columns)}")
    return df

def analyze_file(df: pd.DataFrame, label: str):
    # Keep only rows where 'answer' is a real boolean (True/False)
    mask_bool = df["answer"].map(lambda x: isinstance(x, bool))
    view = df.loc[mask_bool, ["domain", "answer"]].copy()
    view["domain"] = view["domain"].astype(str)

    # 1) overall counts of True/False
    overall = (
        view.groupby("answer", dropna=False)
            .size().reset_index(name="count")
    )
    overall.insert(0, "file", label)

    # 2) per-domain counts of True/False
    by_domain = (
        view.groupby(["domain", "answer"], dropna=False)
            .size().reset_index(name="count")
    )
    by_domain.insert(0, "file", label)

    return overall, by_domain

def main():
    ap = argparse.ArgumentParser(description="Counts of boolean answers overall and by domain for BoolQ subsets.")
    ap.add_argument("--false-path", default=str(SCRIPT_DIR / "boolq_final_false.jsonl"))
    ap.add_argument("--idk-path",   default=str(SCRIPT_DIR / "boolq_final_idk.jsonl"))
    ap.add_argument("--true-path",  default=str(SCRIPT_DIR / "boolq_final_true.jsonl"))
    ap.add_argument("--out-overall", default=str(SCRIPT_DIR / "stims_overall_answer_counts.csv"))
    ap.add_argument("--out-domain",  default=str(SCRIPT_DIR / "stims_answer_counts_by_domain.csv"))
    args = ap.parse_args()

    files = [
        ("final_false", Path(args.false_path)),
        ("final_idk",   Path(args.idk_path)),
        ("final_true",  Path(args.true_path)),
    ]

    overall_frames, domain_frames = [], []

    for label, path in files:
        print(f"[LOAD] {label}: {path}")
        df = load_df(path)
        ov, bd = analyze_file(df, label)
        overall_frames.append(ov)
        domain_frames.append(bd)

    overall_out = pd.concat(overall_frames, ignore_index=True)
    domain_out  = pd.concat(domain_frames,  ignore_index=True)

    Path(args.out_overall).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_domain).parent.mkdir(parents=True, exist_ok=True)
    overall_out.to_csv(args.out_overall, index=False)
    domain_out.to_csv(args.out_domain, index=False)

    # console peek
    print(f"Wrote overall counts -> {args.out_overall}")
    if not overall_out.empty:
        print(overall_out.pivot(index="file", columns="answer", values="count").fillna(0).astype(int))
    else:
        print("(no rows with boolean answers)")

    print()
    print(f"Wrote per-domain counts -> {args.out_domain}")
    if not domain_out.empty:
        with pd.option_context("display.max_rows", 20, "display.max_columns", 6):
            print(domain_out.sort_values(["file","domain","answer"]).head(20))
    else:
        print("(no rows with boolean answers by domain)")

if __name__ == "__main__":
    main()