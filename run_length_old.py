#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_length.py — Length vs Veracity (BoolQ) runner, rolling context only.

Assets:
  data/length/baseline/L{L}.jsonl          # each line: {"length":L,"turn":k,"prompt":"..."}
  data/length/light/L{L}_info-light.jsonl   # each line: {"length":L,"scaffold":"light","turn":k,"prompt":"..."}
  data/length/rich/L{L}_info-rich.jsonl     # each line: {"length":L,"scaffold":"rich","turn":k,"prompt":"..."}

Usage: change scaffold and out-root as needed
  # If scaffold is baseline, lengths is always 1
  python run_length.py \
    --scaffold baseline \
    --lengths 1 \
    --in-dev data/dev.jsonl \
    --in-enriched data/boolq_enriched.jsonl \
    --out-root runs/phi3 \
    --id-include dev_0021-dev_0022 --skip-existing \
    --model microsoft/Phi-3-mini-4k-instruct \
    --device mps --dtype float32 \
    --max-new-tokens-tasks 64 --max-new-tokens-final 96 \
    --temperature 0 --stop-seq "### User" \
      
  # If scaffold is light or rich, lengths can be multiple values but not 1
  python run_length.py \
    --scaffold rich \
    --lengths 41 \
    --in-dev data/dev.jsonl \
    --in-enriched data/boolq_enriched.jsonl \
    --out-root runs/phi3 \
    --num 20 \
    --model microsoft/Phi-3-mini-4k-instruct \
    --device mps --dtype float32 \
    --max-new-tokens-tasks 64 --max-new-tokens-final 96 \
    --temperature 0 --stop-seq "### User" \

# Example to run only a few specific IDs:
    --id-include dev_0003-dev_0020 --skip-existing \
# or a larger range:
    --num 100 \
"""

import argparse, json, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------
# Small IO helpers
# -----------------------
def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def join_dev_with_enriched(dev: List[dict], enr: List[dict]) -> List[Tuple[dict, dict]]:
    idx = {e.get("question"): e for e in enr if isinstance(e.get("question"), str)}
    out = []
    for d in dev:
        q = d.get("question")
        if isinstance(q, str) and q in idx:
            out.append((d, idx[q]))
    return out

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sanitize_id(s: Optional[str], fallback: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s).strip("_")
    return s if s else fallback

# -----------------------
# ID include parsing
# -----------------------
def expand_id_spec(spec: Optional[str]) -> set:
    """
    'dev_0063-dev_0100,dev_0123' -> {'dev_0063',...,'dev_0100','dev_0123'}
    """
    out = set()
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            m1 = re.match(r"^(.*?)(\d+)$", a)
            m2 = re.match(r"^(.*?)(\d+)$", b)
            if m1 and m2 and m1.group(1) == m2.group(1):
                prefix = m1.group(1)
                start, end = int(m1.group(2)), int(m2.group(2))
                width = max(len(m1.group(2)), len(m2.group(2)))
                for n in range(min(start, end), max(start, end) + 1):
                    out.add(f"{prefix}{n:0{width}d}")
            else:
                out.add(a); out.add(b)
        else:
            out.add(part)
    return out

# -----------------------
# Asset loader (one JSONL per length)
# -----------------------
def load_length_file(base_dir: Path, scaffold: str, L: int) -> List[dict]:
    if scaffold not in ("baseline", "light", "rich"):
        raise ValueError("scaffold must be 'baseline', 'light', or 'rich'")

    # Enforce scaffold/length rules here too (defensive)
    if scaffold == "baseline" and L != 1:
        raise ValueError("Baseline scaffold only supports L=1.")
    if scaffold in ("light", "rich") and L == 1:
        raise ValueError(f"{scaffold} scaffold does not include L=1 (use baseline for L=1).")

    # Candidate filenames in priority order
    if scaffold == "baseline":
        candidates = [
            base_dir / "baseline" / "L1_info-baseline.jsonl",
            base_dir / "baseline" / "L1_baseline.jsonl",
            base_dir / "baseline" / "L1.jsonl",
        ]
    else:
        candidates = [
            base_dir / scaffold / f"L{L}_info-{scaffold}.jsonl",
            base_dir / scaffold / f"L{L}_{scaffold}.jsonl",
            base_dir / scaffold / f"L{L}.jsonl",
        ]

    for c in candidates:
        if c.exists():
            rows = read_jsonl(c)
            if len(rows) != L:
                raise ValueError(f"{c} expected {L} lines (turns) but found {len(rows)}")
            rows.sort(key=lambda r: r.get("turn", 0))
            return rows

    # Fallback: any file starting with L{L}_ under the scaffold folder
    for p in (base_dir / scaffold).glob(f"L{L}_*.jsonl"):
        rows = read_jsonl(p)
        rows.sort(key=lambda r: r.get("turn", 0))
        return rows

    raise FileNotFoundError(f"No asset file found for scaffold={scaffold} length={L} under {base_dir/scaffold}")

# -----------------------
# Placeholder substitution for RICH prompts
# -----------------------
def build_placeholder_map(enriched: dict) -> Dict[str, str]:
    topic = enriched.get("topic_primary") or enriched.get("topic") or ""
    related = enriched.get("topic_related") or []
    rel = [str(x) for x in related if isinstance(x, (str, int, float))]

    def get_rel(idx: int, fallback: str = "") -> str:
        if idx < len(rel):
            return rel[idx]
        return fallback or (rel[-1] if rel else topic or "")

    mapping = {
        # lower-case variants
        "{topic_primary}": topic,
        "{related_a}": get_rel(0),
        "{related_b}": get_rel(1),
        "{related_c}": get_rel(2),
        "{related_d}": get_rel(3),
        "{related_e}": get_rel(4),
        "{related_f}": get_rel(5),
        # UPPER-case variants
        "{TOPIC_PRIMARY}": topic,
        "{RELATED_A}": get_rel(0),
        "{RELATED_B}": get_rel(1),
        "{RELATED_C}": get_rel(2),
        "{RELATED_D}": get_rel(3),
        "{RELATED_E}": get_rel(4),
        "{RELATED_F}": get_rel(5),
        # question
        "{QUESTION}": enriched.get("corrected") or enriched.get("question") or "",
    }
    return mapping

def substitute_placeholders(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for k, v in mapping.items():
        out = out.replace(k, str(v))
    return out

# -----------------------
# Minimal HF causal LM runner
# -----------------------
class ModelRunner:
    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "auto"):
        self.model_name = model_name
        self.device_mode = device.lower()
        self.dtype_mode = dtype.lower()
        if model_name in MODEL_REGISTRY:
            cfg = MODEL_REGISTRY[model_name]
            self.backend = cfg["backend"]; model_id = cfg["model_id"]
        else:
            self.backend = "hf_causal"; model_id = model_name
        if self.backend != "hf_causal":
            raise ValueError(f"Unsupported backend: {self.backend}")
        self._init_hf_causal(model_id)

    def _pick_torch_dtype(self, torch):
        if self.dtype_mode == "float32": return torch.float32
        if self.dtype_mode == "float16": return torch.float16
        if self.dtype_mode == "bfloat16": return torch.bfloat16
        if self.device_mode in ("cuda","auto"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _init_hf_causal(self, model_id: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = self._pick_torch_dtype(torch)
        if self.device_mode == "cpu":
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            self.model.to(self.device); self._input_device = self.device
        elif self.device_mode == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            self.device = torch.device("cuda")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            self.model.to(self.device); self._input_device = self.device
        elif self.device_mode == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available.")
            self.device = torch.device("mps")
            if self.dtype_mode == "auto":
                dtype = torch.float32  # safer on MPS
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            self.model.to(self.device); self._input_device = self.device
        elif self.device_mode == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
            self.device = "auto"; self._input_device = self._infer_first_device_from_map()
        else:
            raise ValueError(f"Unknown --device '{self.device_mode}'")
        self.model.eval()

    def _infer_first_device_from_map(self):
        dm = getattr(self.model, "hf_device_map", None)
        if isinstance(dm, dict):
            for v in dm.values():
                if isinstance(v, str) and v != "disk":
                    return v
        return "cpu"

    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0,
                 stop: Optional[List[str]] = None) -> str:
        import torch
        print(f"[DEBUG] Generate called: prompt_len={len(prompt)}, max_tokens={max_new_tokens}")
        with torch.inference_mode():
            print("[DEBUG] Tokenizing input...")
            enc = self.tok(prompt, return_tensors="pt")
            print(f"[DEBUG] Input tokens: {enc['input_ids'].shape}")
            inputs = enc.to(self._input_device) if self.device != "auto" else {k: v.to(self._input_device) for k, v in enc.items()}
            print("[DEBUG] About to call model.generate...")
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False if (temperature is None or temperature == 0) else True,
                temperature=temperature if (temperature and temperature > 0) else 1.0,
                pad_token_id=self.tok.pad_token_id,
            )[0]
        print("[DEBUG] Generation complete, decoding...")
        in_len = inputs["input_ids"].shape[-1]
        gen_only_ids = output_ids[in_len:]
        reply = self.tok.decode(gen_only_ids, skip_special_tokens=True).strip()
        if stop:
            for token in stop:
                i = reply.find(token)
                if i != -1:
                    reply = reply[:i].strip()
                    break
        return reply

MODEL_REGISTRY = {
    "phi2": {"backend": "hf_causal", "model_id": "microsoft/phi-2"},
    "phi3mini": {"backend": "hf_causal", "model_id": "microsoft/Phi-3-mini-4k-instruct"},
}

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Length vs Veracity runner (rolling context).")
    ap.add_argument("--scaffold", choices=["baseline","light","rich"], required=True)
    ap.add_argument("--lengths", required=True, help="Comma list of lengths, e.g., '1' or '3,6,9,12'")
    ap.add_argument("--in-dev", required=True)
    ap.add_argument("--in-enriched", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num", type=int, default=None, help="Cap number of items after filtering")
    ap.add_argument("--id-include", default=None, help="Comma list/ranges of BoolQ ids, e.g. 'dev_0001-dev_0100'")
    ap.add_argument("--skip-existing", action="store_true")
    # model/decoding
    ap.add_argument("--model", default="phi3mini", help="Key or HF model id")
    ap.add_argument("--device", choices=["cpu","cuda","mps","auto"], default="mps")
    ap.add_argument("--dtype", choices=["auto","float32","float16","bfloat16"], default="float16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens-tasks", type=int, default=64)
    ap.add_argument("--max-new-tokens-final", type=int, default=96)
    ap.add_argument("--stop-seq", default="### User")
    # assets root (allow override)
    ap.add_argument("--assets-root", default="data/length", help="Base folder for length assets.")
    args = ap.parse_args()

    # Parse lengths
    try:
        lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    except Exception:
        raise SystemExit("--lengths must be comma-separated integers, e.g., '1' or '3,6,9,12'")

    # Enforce scaffold/length policy
    if args.scaffold == "baseline":
        if set(lengths) != {1}:
            raise SystemExit("For scaffold=baseline you must set --lengths 1 (and only 1).")
    else:
        if 1 in lengths:
            raise SystemExit(f"For scaffold={args.scaffold}, remove L=1 (run baseline separately).")

    dev = read_jsonl(Path(args.in_dev))
    enr = read_jsonl(Path(args.in_enriched))
    joined = join_dev_with_enriched(dev, enr)
    if not joined:
        raise SystemExit("No joined items; check that dev.jsonl and boolq_enriched.jsonl share identical 'question' text.")

    # Filter by id if requested
    items = joined
    if args.id_include:
        wanted = expand_id_spec(args.id_include)
        items = [(d,e) for (d,e) in items if (e.get("id") in wanted) or (d.get("id") in wanted)]
        print(f"[FILTER] id-include -> {len(items)} items")

    # Cap by --num
    if args.num is not None and args.num > 0:
        items = items[: args.num]
    #Debug
    print(f"[DEBUG] Loaded {len(dev)} dev items, {len(enr)} enriched items")
    print(f"[DEBUG] After joining: {len(joined)} items")
    print(f"[DEBUG] After filtering: {len(items)} items")
    if items:
        first_item = items[0]
        print(f"[DEBUG] First item ID: {first_item[1].get('id', 'unknown')}")

    # Load model
    print(f"[INFO] Loading model {args.model} on {args.device} ({args.dtype})")
    runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
    # Warmup
    print("[DEBUG] Model loaded, starting warmup...")
    _ = runner.generate("### User\nok\n### Assistant\n", max_new_tokens=1, temperature=0.0, stop=None)
    print("[DEBUG] Warmup complete")

    # Stop list (trim reply only)
    stop_list = None
    if args.stop_seq and args.stop_seq.strip():
        tok = args.stop_seq.strip()
        stop_list = ["\n" + tok, tok]

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    assets_root = Path(args.assets_root)

    manifest = {
        "mode": "length",
        "scaffold": args.scaffold,
        "lengths": lengths,
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "decoding": {"temperature": args.temperature,
                     "max_new_tokens_tasks": args.max_new_tokens_tasks,
                     "max_new_tokens_final": args.max_new_tokens_final,
                     "stop_seq": args.stop_seq},
        "assets_root": str(assets_root.resolve()),
        "items": [],
    }

    print(f"[DEBUG] Starting processing of {len(items)} items for lengths {lengths}")
    for i, (drow, erow) in enumerate(items):
        print(f"[DEBUG] Processing item {i+1}/{len(items)}: {erow.get('id', 'unknown')}")
        q_raw = drow.get("question", "")
        q_corr = erow.get("corrected") or q_raw
        boolq_id_raw = erow.get("id") or drow.get("id")
        boolq_id = sanitize_id(boolq_id_raw, "unknown")

        enriched_meta = {
            "id": boolq_id_raw,
            "question": q_raw,
            "corrected": q_corr,
            "topic_primary": erow.get("topic_primary"),
            "topic_related": erow.get("topic_related") or [],
        }
        ph_map = build_placeholder_map(enriched_meta)

        for L in lengths:
            print(f"[DEBUG] Processing length L={L} for item {boolq_id}")
            # Output path
            fname = f"{boolq_id}_len_L{L}_{args.scaffold}.json"
            out_path = out_root / fname
            if args.skip_existing and out_path.exists():
                print(f"[SKIP] {fname} (exists)")
                manifest["items"].append({"boolq_id": boolq_id, "length": L, "path": str(out_path.relative_to(out_root)), "skipped": True})
                continue

            # Load assets
            turns = load_length_file(assets_root, args.scaffold, L)
            print(f"[DEBUG] Loaded {len(turns)} turns for L={L}")
            # Build rolling conversation
            transcript = []
            rolling = ""
            t_start = time.time()
            for t in range(1, L + 1):
                print(f"[DEBUG] Processing turn {t}/{L}")
                entry = turns[t-1]
                prompt_tpl = entry.get("prompt", "")
                print(f"[DEBUG] Turn {t} prompt template length: {len(prompt_tpl)} chars")
                # Substitute placeholders
                if args.scaffold == "rich":
                    prompt_text = substitute_placeholders(prompt_tpl, ph_map)
                else:
                    # baseline & light: only final turn may contain {QUESTION};
                    # replace if present (harmless no-op for earlier turns)
                    prompt_text = substitute_placeholders(prompt_tpl, {"{QUESTION}": q_corr})

                # Construct prompt with rolling history
                full_prompt = f"{rolling}### User\n{prompt_text}\n### Assistant\n"
                is_final = (t == L)
                cap = args.max_new_tokens_final if is_final else args.max_new_tokens_tasks

                print(f"[DEBUG] Turn {t} full prompt length: {len(full_prompt)} chars")
                print(f"[DEBUG] About to generate for turn {t}, max_tokens={cap}")

                reply = runner.generate(full_prompt, max_new_tokens=cap,
                                        temperature=args.temperature, stop=stop_list)

                print(f"[DEBUG] Turn {t} generation complete, reply length: {len(reply)} chars")

                transcript.append({"turn": t, "role": "user", "prompt": prompt_text})
                transcript.append({"turn": t, "role": "model", "reply": reply})

                rolling += f"### User\n{prompt_text}\n### Assistant\n{reply}\n"

            elapsed = time.time() - t_start

            obj = {
                "mode": "length",
                "scaffold": args.scaffold,
                "length": L,
                "boolq_id": boolq_id,
                "question": q_corr,
                "gold_answer": drow.get("answer"),
                "enriched": {"topic_primary": enriched_meta["topic_primary"],
                             "topic_related": enriched_meta["topic_related"]},
                "model": {"id": args.model, "device": args.device, "dtype": args.dtype},
                "decoding": {"temperature": args.temperature,
                             "max_new_tokens_tasks": args.max_new_tokens_tasks,
                             "max_new_tokens_final": args.max_new_tokens_final,
                             "stop_seq": args.stop_seq},
                "timing_s": round(elapsed, 3),
                "transcript": transcript,
            }
            write_json(out_path, obj)
            manifest["items"].append({"boolq_id": boolq_id, "length": L, "path": str(out_path.relative_to(out_root))})
            print(f"[LEN] id={boolq_id} L={L} {args.scaffold} -> {fname} ({elapsed:.2f}s)")

    write_json(out_root / "_manifest.json", manifest)
    print(f"[DONE] Wrote outputs to {out_root.resolve()}")
    print(f"[STATS] items={len(items)} lengths={lengths} scaffold={args.scaffold} model={args.model} device={args.device}")
    if manifest["items"]:
        done = sum(1 for r in manifest["items"] if not r.get("skipped"))
        print(f"[FILES] {done} new, {len(manifest['items']) - done} skipped")

if __name__ == "__main__":
    main()
