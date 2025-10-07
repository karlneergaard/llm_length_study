#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_length_old.py — Rolling-context runner for BoolQ-length evals.

This preserves TRUE multi-turn behavior:
T1 → A1 → T2 → A2 → … → TL → AL, with each assistant reply appended to the context.

What’s new in this revision:
- Input is ONLY --in-final (boolq_final.jsonl).
- Scaffolds now include: baseline, meta, semantic, underspecified, misleading.
- Strict template filenames:
    data/length/{scaffold}/L{L}_{scaffold}.jsonl
  And for 'misleading' we branch by gold (NO fallbacks):
    answer==true  -> data/length/misleading/L{L}_misleading_true.jsonl
    answer==false -> data/length/misleading/L{L}_misleading_false.jsonl
- Deterministic anchor generator for {anchor_a..d} (underspecified, misleading).
- Progress prints so you can see where it’s working/lagging.
- Adds model alias: --model phi4-mini -> microsoft/Phi-4-mini-instruct.
- NEW: --dry-run writes prompts/transcripts without importing torch/transformers.

NOTE: There are NO top-level imports of torch/transformers so --dry-run stays light.

# Example (dry-run; writes prompts + transcripts only):
python run_length_old.py \
  --scaffold misleading \
  --lengths 6 \
  --num 5 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_misleading_L6_dry \
  --model phi4-mini \
  --device mps \
  --dry-run

# Example (writes prompts + responses + JSON):
python run_length.py \
  --scaffold misleading \
  --lengths 6 \
  --num 10 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_misleading_L6 \
  --model phi4-mini \
  --device mps \
  --skip-existing

"""

from __future__ import annotations
import argparse
import json
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Iterable, Any, Optional

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
LENGTH_DIR   = DATA_DIR / "length"

SCAFFOLDS = ("baseline", "meta", "semantic", "underspecified", "misleading")

# Convenience aliases; pass full HF model id to --model to bypass
MODEL_ALIASES = {
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
}

# --------------------------------------------------------------------------------------
# Light IO helpers
# --------------------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {path}:{ln} -> {e}")
    return rows

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# --------------------------------------------------------------------------------------
# Template plumbing
# --------------------------------------------------------------------------------------
PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")

def find_placeholders(s: str) -> List[str]:
    return PLACEHOLDER_RE.findall(s)

def strict_format(template: str, mapping: Dict[str, Any], origin: str) -> str:
    needed = set(find_placeholders(template))
    missing = [k for k in needed if k not in mapping]
    if missing:
        raise KeyError(f"Missing placeholders in {origin}: {missing}")
    try:
        return template.format(**mapping)
    except Exception as e:
        raise RuntimeError(f"Failed formatting {origin}: {e}")

def resolve_template_path(scaffold: str, L: int, gold_bool: bool) -> Path:
    """
    Return the exact template file path for (scaffold, L, gold).
    Strict names only; no fallbacks.
    """
    base = LENGTH_DIR / scaffold
    if scaffold == "misleading":
        fname = f"L{L}_misleading_true.jsonl" if gold_bool else f"L{L}_misleading_false.jsonl"
        path = base / fname
    elif scaffold == "baseline":
        path = base / "L1_baseline.jsonl"
    else:
        # meta | semantic | underspecified
        path = base / f"L{L}_{scaffold}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"[TPL] Missing template: {path}")
    return path

# --------------------------------------------------------------------------------------
# Anchor generator (deterministic; used for underspecified & misleading)
# --------------------------------------------------------------------------------------
def _seed_from_topic_or_question(topic_primary: Optional[str], question: str) -> str:
    topic = (topic_primary or "").strip()
    if topic:
        return topic
    toks = re.findall(r"[A-Za-z0-9]+", question.lower())
    return " ".join(toks[:2]) if toks else "topic"

def _title_head(seed: str) -> str:
    words = seed.strip().split()
    if not words:
        return "Topic"
    return " ".join(w[:1].upper() + w[1:] for w in words[:2])

def _acronym(seed: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", seed)]
    if not words:
        return "SYS"
    if len(words) == 1:
        w = words[0]
        return (w[:4] if len(w) >= 4 else w).upper()
    return "".join(w[0].upper() for w in words)[:4]

_ANCHOR_PATTERNS = [
    ("{head} Reference Council", "{acr}RC"),
    ("{head} Standards Council", "{acr}SC"),
    ("{head} Methods Initiative", "{acr}MI"),
    ("{head} Labeling Guide", "{acr}LG"),
]

def generate_anchors(topic_primary: Optional[str], question: str) -> Dict[str, str]:
    seed = _seed_from_topic_or_question(topic_primary, question)
    head = _title_head(seed)
    acr  = _acronym(seed)
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    order = [(h + i) % len(_ANCHOR_PATTERNS) for i in range(len(_ANCHOR_PATTERNS))]
    names = ["anchor_a", "anchor_b", "anchor_c", "anchor_d"]
    anchors = {}
    for i, name in enumerate(names):
        pi = order[i % len(_ANCHOR_PATTERNS)]
        tpat, apat = _ANCHOR_PATTERNS[pi]
        title = tpat.format(head=head, acr=acr)
        acro  = apat.format(head=head, acr=acr)
        anchors[name] = f"{title} ({acro})"
    return anchors

# --------------------------------------------------------------------------------------
# Placeholder mapping per scaffold
# --------------------------------------------------------------------------------------
def build_placeholder_map(item: dict, scaffold: str) -> Dict[str, Any]:
    question = (item.get("corrected") or item.get("question") or "").strip()
    topic_primary = (item.get("topic_primary") or "").strip()
    topic_related = item.get("topic_related") or []

    mapping: Dict[str, Any] = {
        "QUESTION": question,
        "topic_primary": topic_primary if topic_primary else _seed_from_topic_or_question(topic_primary, question),
        "GOLD": "YES" if bool(item.get("answer")) else "NO",
    }

    if scaffold == "semantic":
        # pad related terms to 4
        rel = [str(x) for x in topic_related]
        rel += ["", "", "", ""]
        mapping.update({
            "related_a": rel[0],
            "related_b": rel[1],
            "related_c": rel[2],
            "related_d": rel[3],
        })

    if scaffold in ("underspecified", "misleading"):
        mapping.update(generate_anchors(topic_primary, question))

    return mapping

# --------------------------------------------------------------------------------------
# Lazy HF import (used ONLY when not --dry-run)
# --------------------------------------------------------------------------------------
def _lazy_hf():
    import torch  # imported only if actually generating
    from transformers import AutoTokenizer, AutoModelForCausalLM
    return torch, AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------------------
# Single-turn generation (appended to rolling context)
# --------------------------------------------------------------------------------------
def generate_reply(
    tok, mdl, prompt_text: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stop_regex: Optional[re.Pattern] = None,
) -> str:
    inputs = tok(prompt_text, return_tensors="pt")
    if mdl.device.type != "cpu":
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Extract only what model added
    if text.startswith(prompt_text):
        gen = text[len(prompt_text):]
    else:
        gen = text
    if stop_regex:
        m = stop_regex.search(gen)
        if m:
            gen = gen[:m.start()]
    return gen.strip()

# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def parse_lengths(s: str, scaffold: str) -> List[int]:
    if scaffold == "baseline":
        return [1]
    if not s:
        return [6, 11, 16, 21]
    vals = sorted(set(int(x) for x in s.split(",")))
    for v in vals:
        if v not in (6, 11, 16, 21):
            raise ValueError(f"Invalid L={v} for scaffold={scaffold} (allowed: 6,11,16,21)")
    return vals

def load_final_items(path: Path, id_include: Optional[List[str]], num: Optional[int]) -> List[dict]:
    items = load_jsonl(path)
    if id_include:
        keep = set(id_include)
        items = [r for r in items if str(r.get("id")) in keep]
    if num is not None and num > 0:
        random.shuffle(items)
        items = items[:num]
    return items

def main():
    ap = argparse.ArgumentParser(description="Rolling-context runner for BoolQ-length evals")
    ap.add_argument("--scaffold", required=True, choices=SCAFFOLDS,
                    help="baseline|meta|semantic|underspecified|misleading")
    ap.add_argument("--lengths", default="",
                    help="Comma-separated. baseline forced to 1; others in {6,11,16,21}")
    ap.add_argument("--in-final", required=True,
                    help="Path to data/boolq_final.jsonl")
    ap.add_argument("--out-root", required=True,
                    help="Output root, e.g., runs/phi4_semantic")
    ap.add_argument("--num", type=int, default=None,
                    help="Use only N items (after id filter)")
    ap.add_argument("--id-include", default="",
                    help="Comma list of ids to include")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if final response file already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write prompts/transcripts without generation")

    # Inference knobs
    ap.add_argument("--model", default="phi4-mini", help="HF id or alias (e.g., phi4-mini)")
    ap.add_argument("--device", default="cpu", help="cpu|cuda|mps")
    ap.add_argument("--dtype", default="float16", help="compatibility flag; not strictly used here")
    ap.add_argument("--max-new-tokens-tasks", type=int, default=48, help="Per-turn cap for turns 1..L-1")
    ap.add_argument("--max-new-tokens-final", type=int, default=16, help="Cap for final turn")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=50, help="Print every N items")

    args = ap.parse_args()

    scaffold = args.scaffold
    lengths = parse_lengths(args.lengths, scaffold)
    in_final = Path(args.in_final)
    out_root = Path(args.out_root)

    if scaffold == "baseline":
        lengths = [1]

    if not in_final.exists():
        raise FileNotFoundError(f"--in-final missing: {in_final}")

    id_include = [x for x in args.id_include.split(",") if x] if args.id_include else None
    items = load_final_items(in_final, id_include, args.num)

    print(f"[SETUP] scaffold={scaffold} lengths={lengths} items={len(items)} out_root={out_root}")

    # HF init (only if not dry-run)
    tok = mdl = None
    if not args.dry_run:
        torch, AutoTokenizer, AutoModelForCausalLM = _lazy_hf()
        model_id = MODEL_ALIASES.get(args.model, args.model)
        print(f"[HF] Loading model: {model_id} on device={args.device}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        if args.device == "mps":
            mdl = mdl.to("mps")
        elif args.device == "cuda":
            mdl = mdl.to("cuda")
        else:
            mdl = mdl.to("cpu")
        print("[HF] Ready.")

    # Stop regex: cut off if the model starts a new header we use
    stop_re = re.compile(r"\n+###\s*(User|Assistant)\b", re.IGNORECASE)

    manifest_rows: List[dict] = []
    N = len(items)

    for i, item in enumerate(items, 1):
        boolq_id = str(item.get("id"))
        gold_bool = bool(item.get("answer"))
        question = (item.get("corrected") or item.get("question") or "").strip()
        topic_primary = item.get("topic_primary")
        domain = item.get("domain")

        if (i == 1) or (i % args.log_interval == 0):
            print(f"[ITEM] {i}/{N} id={boolq_id}")

        for L in lengths:
            # Resolve template file(s)
            tmpl_path = resolve_template_path(scaffold, L, gold_bool)
            print(f"[TPL] L={L} -> {tmpl_path.name}")
            tmpl_rows = load_jsonl(tmpl_path)

            # Build placeholder map for this item
            subst = build_placeholder_map(item, scaffold)

            # Output layout
            run_dir = out_root / f"L{L}" / scaffold
            run_dir.mkdir(parents=True, exist_ok=True)
            base = f"{boolq_id}_L{L}_{scaffold}"
            out_prompt = run_dir / f"{base}.prompt.txt"
            out_resp   = run_dir / f"{base}.response.txt"
            out_json   = run_dir / f"{base}.json"

            if args.skip_existing and out_resp.exists():
                print(f"[SKIP] {base} (response exists)")
                continue

            # Rolling context transcript (string) + structured trace
            rolling = ""
            turns_trace: List[dict] = []
            t_start_all = time.time()

            # Generate per turn
            for tr in sorted(tmpl_rows, key=lambda r: int(r.get("turn", 0))):
                t_num = int(tr.get("turn", 0))
                origin = f"{tmpl_path.name}:turn{t_num}"
                prompt_raw = tr.get("prompt", "")
                prompt_text = strict_format(prompt_raw, subst, origin)

                # Build rolling context prompt with our simple headers
                user_block = f"### User\n{prompt_text}\n"
                full_prompt = f"{rolling}{user_block}### Assistant\n"

                # Which cap to use (final vs intermediate)
                is_final_turn = (t_num == L)
                cap = args.max_new_tokens_final if is_final_turn else args.max_new_tokens_tasks

                print(f"[TURN] id={boolq_id} L={L} t={t_num}/{L} (cap={cap})")
                t0 = time.time()

                if args.dry_run:
                    reply = "(dry-run)"
                    gen_ms = 0.0
                else:
                    reply = generate_reply(
                        tok, mdl, full_prompt,
                        max_new_tokens=cap,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop_regex=stop_re,
                    )
                    gen_ms = (time.time() - t0) * 1000.0

                # Append to rolling transcript
                rolling += f"{user_block}### Assistant\n{reply}\n"

                # Save turn trace
                turns_trace.append({
                    "turn": t_num,
                    "prompt": prompt_text,
                    "reply": reply,
                    "elapsed_ms": round(gen_ms, 1),
                })

            # Persist prompt (all user turns concatenated) and final response (last assistant line)
            prompt_only = "\n".join([f"[T{t['turn']}] {t['prompt']}" for t in turns_trace])
            save_text(out_prompt, prompt_only)

            final_reply = turns_trace[-1]["reply"] if turns_trace else ""
            save_text(out_resp, final_reply)

            meta = {
                "boolq_id": boolq_id,
                "L": L,
                "scaffold": scaffold,
                "template_file": str(tmpl_path),
                "mislead_branch": ("true" if (scaffold == "misleading" and gold_bool) else ("false" if scaffold == "misleading" else "")),
                "gold": "YES" if gold_bool else "NO",
                "question": question,
                "topic_primary": topic_primary,
                "domain": domain,
                "elapsed_s": round(time.time() - t_start_all, 3),
                "out_files": {
                    "prompt": str(out_prompt),
                    "response": str(out_resp),
                    "json": str(out_json),
                },
            }

            save_json(out_json, {"meta": meta, "turns": turns_trace, "transcript": rolling})
            print(f"[DONE] {base} in {meta['elapsed_s']}s; final len={len(final_reply)}")
            manifest_rows.append(meta)

    # Manifest
    manifest_path = out_root / "_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[MANIFEST] {len(manifest_rows)} rows -> {manifest_path}")

if __name__ == "__main__":
    main()
