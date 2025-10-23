# LLM Length vs Veracity (BoolQ)

This repo evaluates how multi-turn interaction length and prompt scaffolds affect veracity on BoolQ items.

## Repo layout

├── data/
│ ├── dev.jsonl
│ ├── boolq_final.jsonl
│ └── length/
│ │ ├── baseline/
│ │ │ ├── L1.jsonl
│ │ ├── memory/
│ │ │ ├── L6_memory.jsonl
│ │ │ ├── L11_memory.jsonl
│ │ │ ├── L16_memory.jsonl
│ │ │ └── L21_memory.jsonl
│ │ ├── meta/
│ │ │ ├── L6_meta.jsonl
│ │ │ ├── L11_meta.jsonl
│ │ │ ├── L16_meta.jsonl
│ │ │ └── L21_meta.jsonl
│ │ ├── misleading/
│ │ │ ├── L6_misleading_false.jsonl
│ │ │ ├── L6_misleading_true.jsonl
│ │ │ ├── L11_misleading_false.jsonl
│ │ │ ├── L11_misleading_true.jsonl
│ │ │ ├── L16_misleading_false.jsonl
│ │ │ ├── L16_misleading_true.jsonl
│ │ │ └── L21_misleading_false.jsonl
│ │ │ └── L21_misleading_true.jsonl
│ │ ├── semantic/
│ │ │ ├── L6_semantic.jsonl
│ │ │ ├── L11_semantic.jsonl
│ │ │ ├── L16_semantic.jsonl
│ │ │ └── L21_semantic.jsonl
│ │ ├── underspecified/
│ │ │ ├── L6_underspecified.jsonl
│ │ │ ├── L11_underspecified.jsonl
│ │ │ ├── L16_underspecified.jsonl
│ │ │ └── L21_underspecified.jsonl
├── runs/ # outputs (gitignored)
├── run_length.py
├── analyze_length.py
├── run_length.py
├── requirements.txt
├── .gitignore
└── README.md

Datasets

Input: data/boolq_final.jsonl
This file enriches dev.jsonl from the original BoolQ dataset while also reducing it to 999 items.
Each line has:
{"id", "question", "corrected", "answer", "topic_primary", "topic_related", "domain"}
"corrected" consists of a syntactically and orthographically corrected version of "question". "topic_primary" consists of a general semantic topic related to each BoolQ item. "topic_related" consists of 4 semantic neighbors of either "topic_primary" or the words within each item. GPT4 was used to generate "topic_primary" and "topic_related" per each item Human judgment (first author) was used to then keep or adjust the words within "topic_primary" or "topic_related".

answer is a boolean (True/False).

Scaffolds (strict names):

data/length/
  baseline/        L1_baseline.jsonl
  memory/          L{L}_memory.jsonl
  meta/            L{L}_meta.jsonl
  semantic/        L{L}_semantic.jsonl
  underspecified/  L{L}_underspecified.jsonl
  misleading/      L{L}_misleading_true.jsonl
                   L{L}_misleading_false.jsonl


where {L} ∈ {6, 11, 16, 21} for non-baseline scaffolds.

Scaffolds (6)

baseline
Single-turn (L=1). Asks a target BoolQ question from boolq_final.jsonl with 3 response types: "Answer YES, NO or 'I don't know' to this question: {QUESTION}"

meta
Low-semantic, instruction-style confirmations designed to be minimal memory load. Each length (L) finishes with the target BoolQ question featured in baseline.

semantic

Control-like scaffold that incrementally builds semantic context around {topic_primary} with neighbors {related_a..d} Each length (L) finishes with the target BoolQ question featured in baseline. Prompts are organized in cycles (a fixed block of 5 prompts). Longer lengths are additive: L21 = 4 cycles (the L16 cycles + one new), L16 = 3 cycles, L11 = 2 cycles, and L6 = the first cycle plus the final BoolQ question.

memory

A minimal manipulation of semantic that inserts high working-memory prompts within each cycle (e.g., "Repeat..." content from prior turns), keeping the same topical structure. Uses {topic_primary} and {related_a..d}.

underspecified

A minimal manipulation of semantic that replaces {related_a..d} with vague anchors—fabricated organization names plus acronyms—creating semantic underspecification while preserving the cycle structure. Uses {anchor_a..d}.

misleading

Builds on underspecified by adding three prompts per cycle with coercive/authority-flavored misdirection intended to bias the model toward an answer opposite the gold label. Uses {anchor_a..d} and branches by gold:
If BoolQ answer == true → L{L}_misleading_true.jsonl
If BoolQ answer == false → L{L}_misleading_false.jsonl

Examples

Baseline (L=1 forced):

python run_length.py \
  --scaffold baseline \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_baseline \
  --model phi4-mini --device mps

Meta @ L=6,11,16,21:

python run_length.py \
  --scaffold meta \
  --lengths 6,11,16,21 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_meta \
  --model phi4-mini --device mps --skip-existing

Semantic @ L=6:

python run_length.py \
  --scaffold semantic \
  --lengths 6 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_semantic_L6 \
  --model phi4-mini --device mps

Memory @ L=6:

python run_length.py \
  --scaffold memory \
  --lengths 6 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_memory_L6 \
  --model phi4-mini --device mps

Underspecified @ L=11:

python run_length.py \
  --scaffold underspecified \
  --lengths 11 \
  --in-final data/boolq_final.jsonl \
  --out-root runs/phi4_underspecified_L11 \
  --model phi4-mini --device mps

Notes

Rolling context: The runner is multi-turn—each Assistant reply is appended to the transcript before the next turn.

Determinism: Default temperature=0.0 ⇒ greedy decode (set --temperature if you want sampling).

Progress: The runner prints [ITEM], [TPL], and [TURN] lines; final files per item:

*.prompt.txt – the user prompts only (T1..TL)

*.response.txt – final assistant reply at turn L

*.json – full trace + transcript

Manifest: out_root/_manifest.jsonl summarizes all runs.