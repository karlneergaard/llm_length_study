# LLM Length vs Veracity (BoolQ)

**Goal.** Measure how **interaction length** (rolling multi-turn dialogues) affects **veracity** on BoolQ. We run two scaffolds:
- **info-light**: procedural/meta prompts only (no domain facts),
- **info-rich**: short, domain-expository prompts using the primary topics of target Boolq questions and related semantic neighbors.

Deterministic decoding (temperature=0)

---

## Repo layout

├── data/
│ ├── dev.jsonl
│ ├── boolq_enriched.jsonl
│ └── length/
│ │ ├── light/
│ │ │ ├── L1_info-light.jsonl
│ │ │ ├── L3_info-light.jsonl
│ │ │ ├── L6_info-light.jsonl
│ │ │ ├── L9_info-light.jsonl
│ │ │ └── L12_info-light.jsonl
│ │ ├── rich/
│ │ │ ├── L1_info-rich.jsonl
│ │ │ ├── L3_info-rich.jsonl
│ │ │ ├── L6_info-rich.jsonl
│ │ │ ├── L9_info-rich.jsonl
│ │ │ └── L12_info-rich.jsonl
├── runs/ # outputs (gitignored)├── run_length.py
├── analyze_length.py
├── run_length.py
├── requirements.txt
├── .gitignore
└── README.md

**Asset format.** Each `L*_info-*.jsonl` has exactly _L_ lines:
```json
Info-light L12 example:
{"length":12,"scaffold":"light","turn":1,"prompt":"..."}
...
{"length":12,"scaffold":"light","turn":12,"prompt":"Answer YES or NO: {QUESTION}"}
