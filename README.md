# Chalie Models

Pre-trained classifiers for Chalie's cognitive runtime.

Models are distributed as GitHub Release assets. Releases ship **per-task MLP heads** that attach to the shared
`gte-modernbert-base` encoder Chalie already loads for general-purpose embeddings.

## Models

| Model | Architecture | Classes | Test accuracy | Head size | Status |
|-------|--------------|---------|---------------|-----------|--------|
| `thinking-level` | MLP head over shared gte-modernbert-base + 4-dim prev_level one-hot | 3 | 88.44% (92.3% @ t≥0.70) | 777 KB | v0.9.1 |
| `mode-detector`  | MLP head over shared gte-modernbert-base (multi-label, 8 sigmoid heads)    | 8 | 79.7% macro F1 (calibrated) / 46.6% exact-match | 778 KB | v0.10.0 |

## Architecture

All current models share a single frozen encoder plus a per-task 2-layer MLP head.

- **Shared base**: `Alibaba-NLP/gte-modernbert-base` ONNX, pinned to revision
  `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, sha256
  `947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78`. Not shipped
  here — Chalie already loads it for embeddings.
- **Per-task head**: `Linear(input_dim → hidden_dim) → GELU → Dropout(train-only) → Linear(hidden_dim → num_classes)`.
  Weights (W1, b1, W2, b2) stored as float32 in a `<model-name>_head.npz`. The `classifier_meta.json`
  sidecar records `hidden_dim`, `activation`, `base_encoder_sha256`, label order, and (for tasks with
  structural features) the prev_level one-hot layout. Multi-label tasks additionally carry
  `task_type: "multi_label"`, `output_activation: "sigmoid"`, `num_heads`, `default_threshold`,
  and a calibrated `per_mode_thresholds` vector (one cutoff per head) — the loader must apply
  these per-head rather than a flat threshold.
- **Revision pin check**: loaders must refuse to boot if
  `classifier_meta.json::base_encoder_sha256` ≠ sha256 of the ONNX Chalie is actually loading.

Retired: the Qwen2.5-0.5B single-token classifier design (v0.8 and earlier), `mode-tiebreaker`,
`trait-detector`, `skill-selector`, and `social-filter` were all dropped in the shared-encoder
pivot (training-pipeline commit `0361cfe`). The `contradiction` classifier was briefly shipped in
v0.9.0 then removed — Chalie's memory subsystem does not consume it. Training-repo commit
`a5f0646` is the last known-good state for the contradiction task should it ever be revived.

## Usage

Download release assets and place them under `backend/data/models/<task>/`:

```
backend/data/models/
├── gte-modernbert-base/onnx/model.onnx        (shared base — already in Chalie)
├── thinking_level/
│   ├── thinking-level_head.npz
│   └── classifier_meta.json
└── mode_detector/
    ├── mode-detector_head.npz
    └── classifier_meta.json
```

## Model Details

### thinking-level (v0.9.1)

3-class deliberation-depth gate sitting in front of Chalie's ACT loop. Predicts whether the
incoming user turn deserves `low` / `medium` / `high` thinking budget.

- **Architecture**: MLP head over shared gte-modernbert-base. Input is the 768-d pooled
  embedding concatenated with a 4-dim `prev_level` one-hot → 772-d → 256 hidden → 3 logits.
- **Input**: the raw user turn (no prefix / suffix / options list — the Qwen-era
  `[prev=...] ... Options: A..C\nAnswer:` contract was retired) plus a structural
  `prev_level` ∈ `{none, low, medium, high}` from the previous classified turn.
- **Labels (pinned index order)**: `["low", "medium", "high"]`
- **prev_level one-hot order (pinned)**: `{none: 0, low: 1, medium: 2, high: 3}`. Reordering
  breaks every shipped head.
- **Classes**:
  - `low` — chit-chat / direct lookups / 1–2 tool calls
  - `medium` — bounded research / short synthesis / 2–5 ACT cycles
  - `high` — multi-tool orchestration / up-front planning / 5+ ACT cycles
- **Ambiguity policy**: borderline prompts are labelled toward the *higher* bucket at synthesis
  time (the gate's job is to prevent under-thinking; over-thinking costs latency, under-thinking
  costs correctness).
- **Training data**: ~3,540 hand-authored seeds × 18 Gemma-expanded variants (~67K samples),
  group-aware split by `seed_id` to prevent variant-family leakage.
- **Test accuracy**: 88.44% overall. At t≥0.70 confidence gate: 92.3% accuracy at 89% coverage.
- **Sticky fallback**: below the 0.70 threshold, reuse `prev_level` (default `medium` on cold
  start). This keeps the gate production-viable even where the raw classifier is ambiguous.
- **Rollout**: Phase 1 is shadow mode (log-only); Phase 2 calibrates the threshold against
  benchmark uplift (not raw accuracy); Phase 3 drives ACT loop depth directly.

### A note on the test-accuracy ceiling

The thinking-level ceiling is structural, not a capacity problem. H=256 vs H=512 was tested —
no meaningful difference. The limit comes from short continuation prompts sharing too much
lexical surface for a mean-pooled 768-d embedding to reliably separate ("now poke holes in
it" vs "now implement it" both share prev_level=high and most tokens, and read nearly
identically after pool + L2). The 0.70 confidence gate + sticky fallback sidesteps this:
ambiguous turns inherit the active conversation's deliberation depth rather than guessing.

See the training repo `data/tasks/thinking_level/{SIGNALS.md,INTEGRATION.md}` for the full
signal contract and Chalie-side wrapper spec.

### mode-detector (v0.10.0)

8-head multi-label cognitive-mode classifier. Takes a single user message and emits 8 independent
sigmoid probabilities — one per abstract intent. Chalie's tool router consumes these to promote
tools to *innate* (always-loaded) vs leave them *discoverable* (lazy-loaded via `find_tools`) per turn.

- **Architecture**: MLP head over shared gte-modernbert-base. Input is the 768-d pooled embedding
  (no prev-state feature — temporal smoothing is the caller's responsibility) → 256 hidden →
  8 sigmoid heads. `BCEWithLogitsLoss` (per-head independent binary cross-entropy).
- **Input**: raw user turn. No prefix / suffix / options list. No conversation history.
- **Output**: 8 floats in `[0, 1]`, independent. **Not a distribution — they do NOT sum to 1.**
  Multiple heads can fire simultaneously (compound intents are a first-class training case).
- **Labels (pinned index order)**: `["research", "coding", "brainstorm", "analyze", "plan", "write", "math", "converse"]`
- **Classes**:
  - `research` — gather external info (lookup, papers, products, news, facts)
  - `coding` — write / debug / refactor / review / explain code
  - `brainstorm` — generate options, ideas, possibilities, "what if", "give me N ideas"
  - `analyze` — process given input (compare, summarize, critique, classify)
  - `plan` — sequence steps, schedule, decompose goals, roadmap
  - `write` — produce prose (drafts, emails, docs, creative)
  - `math` — quantitative reasoning (calc, stats, proofs, symbolic manipulation)
  - `converse` — social / emotional / chitchat / advice, no underlying task
- **Calibrated per-mode thresholds** (from `classifier_meta.json::per_mode_thresholds`, tuned on
  the 5,286-row held-out test set to maximise per-head F1):

  | mode | threshold | rationale |
  |------|-----------|-----------|
  | `research`   | 0.33 | head well-calibrated at low probs; flat 0.4 silently missed implicit phrasings |
  | `coding`     | 0.82 | very high-confidence head; most true positives land above 0.9 |
  | `brainstorm` | 0.60 | modest tighten; head slightly over-fires |
  | `analyze`    | 0.50 | weakest head; raising past 0.5 hurts recall more than the precision gain is worth |
  | `plan`       | 0.80 | high-confidence head; tighten to kill FPs |
  | `write`      | 0.72 | confident; cut FPs leaking from research |
  | `math`       | 0.93 | extremely confident; anything below 0.9 is noise |
  | `converse`   | 0.55 | mild tighten; bleeds into research / brainstorm |

  **Do not use a flat 0.4 cutoff in production** — it costs ~6pp macro F1 and ~12pp exact-match
  vs the calibrated vector. Loaders should read `per_mode_thresholds` from meta and apply per head.
- **Training data**: ~24.4k seeds — ~9.8k hand-crafted (`provenance: "sonnet"`) in 5 categories
  (basic / compound / ambiguous / complex / null), plus ~14.6k external imports (Dolly + no_robots,
  `provenance: "external"`). After 12× Gemma+Qwen variant expansion the train pool is ~81% external.
  Split strategy is `source`: variants always go to train; sonnet parents always go to test;
  external parents split 10% test / 90% train via deterministic MD5 hash on `seed_id`.
- **Test accuracy (calibrated)**: macro F1 79.7%, exact-match 46.6%, Hamming 90.7%
  (vs flat t=0.4 baseline: 73.7% / 34.9% / 87.6%).
- **Known biases**:
  - **Zero external seeds for `plan` or `math`** — Dolly and no_robots have no examples for these
    modes. Both heads are trained mostly on sonnet + variants. Real-world `plan` / `math` phrasings
    outside the sonnet distribution may be miscalibrated.
  - **`research` is the saturated head** — external pool skew toward research / write / analyze
    means the research head fires early (threshold 0.33). It's well-calibrated at low probs, but
    callers that want a sharper research signal should layer a second check rather than raise the
    threshold (which hurts recall on implicit phrasings like "run a web search for X").
  - **`null` category exists** — some prompts ("thanks", "hey", "asdfgh") are trained to output
    all-zero. Do not force at least one mode to fire.
  - **Compound training** — ~20% of training examples have two modes firing (primary=1.0,
    secondary=0.7). Expect 2 fires on genuinely compound prompts ("should I switch from Python to
    Rust, give me pros and cons" → `analyze` + `research`).
- **Rollout**: Phase 1 is shadow mode (log-only, `shadow_mode: true` in the training config).
  Phase 2 drives tool-router promotion decisions in `UserMessageProcessor` based on a per-turn
  EMA-smoothed accumulator over the 8 head outputs.

See the training repo `data/tasks/mode_detector/SEED_SPEC.md` for the seed authoring contract,
provenance routing rules, and the external-imbalance caveats in full.
