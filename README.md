# Chalie Models

Pre-trained classifiers for Chalie's cognitive runtime.

Models are distributed as GitHub Release assets. Releases ship **per-task MLP heads** that attach to the shared
`gte-modernbert-base` encoder Chalie already loads for general-purpose embeddings.

## Models

| Model | Architecture | Classes | Test accuracy | Head size | Status |
|-------|--------------|---------|---------------|-----------|--------|
| `thinking-level` | MLP head over shared gte-modernbert-base + 4-dim prev_level one-hot | 3 | 88.44% (92.3% @ t≥0.70) | 777 KB | v0.9.1 |

## Architecture

All current models share a single frozen encoder plus a per-task 2-layer MLP head.

- **Shared base**: `Alibaba-NLP/gte-modernbert-base` ONNX, pinned to revision
  `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, sha256
  `947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78`. Not shipped
  here — Chalie already loads it for embeddings.
- **Per-task head**: `Linear(input_dim → hidden_dim) → GELU → Dropout(train-only) → Linear(hidden_dim → num_classes)`.
  Weights (W1, b1, W2, b2) stored as float32 in a `<model-name>_head.npz`. The `classifier_meta.json`
  sidecar records `hidden_dim`, `activation`, `base_encoder_sha256`, label order, and (for tasks with
  structural features) the prev_level one-hot layout.
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
└── thinking_level/
    ├── thinking-level_head.npz
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
