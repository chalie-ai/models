# Chalie Models

Pre-trained classifiers for Chalie's cognitive runtime.

Models are distributed as GitHub Release assets. Each release contains the full checkpoint directory (safetensors + tokenizer + metadata) for one or more models.

## Models

| Model | Base | Classes | Accuracy | Size | Status |
|-------|------|---------|----------|------|--------|
| `mode-tiebreaker` | SmolLM2-135M | 2 (A/B) | 85.1% eval | 259MB | v0.1.0 |
| `contradiction` | Qwen2.5-0.5B | 5 (A-E) | — | — | v0.7.0 |
| `thinking-level` | Qwen2.5-0.5B | 3 (A/B/C) | 99.3% eval | 11KB head | v0.8.0 |

## Usage

Download model assets from the [Releases](https://github.com/chalie-ai/models/releases) page and place them in the models directory configured via `MODELS_DIR`.

```
models/
  mode-tiebreaker/
    model.safetensors
    config.json
    tokenizer.json
    tokenizer_config.json
    classifier_meta.json
    ...
```

## Model Details

### mode-tiebreaker (v0.1.0)

Breaks ties when Chalie's deterministic mode router can't decide between two engagement modes. Binary classifier (A = first candidate, B = second candidate).

- **Base model**: HuggingFaceTB/SmolLM2-135M
- **Architecture**: Single-token classifier (logit comparison at last position)
- **Training data**: 35K samples (synthetic + LLM-augmented edge cases)
- **Inference**: Sub-millisecond on CPU via ONNX or native PyTorch
- **Eval accuracy**: 85.1% overall, 100% on ACKNOWLEDGE pairs, 96.3% on ACT/CLARIFY
- **Note**: ACT/RESPOND ties (59.3%) are handled by architectural preference — ACT is the safe default since it can fall back to RESPOND mid-execution

### thinking-level (v0.8.0)

3-class deliberation-depth gate sitting in front of Chalie's ACT loop. Predicts whether the incoming user turn deserves `low` / `medium` / `high` thinking budget.

- **Base model**: Qwen/Qwen2.5-0.5B (shared base — see `qwen2.5-0.5b_base`)
- **Architecture**: Single-token classifier (logit comparison at last position), pruned `lm_head` (3×hidden_dim)
- **Distribution**: Shared base + slim head — release ships only `thinking-level_head.npz` (~11 KB) plus `classifier_meta.json`
- **Training data**: ~3,540 hand-authored seeds × 18 Gemma-expanded variants (~63K samples), group-aware split by `seed_id`
- **Inference**: Sub-millisecond on CPU once the shared base is warm
- **Eval accuracy**: 99.3% on 9,559-sample held-out test set, balanced across classes
- **Input contract**: `[prev=<none|low|medium|high>] <user_turn>\nOptions: A: low | B: medium | C: high\nAnswer:` — byte-exact, see training repo `data/tasks/thinking_level/SIGNALS.md`
- **Classes**:
  - **A → low** — chit-chat / direct lookups / 1-2 tool calls
  - **B → medium** — bounded research / short synthesis / 2-5 ACT cycles
  - **C → high** — multi-tool orchestration / up-front planning / 5+ ACT cycles
- **Confidence threshold**: 0.70 — below that, sticky-fallback to `prev_level` (or `medium` on cold start)
- **Rollout**: Phase 1 shadow mode (log-only) before driving ACT depth
