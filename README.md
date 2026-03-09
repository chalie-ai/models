# Chalie Models

Pre-trained classifiers for Chalie's cognitive runtime.

Models are distributed as GitHub Release assets. Each release contains the full checkpoint directory (safetensors + tokenizer + metadata) for one or more models.

## Models

| Model | Base | Classes | Accuracy | Size | Status |
|-------|------|---------|----------|------|--------|
| `mode-tiebreaker` | SmolLM2-135M | 2 (A/B) | 85.1% eval | 259MB | v0.1.0 |
| `contradiction` | Qwen3.5-0.5B | 5 (A-E) | — | — | Training |

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
