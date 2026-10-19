# Theseus cross-family GSM8K transfer — experiment log

Fill this in as you actually run each step on the server (this file is a log of what happened,
not a plan — the plan/recipe with full config files lives in
`docs/theseus_cross_family_runbook.md`). Keep every entry even for failed/aborted runs; a
diagnostic-explained failure is still a result.

## Run metadata

- Date started:
- Git commit (`git rev-parse HEAD`):
- Source checkpoint (base): 
- Source checkpoint (RLVR/fine-tuned): 
- Source `model_type` / shape (layers, hidden, intermediate):
- Target base: `meta-llama/Llama-3.2-1B`
- Target weights (`target_base_ckpt`, your UltraChat SFT path):
- GPU/host:

## Step 1 — environment

- Command:
  ```bash
  uv sync --extra dev
  uv run pytest tests/test_theseus_layer_filter.py tests/test_theseus_alignment_diagnostics.py tests/test_theseus_rebase.py -q
  ```
- Result (pass/fail count):
- Notes:

## Step 2 — shape check + family adapter

- Command:
  ```bash
  python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('<source>'); print(c.model_type, c.num_hidden_layers, c.hidden_size, c.intermediate_size)"
  ```
- Output:
- Adapter needed? (y/n, name if added):
- Module tree spot-check (`self_attn.q_proj` etc. present as expected?):

## Step 3 — re-baseline target floor (no transport)

- Command:
  ```bash
  uv run python -m merge_and_rebase.eval.llm_rebase \
    --config configs/examples/reference/llm/theseus_gqa.yaml \
    --override method.name=identity \
    --override target_base_ckpt=<path> \
    --override harness_tasks='["gsm8k"]'
  ```
- GSM8K (target floor, replaces the old 36.2%):
- Notes:

## Step 4 — Experiment 0: sanity checks

| Check | Expected | Observed | Pass? |
|---|---|---|---|
| 0a shape audit (layers/hidden/intermediate match) | 16/2048/8192 both sides | | |
| 0b task-vector norms nonzero per layer | all > 0 | | |
| 0b norm distribution shape | larger mid/late layers | | |
| 0c calibration forward pass: NaN/Inf | none | | |
| 0d norm_ratio at one test layer | ≈ 1.0 | | |

- Blocking issues found (if any):

## Step 5 — Experiment 1: diagnostics (no merging)

- Command/script used:
- CSV path:
- Per-layer summary (fill from CSV, abbreviated):

| Layer | module | recon_error | rank_90 | fidelity_score |
|---|---|---|---|---|
| | | | | |

- Overall shape across depth (U-shaped? flat-low? flat-high?):
- Stop-condition check (rank_90 > 1500 everywhere?): 
- Candidate layer sets for Experiment 3 (variant A/B/C):
  - A (fidelity > 0.5):
  - B (recon_error < 0.4):
  - C (top-K by fidelity, K=4/8/12):

## Step 6 — Experiment 2: full uniform transport, alpha sweep

| alpha | GSM8K | wall-clock | coherent output? (spot-check) | notes |
|---|---|---|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.2 | | | | |
| 0.3 | | | | |
| 0.5 | | | | |
| 0.7 | | | | |
| 1.0 | | | | |

- Best alpha:
- Did any alpha clear the target floor from Step 3?

## Step 7 — Experiment 3: selective transport

| Variant | layers included | alpha | GSM8K | vs. uniform at same alpha |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C (K=4) | | | | |
| C (K=8) | | | | |
| C (K=12) | | | | |
| attn-only vs attn+MLP | | | | |

- Does selective beat uniform? (the core finding, independent of absolute GSM8K)

## Step 8 — final comparison

| Method | GSM8K | Notes |
|---|---|---|
| Target floor (Step 3) | | |
| OPD 1000 steps | 26.5% | existing baseline |
| HeteroFusion | 7.2% | existing baseline |
| Theseus full (best alpha) | | |
| Theseus selective (best config) | | |

## Interpretation

- Outcome band (strong / moderate / weak / failure, per the runbook's success bands):
- One-paragraph explanation tying the GSM8K result back to the Experiment 1 diagnostics:
- Open questions / next steps:
