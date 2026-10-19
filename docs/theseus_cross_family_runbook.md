# Theseus cross-family GSM8K transfer: server runbook

Scope: **Theseus method only** (`rebase/methods/theseus.py`, `theseus_gqa.py`), run through the
**LLM entry point** (`merge_and_rebase.eval.llm_rebase`) — not the paper's vision entry point
(`eval.vision_rebase`, `configs/vision8_theseus_all.json`), and not the other methods this repo
hosts (GradFix, TAK, DELTA, Ariadne, BiCo). Not using `vision_rebase` or the other three papers.

Target model: **base Llama-3.2-1B + your own UltraChat SFT**, not `Llama-3.2-1B-Instruct` — chosen
specifically because Instruct already has non-trivial GSM8K ability baked in, which would
confound "did Theseus transport GSM8K ability" with "was it already there." Source model: open —
pick any same-shape (16 layers / hidden 2048 / intermediate 8192) decoder-family checkpoint with a
real, verified RLVR/GSM8K-training gain over its own base. Only GSM8K is in scope for now (no
MMLU/AlpacaEval/SST yet).

## 0. What changed in this repo, and why

| Change                                            | File(s)                                                                             | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layer_filter` method param (selective transport) | `rebase/methods/_shared.py`, `rebase/methods/theseus.py`                            | Experiment 3 needs "transport only these layers, leave the rest of the target's weights untouched." Nothing in the repo did this before. Implemented as a post-filter on the computed Procrustes transforms: a layer dropped from `layer_filter` becomes a "missing transform," which the existing (already-tested) non-strict apply path already treats as zero-contribution passthrough — so excluded layers keep the target's original weights, which is exactly selective transport.                                                                                                                                                                                          |
| Alignment-diagnostic math library                 | `eval/diagnostics/theseus_alignment.py`                                             | Experiment 1 needs `recon_error`, `rank_90`, `fidelity`-style, `norm_ratio` numbers per layer. The repo's own diagnostics (`_PrecomputeDiagnostics`/`_ApplyDiagnostics`) only count transported/skipped/wrong-shape keys — they don't measure alignment _quality_. This is new, pure math (no model/GPU dependency), unit-tested against synthetic ground truth. See the module docstring for exactly which formulas are verbatim from the paper (arXiv 2602.12952, Eq. 3 / Eq. 11) versus this project's own operational proxy (`recon_error`, `rank_at_variance`, the scalar `fidelity_score` reduction) — don't attribute the proxy metrics to the paper when writing this up. |
| Unit tests                                        | `tests/test_theseus_layer_filter.py`, `tests/test_theseus_alignment_diagnostics.py` | Both ran clean on CPU with synthetic tensors (`uv run --extra dev pytest tests/test_theseus_layer_filter.py tests/test_theseus_alignment_diagnostics.py tests/test_theseus_rebase.py` → 38 passed). No GPU or real checkpoint needed to trust this part; the full suite's pre-existing `test_reference_configs.py` failures are a Windows-locale (`gbk` codec) issue reading the YAML files, unrelated to these changes — confirmed by reproducing the same failures on a clean `git stash`. They won't occur on the Ubuntu server with a UTF-8 locale.                                                                                                                           |

What's **not** changed and doesn't need to be: `procrustes_maps()`/SVD, the `τ_B = T_out τ_A T_in^T`
transport rule, the alpha-sweep loop, `depth_alignment` (BRACE vs. `index_match`), and GSM8K/MMLU
evaluation (both are plain lm-eval-harness task names, already passthrough via `harness_tasks`).

What's still **missing** and needs a small, contained addition once you've picked the source model:
a family adapter for it, if it isn't already one of `llama` / `qwen2` / `qwen3` registered in
`rebase/model_families/registry.py`. See step 2 below — it's normally ~5 lines if the checkpoint
uses standard HF decoder naming (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`).

## 1. Server setup

```bash
git clone <this repo> && cd merge-and-rebase
uv sync --extra dev
# sanity: confirm the new code and tests are intact before touching any real checkpoint
uv run pytest tests/test_theseus_layer_filter.py tests/test_theseus_alignment_diagnostics.py tests/test_theseus_rebase.py -q
```

## 2. Lock in the two checkpoints

**Target** (fixed): point `target_model_name_or_path` at the stock `meta-llama/Llama-3.2-1B`
config/tokenizer, and override the actual weights with your UltraChat SFT checkpoint via
`target_base_ckpt` (a config key the pipeline already supports — "override the target base
weights"). Do **not** reuse the old `36.2%` GSM8K baseline number: that was measured on
`Llama-3.2-1B-Instruct`, not your SFT'd checkpoint. Re-run the baseline eval first:

```bash
uv run python -m merge_and_rebase.eval.llm_rebase \
  --config configs/examples/reference/llm/theseus_gqa.yaml \
  --override method.name=identity \
  --override target_base_ckpt=/path/to/your/llama3.2-1b-ultrachat-sft \
  --override harness_tasks='["gsm8k"]'
```

(`method: identity` just scores the target as-is — no transport — to get your real floor number.)

**Source** (your pick): verify all three checkpoints and the source module layout before writing any config. The
audit instantiates the source architecture with empty/meta weights, so printing its module tree does not load a 1B
checkpoint into RAM or VRAM:

```bash
mkdir -p artifacts/theseus
uv run theseus-model-audit \
  --source-base <source-base-checkpoint> \
  --source-tuned <source-rlvr-checkpoint> \
  --target meta-llama/Llama-3.2-1B \
  --json-output artifacts/theseus/model_audit.json \
| tee artifacts/theseus/model_audit.log
```

The command exits nonzero when a checkpoint is not 16 layers / hidden 2048 / intermediate 8192, when the source
base and tuned configs differ architecturally, or when it cannot identify the source decoder-layer container.
Use `--trust-remote-code` only when the source checkpoint actually requires it.

If `cfg.model_type` isn't already `llama`, `qwen2`, or `qwen3` (check
`src/merge_and_rebase/rebase/model_families/registry.py`), add a family adapter — this is the
only piece that depends on which source you choose, and it's small as long as the checkpoint uses
standard HF decoder module naming:

```python
# src/merge_and_rebase/rebase/model_families/hf_decoder.py
class YourFamilyDecoderAdapter(HfDecoderAdapter):
    name: str = "your_family"
    _MODEL_TYPES = frozenset({"your_model_type"})  # cfg.model_type from above

# src/merge_and_rebase/rebase/model_families/registry.py
from .hf_decoder import YourFamilyDecoderAdapter
register(YourFamilyDecoderAdapter())
```

Then run `python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('<source-checkpoint>'); print(m)"`
once and eyeball that the printed module tree really is `model.layers[i].self_attn.{q,k,v,o}_proj`
/ `model.layers[i].mlp.{gate,up,down}_proj` — if the names differ even slightly (fused qkv, a
differently-named norm, an extra QK-norm module like Qwen3's), that's the #1 real failure mode
("hook fires on wrong module") and must be caught here, not after a multi-hour run.

## 3. Experiment 0 — sanity checks (no transport yet)

Extract and validate the source task vector from the two complete dense Hugging Face checkpoints:

```bash
mkdir -p artifacts/theseus
uv run theseus-extract-task-vector \
  --base <source-base-checkpoint> \
  --tuned <source-rlvr-checkpoint> \
  --output artifacts/theseus/source_rlvr_task_vector.safetensors \
  --output-dtype fp32 \
| tee artifacts/theseus/task_vector_extract.log
```

The command enforces identical floating-parameter keys and shapes, computes subtraction in fp32, rejects NaN/Inf
and an effectively-zero result, and writes four artifacts:

- `source_rlvr_task_vector.safetensors`: tensors keyed exactly like the source model, with `tau = tuned - base`;
- `source_rlvr_task_vector.stats.csv`: per-parameter shape, Frobenius norm, relative norm and near-zero flag;
- `source_rlvr_task_vector.layers.csv`: aggregate Frobenius norm by decoder layer;
- `source_rlvr_task_vector.summary.json`: checkpoint provenance and global counts/norm.

Use `--output-dtype bf16` only to reduce artifact size after the fp32 extraction has been validated; retain fp32 for
the diagnostic run and paper numbers. This utility expects a full dense tuned checkpoint, not a LoRA-only adapter.

Watch for: every transportable layer should show nonzero norm; a near-zero layer was frozen by
RLVR and should be dropped from consideration for Experiment 3's layer-selection variants. NaN/Inf
anywhere here means something is already numerically broken — stop before step 4.

### OLMo2 -> Llama attention-only smoke run

For the 16-layer `OLMo-2-0425-1B` / `Llama-3.2-1B` pair, run the checked-in smoke config before any full sweep:

```bash
mkdir -p artifacts/theseus/logs
set -o pipefail
uv run --extra dev pytest \
  tests/test_llm_family_adapter.py \
  tests/test_family_adapter_layout_api.py \
  tests/test_theseus_layer_filter.py \
  tests/test_theseus_cross_tokenizer_alignment.py \
  tests/test_theseus_rebase.py -q

# Plumbing control: alpha=0 must leave the target unchanged.
uv run python -m merge_and_rebase.eval.llm_rebase \
  --config configs/examples/llm/theseus_olmo2_llama32_attention_smoke.json \
  --alpha 0 \
  --save-merged artifacts/theseus/smoke_attention_alpha_0.pt \
| tee artifacts/theseus/logs/attention_smoke_alpha_0.log

# First non-zero transport.
uv run python -m merge_and_rebase.eval.llm_rebase \
  --config configs/examples/llm/theseus_olmo2_llama32_attention_smoke.json \
| tee artifacts/theseus/logs/attention_smoke.log
```

The `parameter_filter` is applied before Procrustes SVDs, so this run fits exactly the 16 x 4 attention projection
weights and does not allocate an 8192 x 8192 MLP SVD. The required log gates are:

- `parameter_filter kept 64/... parameters`;
- `registering hooks on 64 selected modules` (no MLP hooks in this round);
- precompute diagnostics show `transport slots=64 usable=64`, with zero incomplete/unsupported/skipped counts;
- `head-aware attention transforms = 64 weights`;
- apply diagnostics report `active=64`, with zero wrong-shape and transport-failure counts;
- no NaN/Inf and coherent sample generations.

`out_of_scope_zero` is expected to be non-zero: it counts the deliberately excluded MLP/norm task-vector keys,
not a transport failure.

Do not start the full alpha sweep unless all gates pass. The full-calibration template is
`configs/examples/llm/theseus_olmo2_llama32_attention_full.json`; copy it to a per-alpha config and change both
`alpha.value` and `save.merged` together so results cannot overwrite one another.

For 0d (norm preservation), test a **square dense-Procrustes** map (for example plain THESEUS on
one q-projection) with `theseus_alignment.norm_ratio(tau_a, t_in, t_out)`; that ratio should be
~1.0. Do not impose this invariant on OLMo2 -> Llama k/v maps or on duplicated/pooled
`theseus_gqa` head blocks: those transforms are rectangular or not globally orthogonal, so their
ratio is a measurement to report, not an expected identity.

## 4. Experiment 1 — diagnostics (no merging)

Reuse `collect_activations` (already in `rebase/methods/theseus.py`) with `store_raw=True`,
`store_a_gram=True`, `store_b_gram=True`, and pass `source_model_ft=<source RLVR model>` so the
fine-tuned source's activations are captured in the same pass (stored under the `ft::` key
prefix — see that function's docstring). Then, per layer/module key, feed the accumulated
Gram/cross-covariance tensors into `eval/diagnostics/theseus_alignment.py`:

```python
from merge_and_rebase.rebase.methods.theseus import collect_activations
from merge_and_rebase.eval.diagnostics.theseus_alignment import (
    procrustes_map, recon_error, rank_at_variance, norm_ratio, functional_gram, fidelity_score,
)
import csv

registry = collect_activations(
    source_model=source_base, target_model=target_base_model,
    source_dataloader=calib_loader_src, target_dataloader=calib_loader_tgt,
    device="cuda", seq_align="interpolate", n_batches=100, seed=0,
    store_raw=True, store_a_gram=True, store_b_gram=True,
    source_model_ft=source_rlvr_model, family_adapter=family_adapter,
)

rows = []
for module_name in <per-layer module names from family_adapter.block_components(...)>:
    for side in ("in", "out"):
        store = registry.get(f"{module_name}.{side}")
        if store is None or store.at_a is None:
            continue
        t = procrustes_map(store.at_b)
        err = recon_error(store.at_a, store.bt_b, store.at_b, t)
        rank90 = rank_at_variance(store.at_b, threshold=0.9)
        rows.append({"layer": module_name, "side": side, "recon_error": err, "rank_90": rank90})
# fidelity_score needs the "out" side's ft vs base raw activations (functional_gram), plus the
# transported tau_B at this layer -- wire this from theseus's own tau_A/t_in/t_out once the
# Experiment 2 config below has run once with save.merged set, or compute it standalone the same
# way theseus.apply() does ($T_out^T \tau_A T_in$).

with open("exp1_diagnostics.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=rows[0].keys()).writerows(rows)
```

**Verify the plumbing before trusting the CSV**: print `store.at_b.shape` for one known layer and
confirm it matches `(source_hidden_or_head_dim, target_hidden_or_head_dim)` as expected from the
Experiment 0a shape audit — a silent wrong-module hook looks like a normal-shaped but meaningless
tensor, it won't crash.

**Numbers to read off this CSV:**

- `recon_error` per layer: expect U-shaped across depth (better early/late, worse mid-stack). Flat
  and `> 0.7` everywhere → stop, no shared cross-family subspace, that's your finding.
- `rank_90` per layer: `< 500/2048` is usable structure; consistently `> 1500` everywhere is the
  stop condition.
- `fidelity_score`: distinct from `recon_error` — this is whether the _task vector's effect_
  survives, not whether raw activations align. This is the number that should predict Experiment
  3's selective-transport GSM8K gain.

## 5. Experiment 2 — full uniform transport, alpha sweep

```yaml
# configs/theseus_gqa_cross_family_full.yaml
method:
  name: theseus_gqa
  params:
    num_batches: 100
    seed: 0
    seq_align: interpolate
    device_transform: gpu
depth_alignment:
  rule: index_match # 16 == 16 layers: clean 1:1, no BRACE depth-bridging needed/wanted here
alpha:
  search: false
  value: 1.0 # sweep: 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 (separate runs, or alpha.search: true)
source_model_name_or_path: <source-base-checkpoint>
target_model_name_or_path: meta-llama/Llama-3.2-1B
target_base_ckpt: /path/to/your/llama3.2-1b-ultrachat-sft
tuned_bodies:
  gsm8k: <source-rlvr-checkpoint>
model_kind: causal_lm
device: cuda
dtype: bf16
calibration_batch_size: 4
calibration_max_length: 192
calibration_dataset:
  path: gsm8k
  name: main
  split: train
  text_columns: [question, answer]
  text_template: "{question}\n{answer}"
harness_tasks: [gsm8k]
harness_num_fewshot: 0
harness_batch_size: 4
```

```bash
for a in 0.05 0.1 0.2 0.3 0.5 0.7 1.0; do
  uv run python -m merge_and_rebase.eval.llm_rebase \
    --config configs/theseus_gqa_cross_family_full.yaml \
    --override alpha.value=$a \
    --override save.merged=runs/full_alpha_$a.pt \
  | tee logs/full_alpha_$a.log
done
```

**Watch per alpha**: GSM8K vs. your freshly-measured target floor (not 36.2%); wall-clock (should
be minutes); do a few manual generations at each alpha for coherence. If GSM8K is below the floor
at every alpha, stop raising alpha — go to Experiment 3 instead of chasing a higher value.

## 6. Experiment 3 — selective transport using `layer_filter`

Pick layer indices from the Experiment 1 CSV (e.g. `fidelity_score > 0.5`, or
`recon_error < 0.4`, or top-K by `fidelity_score`), then:

```yaml
method:
  name: theseus_gqa
  params:
    num_batches: 100
    seed: 0
    seq_align: interpolate
    device_transform: gpu
    layer_filter:
      include: [2, 3, 9, 11] # from the Exp 1 CSV; "exclude: [...]" is the other direction
```

Everything else is identical to the Experiment 2 config. Re-run the same alpha sweep.

**Watch**: does selective beat uniform at matched alpha? That correlation — high-`fidelity_score`
layers kept → better GSM8K than uniform — is the actual finding regardless of the absolute GSM8K
number you land on.

## 7. Final comparison

| Method                                                         | GSM8K                  | Notes                                      |
| -------------------------------------------------------------- | ---------------------- | ------------------------------------------ |
| Target floor (your UltraChat SFT'd Llama-3.2-1B, no transport) | _(re-measure, step 2)_ | replaces the old 36.2% Instruct number     |
| OPD 1000 steps                                                 | 26.5%                  | existing baseline                          |
| HeteroFusion                                                   | 7.2%                   | existing baseline, catastrophic forgetting |
| Theseus full (best alpha)                                      | ?                      | Experiment 2                               |
| Theseus selective (best config)                                | ?                      | Experiment 3                               |

Success bands (GSM8K only, per your current scope): **strong** ≥ floor + 4pp; **moderate** small
gain, no coherence loss; **weak** ≈ floor, stable; **failure** well below floor — document the
Experiment 1 diagnostics either way, since the layer-wise alignment story is a finding independent
of where the GSM8K number lands.

#### my short cut cmds

##### export a saved merged state as a Hugging Face checkpoint

`--save-merged` writes a complete target-model state dict, not a delta. Export it with strict key checking and copy
the target tokenizer/config into a standard Hugging Face directory:

```bash
uv run theseus-export-hf \
  --base /data/shared_ckpt/Llama-3.2-1B \
  --state artifacts/theseus/smoke_attention_alpha_0.pt \
  --output artifacts/theseus/smoke_attention_alpha_0_hf \
  --output-dtype fp32 \
  --max-shard-size 2GB
```

The output path must not already exist, preventing stale model shards from an older export. Use `bf16` instead of
`fp32` only when the smaller evaluation artifact is intentional; retain the original fp32 `.pt` for provenance.

##### exteract tuned weights

uv run theseus-extract-task-vector \
 --base /data/shared_ckpt/OLMo-2-0425-1B \
 --tuned /data/shared_ckpt/opd_teacher \
 --output artifacts/theseus/source_rlvr_task_vector.safetensors \
 --output-dtype fp32 \
 | tee artifacts/theseus/task_vector_extract.log

##### checking shapes of two

uv run theseus-model-audit \
 --source-base /data/shared_ckpt/OLMo-2-0425-1B \
 --source-tuned /data/shared_ckpt/opd_teacher \
 --target /data/shared_ckpt/Llama-3.2-1B \
 --json-output artifacts/theseus/model_audit.json \
 | tee artifacts/theseus/model_audit.log
