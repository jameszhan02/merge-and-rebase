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

| Change | File(s) | Why |
|---|---|---|
| `layer_filter` method param (selective transport) | `rebase/methods/_shared.py`, `rebase/methods/theseus.py` | Experiment 3 needs "transport only these layers, leave the rest of the target's weights untouched." Nothing in the repo did this before. Implemented as a post-filter on the computed Procrustes transforms: a layer dropped from `layer_filter` becomes a "missing transform," which the existing (already-tested) non-strict apply path already treats as zero-contribution passthrough — so excluded layers keep the target's original weights, which is exactly selective transport. |
| Alignment-diagnostic math library | `eval/diagnostics/theseus_alignment.py` | Experiment 1 needs `recon_error`, `rank_90`, `fidelity`-style, `norm_ratio` numbers per layer. The repo's own diagnostics (`_PrecomputeDiagnostics`/`_ApplyDiagnostics`) only count transported/skipped/wrong-shape keys — they don't measure alignment *quality*. This is new, pure math (no model/GPU dependency), unit-tested against synthetic ground truth. See the module docstring for exactly which formulas are verbatim from the paper (arXiv 2602.12952, Eq. 3 / Eq. 11) versus this project's own operational proxy (`recon_error`, `rank_at_variance`, the scalar `fidelity_score` reduction) — don't attribute the proxy metrics to the paper when writing this up. |
| Unit tests | `tests/test_theseus_layer_filter.py`, `tests/test_theseus_alignment_diagnostics.py` | Both ran clean on CPU with synthetic tensors (`uv run --extra dev pytest tests/test_theseus_layer_filter.py tests/test_theseus_alignment_diagnostics.py tests/test_theseus_rebase.py` → 38 passed). No GPU or real checkpoint needed to trust this part; the full suite's pre-existing `test_reference_configs.py` failures are a Windows-locale (`gbk` codec) issue reading the YAML files, unrelated to these changes — confirmed by reproducing the same failures on a clean `git stash`. They won't occur on the Ubuntu server with a UTF-8 locale. |

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

**Source** (your pick): verify its exact shape before writing any config —

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("<source-checkpoint>")
print(cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size)
# must be: 16, 2048, 8192 to use depth_alignment.rule=index_match cleanly (see the index_match note below)
```

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

```python
# scripts/exp0_sanity.py (ad hoc, not part of the package)
import torch
from transformers import AutoModelForCausalLM

source = AutoModelForCausalLM.from_pretrained("<source-checkpoint>")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# 0a/0b: per-layer task-vector norms (needs the source's own base, pre-RLVR, checkpoint too)
source_base = AutoModelForCausalLM.from_pretrained("<source-base-checkpoint>")
for name, p in source.state_dict().items():
    if name in source_base.state_dict():
        delta_norm = (p - source_base.state_dict()[name]).float().norm().item()
        if delta_norm > 1e-6:
            print(f"{name}: ||tau_A|| = {delta_norm:.4f}")
```

Watch for: every transportable layer should show nonzero norm; a near-zero layer was frozen by
RLVR and should be dropped from consideration for Experiment 3's layer-selection variants. NaN/Inf
anywhere here means something is already numerically broken — stop before step 4.

For 0d (norm preservation), use the new library directly once you have one real `t_in`/`t_out`
pair and task-vector slice (wire-up shown in step 4): `theseus_alignment.norm_ratio(tau_a, t_in,
t_out)` must print ~1.0. This is an exact identity (paper Eq. 11) — any deviation beyond float
error means the SVD/transform code path itself is broken, not a cross-family alignment finding.

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
- `fidelity_score`: distinct from `recon_error` — this is whether the *task vector's effect*
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
  rule: index_match   # 16 == 16 layers: clean 1:1, no BRACE depth-bridging needed/wanted here
alpha:
  search: false
  value: 1.0           # sweep: 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 (separate runs, or alpha.search: true)
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
      include: [2, 3, 9, 11]   # from the Exp 1 CSV; "exclude: [...]" is the other direction
```
Everything else is identical to the Experiment 2 config. Re-run the same alpha sweep.

**Watch**: does selective beat uniform at matched alpha? That correlation — high-`fidelity_score`
layers kept → better GSM8K than uniform — is the actual finding regardless of the absolute GSM8K
number you land on.

## 7. Final comparison

| Method | GSM8K | Notes |
|---|---|---|
| Target floor (your UltraChat SFT'd Llama-3.2-1B, no transport) | *(re-measure, step 2)* | replaces the old 36.2% Instruct number |
| OPD 1000 steps | 26.5% | existing baseline |
| HeteroFusion | 7.2% | existing baseline, catastrophic forgetting |
| Theseus full (best alpha) | ? | Experiment 2 |
| Theseus selective (best config) | ? | Experiment 3 |

Success bands (GSM8K only, per your current scope): **strong** ≥ floor + 4pp; **moderate** small
gain, no coherence loss; **weak** ≈ floor, stable; **failure** well below floor — document the
Experiment 1 diagnostics either way, since the layer-wise alignment story is a finding independent
of where the GSM8K number lands.
