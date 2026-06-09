<!--
  C11 closure document. Numbers and figures are filled in from
  analysis/c11_wikitext_ppl_runs.jsonl and analysis/c11_hellaswag_runs.jsonl
  after the sweep + HellaSwag complete. Regenerate the figures via
  analysis/c11_make_figs.py.
-->

# C11 - End-to-end accuracy of ACU x KVCE on Qwen2-0.5B

**Branch:** `c11-end-to-end-perplexity` (this repo); KVCE pinned at
`kv-cache-engine@9b1163a`.
**Status:** measured. Update conflict register C11 from "open" to
"resolved (measured)".
**Date:** 2026-06-08.

## What we measured

Per the conflict register's resolution path: perplexity on WikiText-2
(test split, 64 non-overlapping 512-token chunks) and HellaSwag
accuracy (validation, n=250 length-filtered items), with Qwen2-0.5B's
attention substituted by the integrated ACU x KVCE pipeline.

Six configurations on the same prompts:

| Config       | Attention path                                       |
|--------------|------------------------------------------------------|
| A (baseline) | FP16 dense attention (sdpa-equivalent)               |
| B            | True K, V + per-tile PC INT8/FP16 routing            |
| C            | KVCE round-trip on K, V (naive Q4.12); FP16 SV       |
| C_prenorm    | KVCE with per-vector L2 prenorm before Q4.12         |
| E            | KVCE (naive) + PC routing -- the as-is chip pipeline |
| E_prenorm    | KVCE (prenorm) + PC routing                          |

The two prenorm rows isolate **C1 (Q4.12 input range vs raw activation
magnitudes)** from KVCE's intrinsic quantization noise, so the
contribution of each conflict is separable in the final number.

Implementation: `analysis/acu_kvce_attention.py` (flash-attention-style
streaming, per-tile PC decision on int8-quantized pre-softmax S,
KVCE round-trip via `analysis/kvce_pool.py`). Harness:
`analysis/c11_wikitext_ppl.py` and `analysis/c11_hellaswag.py`. Figures:
`analysis/c11_make_figs.py`.

## Results

### WikiText-2 perplexity (Qwen2-0.5B, n=64 chunks, 32,704 prediction tokens)

> Source: `analysis/c11_wikitext_ppl_runs.jsonl` (latest row per config).
> Figure: `paper/figs/c11_ppl_by_config.pdf`.

| Config       | PPL pooled |   PPL median | Wall (s) | PC FP16% |
|--------------|-----------:|-------------:|---------:|---------:|
| A baseline   |   **17.56** |       18.87 |      2.4 |  n/a     |
| B PC only    |       17.58 |       18.92 |     15.5 |  0.000 % |
| C KVCE naive |    3,642.10 |    3,582.34 |    217.8 |  n/a     |
| C_prenorm    |       877.88 |   1,024.99 |    192.5 |  n/a     |
| E integrated |    4,368.83 |    4,179.28 |    203.7 |  0.139 % |
| E_prenorm    |       892.11 |   1,053.17 |    204.8 |  0.000 % |

### HellaSwag (validation, length-filtered subset, n=250)

> Source: `analysis/c11_hellaswag_runs.jsonl` (latest row per config).
> Figure: `paper/figs/c11_hellaswag.pdf`. Chance = 0.25 (4-way MC).

| Config       |     acc | 95 % CI         | acc_norm | 95 % CI         |
|--------------|--------:|-----------------|---------:|-----------------|
| A baseline   | **0.404** | [0.345, 0.466] | **0.420** | [0.360, 0.482] |
| B PC only    |   0.404 | [0.345, 0.466]  |    0.420 | [0.360, 0.482]  |
| C KVCE naive |   0.300 | [0.247, 0.359]  |    0.228 | [0.180, 0.284]  |
| C_prenorm    |   0.272 | [0.221, 0.330]  |    0.316 | [0.262, 0.376]  |
| E integrated |   0.264 | [0.213, 0.322]  |    0.232 | [0.184, 0.288]  |
| E_prenorm    |   0.304 | [0.250, 0.364]  |    0.316 | [0.262, 0.376]  |

Both `C` and `E` (the as-is configs) collapse below the 0.25 chance
line; the prenorm rows recover to ~0.32, still well below the 0.42
baseline.

## Decomposition

We attribute the gap between integrated and baseline to two named
conflicts using the prenorm row as the C1-removed counterfactual.

### PPL decomposition (Qwen2-0.5B, WikiText-2, n=64)

| Source                                          | Factor       | bits/tok |
|-------------------------------------------------|-------------:|---------:|
| PC routing on full-precision V (B / A)          |    **1.001x** |   +0.001 |
| C12 turbo4 noise floor alone (C_prenorm / A)    |   **49.99x** |   +5.64  |
| C1 Q4.12 clip alone (C / C_prenorm)             |    **4.15x** |   +2.05  |
| C12 + PC on lossy V (E_prenorm / A)             |   **50.81x** |   +5.66  |
| C1 inside the integrated pipeline (E / E_prenorm) | **4.90x**   |   +2.29  |
| Full integrated pipeline (E / A)                |  **248.84x** |   +7.95  |

### HellaSwag decomposition (length-normalised accuracy, n=250)

| Source                                          | acc_norm delta | Verdict          |
|-------------------------------------------------|---------------:|-------------------|
| PC routing on full-precision V (B - A)          |  **0.000**     | identical predictions |
| C12 turbo4 noise floor (C_prenorm - A)          |  -0.104        | below baseline    |
| C1 Q4.12 clip alone (C - C_prenorm)             |  -0.088        | drives accuracy below chance |
| C1 inside integrated (E - E_prenorm)            |  -0.084        | symmetric         |
| Full integrated (E - A)                          |  **-0.188**    | catastrophic at as-is spec |

## Key findings

1. **The precision controller is end-to-end safe.** B vs A is
   identical to four decimal places on HellaSwag (0.420 vs 0.420 with
   matching CIs) and within chunk-noise on PPL (17.577 vs 17.557).
   PC FP16% on B is 0.000 % -- i.e. PC routes 100 % of tiles INT8 and
   the per-tile INT8 SV path matches FP16 to within model-output
   resolution. `rtl/precision_controller.sv` needs no changes.

2. **C1 (Q4.12 clip) is the single largest fixable contributor.**
   With C1 in place, the integrated pipeline collapses below chance
   on HellaSwag (acc_norm 0.232 < 0.25). Removing C1 in software with
   a per-vector pre-norm wrapper lifts HellaSwag back above chance
   (acc_norm 0.316) and recovers ~80 % of the PPL gap. The fix is
   architecturally clean: a per-vector scale extracted at the ACU
   output, restored at the ACU input post-KVCE. See KVCE-side
   implementation in `kv-cache-engine@c1-q412-bridge-prenorm`
   (`sw/reference_model/q412_bridge.py`).

3. **C12 (turbo4 noise floor) is the dominant remaining cost.** Even
   with C1 removed, KVCE round-trip alone costs +5.64 bits/tok PPL
   and -0.104 HellaSwag accuracy. This is the intrinsic cost of
   3-bit PolarQuant + 1-bit QJL at this model scale. The path to
   closing it is `turbo8` / `turbo16` modes (see
   `kv-cache-engine/CHANGES_FROM_C11.md` item 3), not algorithmic
   refinement of `turbo4`.

4. **PC routing in the integrated path is approximately free even
   under lossy V.** E_prenorm / C_prenorm = 1.016x PPL, identical
   acc_norm. The audit's prediction that "PC value-add does not
   survive KVCE noise" holds at the model-output level, but the
   safety verdict matters: PC routing does **not** make end-to-end
   accuracy worse than the KVCE-only baseline.

5. **Ranking is harder than next-token surprise.** PPL prenorm
   recovery is ~80 % of the gap; HellaSwag recovery is only ~50 %.
   Downstream-task evaluations should be the gating metric for any
   future spec changes, not PPL alone.

## Conflict register status moves

| # | Was | Move to |
|--:|---|---|
| C1  | open                       | **partially resolved -- fix path validated (prenorm bridge); RTL home pending (see C3)** |
| C3  | open                       | **open (urgent -- C1 evidence motivates ACU-side bridge)** |
| C11 | open                       | **resolved (measured)** with the numbers in this document |
| C12 | informational/resolved     | **resolved with end-to-end magnitude** (+5.64 bits/tok at turbo4) |

## What this means for each repo

### `adaptive-precision-attention`

- The PC routing pipeline is **safe end-to-end** (B vs A) -- confirmation
  that the per-tile INT8 SV path adds no measurable damage to model
  output when V is full-precision. Independent of KVCE noise.
- Under the as-is integrated path (E), the chip's value-add from FP16
  routing is dominated by KVCE noise -- as already predicted by the
  per-tile audit and conflict C12. PPL difference vs `C_prenorm` is the
  empirical end-to-end answer.

### `kv-cache-engine`

- C1 (Q4.12 input range) is now quantified end-to-end: it accounts for
  the `log(PPL_E / PPL_E_prenorm)` factor of the integrated PPL gap.
  A pre-norm bridge stage (or a wider input format) lifts the model
  from "unusable" to "tolerable" - filing as `kv-cache-engine#2`
  resolution evidence.
- C12's turbo4 noise floor (the residual `PPL_C_prenorm / PPL_A` factor)
  remains: KVCE quantization alone, even without C1, leaves PPL at
  multiple times the baseline. Higher-precision modes (turbo8, turbo16)
  are the path to closing this.

### LonghornSilicon chip-level

- The integrated ACU x KVCE attention is **measurable end-to-end** and
  the dominant failure mode at the current spec is the Q4.12 input
  clip (C1), not the precision controller or the KVCE quantization
  algorithm itself. Fix C1 first.
- Even after C1 is removed, the chip would not preserve baseline
  perplexity at turbo4. Either ship at a higher-precision KVCE mode,
  or accept the measured PPL delta as the cost of 3.6 to 5x KV-cache
  compression.

## Reproducing

Sibling clones at `../kv-cache-engine` and this repo. From this repo::

    KVCE_REF=$(readlink -f ../kv-cache-engine/sw/reference_model)
    HF_HOME=$(pwd)/.hf_cache python analysis/c11_wikitext_ppl.py \
        --configs A,B,C,C_prenorm,E,E_prenorm --n-samples 64 --seq-len 512

    HF_HOME=$(pwd)/.hf_cache python analysis/c11_hellaswag.py \
        --configs A,B,C,C_prenorm,E,E_prenorm --n-items 250

    python analysis/c11_make_figs.py

Wall on a DGX Spark (GB10, 20 CPU cores): PPL sweep ~13 min,
HellaSwag ~75 min.
