# Changes to APA driven by the C11 end-to-end sweep

This document lists the concrete changes for `adaptive-precision-attention`
that follow from the end-to-end accuracy measurement
(`docs/findings/c11_end_to_end_accuracy.md`). Each change cites its
evidence; nothing here is speculative.

Status legend: **shipped** (in this repo, on a branch); **proposed**
(spec'd but not implemented); **escalated** (needs a design decision).

---

## 0. PC routing is end-to-end safe -- NO CHANGES to RTL  **[vindicated]**

**Evidence:**
- `analysis/c11_wikitext_ppl_runs.jsonl`: config B (PC routing only,
  true K/V) PPL = 17.58 vs baseline A PPL = 17.56. Ratio = 1.001x
  = +0.001 bits/tok. Well inside chunk-to-chunk noise.
- `analysis/c11_hellaswag_runs.jsonl`: configs A and B give identical
  predictions on all 250 items (acc = 0.404, acc_norm = 0.420
  bit-for-bit). PC routing flips zero rankings.
- Integration audit (`docs/findings/kvce-acu-integration-audit.md`):
  per-tile rMSE 0.0002 -- the per-tile signal that predicted this.

**Implication:** `rtl/precision_controller.sv` is correct as
designed. No RTL changes. No reference-model changes. Document the
empirical safety in the next datasheet revision.

---

## 1. Define and own the Q4.12 <-> INT8/FP16 bridge (the C3 sub-block)  **[escalated]**

**Open conflict:** C3 (int16 Q4.12 <-> int8/fp16 bridge unspecified).

**Why now:** C11 made the *float-to-Q4.12* half of the bridge
empirically necessary (closing C1 needs per-vector pre-norm at the
boundary -- see KVCE side: `kv-cache-engine/CHANGES_FROM_C11.md`
item 1). The *Q4.12-to-INT8/FP16* half is the chip-side mirror of
the same bridge. Both halves should be spec'd together.

**Recommended placement:** ACU input/output stage. The bridge has
to know:
- the per-vector scale (for the prenorm side), and
- the precision controller's INT8/FP16 decision (for the routing side).

Both naturally live on the ACU side. KVCE stays a pure Q4.12 codec;
its bit-exact contract is preserved.

**What this repo needs:**
- A new sub-block in `rtl/` (proposed: `acu_kv_bridge.sv`) with:
  - Per-vector scale extract on the K, V outbound path.
  - Per-vector scale apply + INT8/FP16 cast on the K_hat, V_hat
    inbound path.
- Reference model in `sw/reference_model/acu_kv_bridge_ref.py`.
- ISA / port spec in `docs/`.

**Estimate:** 1-2 days of RTL + ref work, plus signoff. Out of scope
for this branch. Filing as the C3 resolution path.

---

## 2. Integration testvector generator: real-LLM mode  **[proposed]**

**Why:** the C11 sweep showed PC routing is safe on real Qwen2-0.5B
activations. The existing `analysis/gen_rtl_testvectors.py` synthesises
testvectors from `N(0, sigma)` distributions -- realistic for
mid-network layers but does not stress the early-layer / outlier
regime that drove the C1 finding.

**Proposed change:**
- Add a `--from-llm-capture` flag to `analysis/gen_rtl_testvectors.py`
  that consumes the same activation captures the C11 harness uses
  (Qwen2-0.5B post-`k_proj` / `q_proj` / `v_proj` per layer).
- Emit per-layer testvector sets so RTL signoff can show stable PC
  behaviour across the realistic activation distribution, not just
  the synthetic one.

**Estimate:** ~2 hours. Useful for next signoff cycle.

---

## 3. Documentation: PC FP16 utilisation by KVCE mode  **[proposed]**

**Why:** the C11 sweep shows PC FP16% = 0.14% under `turbo4` (E config,
~33,800 tiles). This was already predicted by the per-tile audit
(0.18% over 2,744 tiles). It is informational, not a defect.

**Proposed change:**
- Append a "PC FP16 utilisation under KVCE modes" subsection to
  `docs/findings/kvce-acu-integration-audit.md` (one paragraph) and
  to `paper/` proceedings.
- Add a roadmap entry: "re-measure PC FP16 utilisation under turbo8 /
  turbo16 once KVCE adds those modes." This is the empirical
  re-validation that determines whether the FP16 datapath earns its
  area in higher-precision regimes.

---

## 4. Move integration test from Python ref to the in-repo bridge  **[proposed]**

**Why:** `analysis/integration_test_kv_pc.py` and
`analysis/c11_wikitext_ppl.py` currently use
`analysis/kvce_pool.py`'s own prenorm implementation. The KVCE repo
has now added a canonical bridge at
`kv-cache-engine/sw/reference_model/q412_bridge.py`. The two should
converge so there's a single source of truth.

**Proposed change:**
- Replace `kvce_pool.py`'s `mode in ("naive", "prenorm")` with calls
  through `Q412Bridge`. Keep the multiprocessing parallelism; just
  delegate the per-vector conversion to the bridge.
- Re-run the n=64 PPL + n=250 HellaSwag sweep after the convergence
  to confirm the numbers don't shift.

**Estimate:** 30 min code + 90 min re-run.

---

## Conflict register status moves

After landing items 1 and 2 (and once KVCE lands its prenorm bridge):

| # | Was | Move to | Evidence |
|--:|---|---|---|
| C1 | open | **partially resolved -- fix path validated; RTL home pending (see C3)** | this repo's C11 sweep + KVCE's `q412_bridge.py` |
| C3 | open | **open (urgent -- empirical evidence motivates ACU-side bridge)** | item 1 above |
| C11 | open | **resolved (measured)** | `docs/findings/c11_end_to_end_accuracy.md` |
| C12 | informational/resolved | **resolved with magnitude (+5.64 bits/tok at turbo4)** | this repo's C11 sweep |

These edits land in
`docs/findings/kvce_acu_architectural_conflicts.tex` after the
HellaSwag sweep completes (so the C1 numbers carry the final n=250 CIs).
