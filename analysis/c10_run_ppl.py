"""C10: end-to-end PPL with per-layer retuned centroids.

Runs C_prenorm twice on the same WikiText sample, once with the chip's
default Lloyd-Max-for-Gaussian centroids (the C11 baseline) and once
with per-layer Lloyd-Max-retuned centroids from
c10_retune_centroids.py. Reports the recovery on the +5.64-bits/tok
C12 noise floor measured in C11.

Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c10_run_ppl.py --n-samples 16

Outputs:
    analysis/c10_ppl_runs.jsonl     (append-only per-config rows)
    analysis/c10_ppl_summary.json   (latest baseline + retuned + delta)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import acu_kvce_attention as akvce
from c11_wikitext_ppl import load_wikitext_chunks, nll_for_chunk
from kvce_pool import shutdown_pool


DEFAULT_MODEL = "Qwen/Qwen2-0.5B"


def run_one(model, chunks, device, mode: str, label: str,
            centroid_tables_path: str | None) -> dict:
    akvce.set_config(mode)
    akvce.set_kvce_layers(None)
    akvce.set_capture_mode(False)
    akvce.set_centroid_tables(centroid_tables_path)
    akvce.reset_stats()
    t0 = time.time()
    nll_total, tok_total = 0.0, 0
    per_chunk_ppl = []
    for chunk in chunks:
        nll, ntok = nll_for_chunk(model, chunk, device)
        if not math.isnan(nll) and ntok > 0:
            nll_total += nll
            tok_total += ntok
            per_chunk_ppl.append(math.exp(nll / ntok))
    dt = time.time() - t0
    ppl = math.exp(nll_total / max(tok_total, 1)) if tok_total else float("nan")
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": label,
        "mode": mode,
        "centroid_tables_path": centroid_tables_path,
        "ppl_pooled": ppl,
        "ppl_per_chunk": per_chunk_ppl,
        "tokens_total": tok_total,
        "wall_s": dt,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", default="C_prenorm",
                    choices=["C_prenorm", "C", "E_prenorm", "E"])
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tables",
        default=str(REPO_ROOT / "analysis" / "c10_centroid_tables.json"),
        help="Per-layer centroid JSON.",
    )
    ap.add_argument(
        "--runs-out",
        default=str(REPO_ROOT / "analysis" / "c10_ppl_runs.jsonl"),
    )
    ap.add_argument(
        "--summary-out",
        default=str(REPO_ROOT / "analysis" / "c10_ppl_summary.json"),
    )
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Only run the retuned config (reuse a prior "
                         "baseline_C_prenorm row from the JSONL).")
    args = ap.parse_args(argv)

    if not Path(args.tables).exists():
        print(f"[error] centroid table not found: {args.tables}",
              file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] model={args.model}  mode={args.mode}  "
          f"n_samples={args.n_samples}  tables={args.tables}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    akvce.register("acu_kvce")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="acu_kvce",
    )
    model.eval()
    device = next(model.parameters()).device

    # Also measure mode A so we always have a baseline_A reference even
    # if the JSONL is fresh.
    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_samples,
                                  seed=args.seed)
    print(f"[setup] {len(chunks)} chunks prepared", flush=True)

    runs_path = Path(args.runs_out)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    def log_row(rec: dict) -> None:
        with runs_path.open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if k != "ppl_per_chunk"}) + "\n")

    print("\n[1/3] baseline A (no KVCE)", flush=True)
    rec_A = run_one(model, chunks, device, "A", "baseline_A", None)
    print(f"  PPL = {rec_A['ppl_pooled']:.3f}  wall={rec_A['wall_s']:.1f}s",
          flush=True)
    log_row(rec_A)

    if not args.skip_baseline:
        print(f"\n[2/3] {args.mode} with DEFAULT centroids", flush=True)
        rec_def = run_one(model, chunks, device, args.mode,
                          f"{args.mode}_default", None)
        print(f"  PPL = {rec_def['ppl_pooled']:.3f}  "
              f"wall={rec_def['wall_s']:.1f}s", flush=True)
        log_row(rec_def)
    else:
        print(f"\n[2/3] {args.mode} baseline skipped (use prior JSONL row)",
              flush=True)
        rec_def = None

    print(f"\n[3/3] {args.mode} with RETUNED centroids", flush=True)
    rec_ret = run_one(model, chunks, device, args.mode,
                      f"{args.mode}_retuned", args.tables)
    print(f"  PPL = {rec_ret['ppl_pooled']:.3f}  "
          f"wall={rec_ret['wall_s']:.1f}s", flush=True)
    log_row(rec_ret)

    # Summary.
    A = rec_A["ppl_pooled"]
    R = rec_ret["ppl_pooled"]
    if rec_def is None:
        # Pull the most recent default row from the JSONL.
        try:
            with runs_path.open() as f:
                rows = [json.loads(l) for l in f if l.strip()]
            default_rows = [r for r in rows
                            if r["label"] == f"{args.mode}_default"]
            if default_rows:
                D = default_rows[-1]["ppl_pooled"]
            else:
                D = float("nan")
        except Exception:
            D = float("nan")
    else:
        D = rec_def["ppl_pooled"]

    log_A = math.log(A) if A > 0 else float("nan")
    log_D = math.log(D) if D > 0 else float("nan")
    log_R = math.log(R) if R > 0 else float("nan")
    gap_default = log_D - log_A
    gap_retuned = log_R - log_A
    recovered = gap_default - gap_retuned

    summary = {
        "model": args.model,
        "mode": args.mode,
        "n_samples": args.n_samples,
        "seq_len": args.seq_len,
        "tables": args.tables,
        "ppl_baseline_A": A,
        "ppl_default_centroids": D,
        "ppl_retuned_centroids": R,
        "log_gap_default_nats": gap_default,
        "log_gap_retuned_nats": gap_retuned,
        "log_gap_recovered_nats": recovered,
        "fraction_recovered": (recovered / gap_default) if gap_default > 0 else float("nan"),
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"baseline A           : PPL = {A:8.3f}")
    print(f"{args.mode} default  : PPL = {D:8.3f}  gap = +{gap_default:.3f} nats")
    print(f"{args.mode} retuned  : PPL = {R:8.3f}  gap = +{gap_retuned:.3f} nats")
    print(f"recovered            : {recovered:+.3f} nats "
          f"({100 * summary['fraction_recovered']:+.1f}% of default gap)")
    print(f"[done] wrote {args.summary_out}")
    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
