"""C10: capture per-layer post-rotation coordinate distributions.

Runs Qwen2-0.5B in mode `C_prenorm` (KVCE on every layer, C1 off) with
the codec's rotation-capture hook active. For each compress_key /
compress_value call inside KVCE, the post-rotation coordinate vector is
recorded, tagged with its layer index. After N forward passes the
buffer is drained per layer and written to disk as a single .npz with
arrays L{L}_K and L{L}_V (float32, shape [n_vectors, vector_dim]).

These captures are the input to c10_retune_centroids.py.

Why C_prenorm (not mode A): the production distribution L1 sees is
KVCE-corrupted K from L0. Capturing under C_prenorm reflects what each
layer's KVCE actually has to quantize. (One-pass approximation; if
retuning L0 changes the L1 distribution materially, this can be
iterated.)

Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c10_capture_per_layer.py --n-samples 8

Outputs:
    analysis/c10_captures.npz
    analysis/c10_capture_stats.json (per-layer K/V counts and norm stats)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-samples", type=int, default=8,
                    help="Chunks of seq_len tokens to push through. Each "
                         "chunk yields ~seq_len x Hkv vectors per layer.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="C_prenorm",
                    choices=["C_prenorm", "C"])
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis" / "c10_captures.npz"),
    )
    ap.add_argument(
        "--stats-out",
        default=str(REPO_ROOT / "analysis" / "c10_capture_stats.json"),
    )
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] model={args.model}  mode={args.mode}  "
          f"n_samples={args.n_samples}  seq_len={args.seq_len}", flush=True)

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
    n_layers = model.config.num_hidden_layers

    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_samples,
                                  seed=args.seed)
    print(f"[setup] {len(chunks)} chunks prepared", flush=True)

    akvce.set_config(args.mode)
    akvce.set_kvce_layers(None)
    akvce.set_capture_mode(True)
    akvce.reset_stats()

    t0 = time.time()
    nll_total, tok_total = 0.0, 0
    for i, chunk in enumerate(chunks):
        nll, ntok = nll_for_chunk(model, chunk, device)
        if not math.isnan(nll) and ntok > 0:
            nll_total += nll
            tok_total += ntok
        print(f"  chunk {i+1}/{len(chunks)}: nll={nll:.4f} ntok={ntok}",
              flush=True)
    dt = time.time() - t0
    ppl = math.exp(nll_total / max(tok_total, 1)) if tok_total else float("nan")
    print(f"\n[capture] wall={dt:.1f}s  PPL={ppl:.3f}  "
          f"(sanity vs C11 C_prenorm reference)", flush=True)

    # Drain.
    buf = akvce.pop_capture_buffer()
    akvce.set_capture_mode(False)

    if not buf:
        print("[error] capture buffer empty; was KVCE_LAYERS set?",
              file=sys.stderr)
        shutdown_pool()
        return 1

    # Pack into a single .npz with arrays L{idx}_K and L{idx}_V.
    archive = {}
    stats = {"n_layers_captured": len(buf), "per_layer": {}, "model": args.model,
             "mode": args.mode, "n_samples": args.n_samples,
             "seq_len": args.seq_len, "ppl_sanity": ppl,
             "n_tokens_total": tok_total}
    for L in sorted(buf.keys()):
        K = buf[L]["K"]
        V = buf[L]["V"]
        archive[f"L{L}_K"] = K.astype(np.float32)
        archive[f"L{L}_V"] = V.astype(np.float32)
        coords_K = K.reshape(-1)
        coords_V = V.reshape(-1)
        stats["per_layer"][str(L)] = {
            "K_vectors": int(K.shape[0]),
            "V_vectors": int(V.shape[0]),
            "K_coord_mean": float(coords_K.mean()) if coords_K.size else 0.0,
            "K_coord_std":  float(coords_K.std()) if coords_K.size else 0.0,
            "K_coord_max":  float(np.abs(coords_K).max()) if coords_K.size else 0.0,
            "V_coord_mean": float(coords_V.mean()) if coords_V.size else 0.0,
            "V_coord_std":  float(coords_V.std()) if coords_V.size else 0.0,
            "V_coord_max":  float(np.abs(coords_V).max()) if coords_V.size else 0.0,
        }
        print(f"  L{L:>2}: K={K.shape[0]:>6} vecs  V={V.shape[0]:>6} vecs  "
              f"K_std={stats['per_layer'][str(L)]['K_coord_std']:.4f}  "
              f"K_max={stats['per_layer'][str(L)]['K_coord_max']:.3f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **archive)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2))
    print(f"\n[done] wrote {args.out}\n[done] wrote {args.stats_out}",
          flush=True)
    shutdown_pool()
    return 0


if __name__ == "__main__":
    sys.exit(main())
