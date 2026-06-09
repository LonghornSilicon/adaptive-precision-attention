"""C10 v2: capture per-layer K/V from a CLEAN (mode A, no KVCE) forward.

Differs from c10_capture_per_layer.py in one important way: that script
ran C_prenorm (KVCE on every layer) with capture on, so the captures
reflected the distribution at each layer AFTER all upstream KVCE noise
had already corrupted the inputs. Fitting Lloyd-Max on those samples
optimises the centroids for the wrong distribution: the one each layer
sees in the production-with-default-centroids regime, not the true
clean distribution.

This script runs mode A (FP16 dense attention, no KVCE anywhere) and
uses a PyTorch forward pre-hook on each Qwen2Attention to grab the K
and V projections before they would have hit KVCE. Those clean K/V get
pushed through KVCE's normalize+rotate (no quantize, no decompress)
serially to extract post-rotation coordinates per layer. Retrain
Lloyd-Max on these.

Run::

    HF_HOME=/home/chaithu/lhs/.hf_cache \\
    KVCE_REF=/home/chaithu/lhs/kv-cache-engine/sw/reference_model \\
    python analysis/c10_capture_clean.py --n-samples 4

Outputs:
    analysis/c10_captures_clean.npz
    analysis/c10_capture_clean_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "analysis"))
KVCE_REF = os.environ.get(
    "KVCE_REF", "/home/chaithu/lhs/kv-cache-engine/sw/reference_model")
sys.path.insert(0, KVCE_REF)

from c11_wikitext_ppl import load_wikitext_chunks  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2-0.5B"
COORD_FRAC = 12


def rotate_only(engine, K_float: np.ndarray) -> np.ndarray:
    """Pass a [N, D] float array through KVCE's normalize + rotate
    (no quantize, no decompress). Returns the post-rotation coords
    in float (coord frame)."""
    out = np.empty_like(K_float)
    inv_scale = 1.0 / (1 << COORD_FRAC)
    for i, v in enumerate(K_float):
        q = (np.round(v * (1 << COORD_FRAC))
               .clip(-32768, 32767)
               .astype(np.int32)
               .tolist())
        norm = engine.compute_norm(q)
        normalized = engine.normalize(q, norm)
        rotated = engine.rotate(normalized)
        out[i] = np.asarray(rotated, dtype=np.float32) * inv_scale
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis" / "c10_captures_clean.npz"),
    )
    ap.add_argument(
        "--stats-out",
        default=str(REPO_ROOT / "analysis"
                    / "c10_capture_clean_stats.json"),
    )
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[setup] model={args.model}  n_samples={args.n_samples}  "
          f"mode=A (clean FP16)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_cache_engine_ref import KVCacheEngine, KVCacheEngineInfo

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    # Find Qwen2 attention modules and hook them.
    captured_K: dict[int, list[np.ndarray]] = {L: [] for L in range(n_layers)}
    captured_V: dict[int, list[np.ndarray]] = {L: [] for L in range(n_layers)}

    def make_hook(L: int):
        # In Qwen2 the attention forward computes q,k,v_proj then reshapes.
        # We grab K,V from the attention impl's args via a pre-hook on the
        # ALL_ATTENTION_FUNCTIONS dispatch is awkward; simplest: hook the
        # k_proj and v_proj outputs of the attention module.
        def hook(_module, _inputs, output):
            # output shape: [B, N, Hkv * D]; reshape to [Hkv, N, D]
            o = output.detach().float().cpu().numpy()
            # The hook is registered separately on k_proj and v_proj;
            # output here is whatever that linear produced.
            return  # placeholder, we'll fill via separate K vs V hooks
        return hook

    def make_kv_hook(L: int, kind: str):
        bucket = captured_K if kind == "K" else captured_V
        def hook(_module, _inputs, output):
            # output: [B, N, Hkv * D] (k_proj/v_proj output)
            o = output.detach().float().cpu().numpy()
            B, N, full = o.shape
            # Reshape into [B*N*Hkv, D]. We need D = head_dim.
            head_dim = model.config.head_dim if hasattr(model.config, "head_dim") \
                else (model.config.hidden_size // model.config.num_attention_heads)
            hkv = full // head_dim
            o = o.reshape(B, N, hkv, head_dim).reshape(-1, head_dim)
            bucket[L].append(o.astype(np.float32))
        return hook

    handles = []
    for L, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        handles.append(attn.k_proj.register_forward_hook(make_kv_hook(L, "K")))
        handles.append(attn.v_proj.register_forward_hook(make_kv_hook(L, "V")))
    print(f"[setup] hooks on {len(handles)} k_proj+v_proj modules", flush=True)

    chunks = load_wikitext_chunks(tokenizer, args.seq_len, args.n_samples,
                                  seed=args.seed)
    print(f"[setup] {len(chunks)} chunks prepared", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            input_ids = torch.as_tensor(chunk, dtype=torch.long,
                                        device=device).unsqueeze(0)
            _ = model(input_ids=input_ids, use_cache=False)
            print(f"  chunk {i+1}/{len(chunks)} ok", flush=True)
    dt = time.time() - t0
    for h in handles:
        h.remove()

    # Apply normalize+rotate per layer.
    print(f"\n[setup] forward done ({dt:.1f}s); rotating per layer", flush=True)
    eng = KVCacheEngine(KVCacheEngineInfo(vector_dim=64))

    archive = {}
    stats = {"n_layers_captured": n_layers, "per_layer": {},
             "model": args.model, "mode": "A_clean",
             "n_samples": args.n_samples,
             "seq_len": args.seq_len, "fwd_wall_s": dt}
    t1 = time.time()
    for L in range(n_layers):
        if not captured_K[L]:
            continue
        K_raw = np.concatenate(captured_K[L], axis=0).astype(np.float32)
        V_raw = np.concatenate(captured_V[L], axis=0).astype(np.float32)
        K_rot = rotate_only(eng, K_raw)
        V_rot = rotate_only(eng, V_raw)
        archive[f"L{L}_K"] = K_rot
        archive[f"L{L}_V"] = V_rot
        cK = K_rot.reshape(-1)
        cV = V_rot.reshape(-1)
        stats["per_layer"][str(L)] = {
            "K_vectors": int(K_rot.shape[0]),
            "V_vectors": int(V_rot.shape[0]),
            "K_coord_std": float(cK.std()),
            "K_coord_max": float(np.abs(cK).max()),
            "V_coord_std": float(cV.std()),
            "V_coord_max": float(np.abs(cV).max()),
            "K_raw_max":   float(np.abs(K_raw).max()),
            "V_raw_max":   float(np.abs(V_raw).max()),
        }
        print(f"  L{L:>2}: K={K_rot.shape[0]:>5}  K_std={cK.std():.4f}  "
              f"K_max={np.abs(cK).max():.3f}  "
              f"(raw_max K={np.abs(K_raw).max():.2f} V={np.abs(V_raw).max():.2f})",
              flush=True)
    dt_rot = time.time() - t1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **archive)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2))
    print(f"\n[done] forward={dt:.1f}s  rotate={dt_rot:.1f}s  "
          f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
