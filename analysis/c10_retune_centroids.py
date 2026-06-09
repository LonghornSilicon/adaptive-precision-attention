"""C10: Lloyd-Max retune of per-layer centroid tables.

Reads the capture .npz from c10_capture_per_layer.py, runs the KVCE
ref-model's Lloyd-Max trainer (centroid_lloyd_max.lloyd_max) on each
layer's coords, and writes a JSON keyed by layer_idx::

    {
      "0": {"centroids": [...8 floats...], "boundaries": [...7 floats...]},
      "1": {...},
      ...
    }

This file is the input to acu_kvce_attention.set_centroid_tables() and
c10_run_ppl.py.

By default fits ONE table per layer on the union of K and V coords (the
chip stores one centroid table per engine; K and V share it). Pass
`--separate-kv` to fit K and V independently and emit two JSONs.

Run::

    python analysis/c10_retune_centroids.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
KVCE_REF = Path(os.environ.get(
    "KVCE_REF", "/home/chaithu/lhs/kv-cache-engine/sw/reference_model"))
sys.path.insert(0, str(KVCE_REF))

from centroid_lloyd_max import (  # noqa: E402
    lloyd_max, calibrate_qjl_scale,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--captures",
        default=str(REPO_ROOT / "analysis" / "c10_captures.npz"),
    )
    ap.add_argument("--n-levels", type=int, default=8,
                    help="Number of centroids per table. 8 = chip turbo4.")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "analysis" / "c10_centroid_tables.json"),
    )
    ap.add_argument("--max-samples-per-layer", type=int, default=400_000,
                    help="Cap on samples passed to Lloyd-Max per layer "
                         "(for wall time). 400k is well above the 200k the "
                         "centroid override test gate validated at.")
    ap.add_argument("--separate-kv", action="store_true",
                    help="Fit K and V tables independently. Default: union.")
    ap.add_argument("--layers", default="",
                    help="Comma-separated subset (e.g. '0,1,23'). "
                         "Empty = all layers found in the capture.")
    ap.add_argument("--skip-qjl-calibration", action="store_true",
                    help="Omit the per-layer qjl_scale field. The codec "
                         "will fall back to sqrt(pi/2) — the C11-era "
                         "behaviour that regressed PPL in the smoke test.")
    ap.add_argument("--qjl-calibration-samples", type=int, default=2000,
                    help="K-vector count for the QJL alpha calibration "
                         "(2D, full vectors). Calibration is O(d^2 * N) so "
                         "2-5k is plenty.")
    ap.add_argument("--rotation-seed", type=int, default=42,
                    help="Must match the engine's rotation_seed for the "
                         "Rademacher matrix to be the same one the codec "
                         "uses at PPL time.")
    args = ap.parse_args(argv)

    arc = np.load(args.captures)
    layers = sorted({int(k[1:].split("_")[0])
                     for k in arc.files if k.startswith("L")})
    if args.layers:
        wanted = set(int(s) for s in args.layers.split(",") if s)
        layers = [L for L in layers if L in wanted]
    print(f"[retune] captures: {args.captures}")
    print(f"[retune] layers: {layers}")
    print(f"[retune] n_levels={args.n_levels}  separate_kv={args.separate_kv}")

    rng = np.random.default_rng(0)
    out = {}
    t0 = time.time()
    for L in layers:
        K = arc[f"L{L}_K"].reshape(-1).astype(np.float64)
        V = arc[f"L{L}_V"].reshape(-1).astype(np.float64)

        def maybe_subsample(x: np.ndarray) -> np.ndarray:
            if x.size > args.max_samples_per_layer:
                idx = rng.choice(x.size, args.max_samples_per_layer, replace=False)
                return x[idx]
            return x

        # K vectors preserved as (n, d) for the QJL alpha calibration. K
        # is the only path that uses QJL (compress_value has no residual
        # sketch), so alpha is calibrated on K samples only.
        K_vecs = arc[f"L{L}_K"].astype(np.float64)
        if K_vecs.shape[0] > args.qjl_calibration_samples:
            cal_idx = rng.choice(K_vecs.shape[0],
                                 args.qjl_calibration_samples, replace=False)
            K_vecs_cal = K_vecs[cal_idx]
        else:
            K_vecs_cal = K_vecs

        if args.separate_kv:
            cK, bK = lloyd_max(maybe_subsample(K), n_levels=args.n_levels,
                               max_iter=args.max_iter, tol=args.tol)
            cV, bV = lloyd_max(maybe_subsample(V), n_levels=args.n_levels,
                               max_iter=args.max_iter, tol=args.tol)
            entry = {
                "K": {"centroids": cK, "boundaries": bK},
                "V": {"centroids": cV, "boundaries": bV},
            }
            if not args.skip_qjl_calibration:
                alpha_K = calibrate_qjl_scale(
                    K_vecs_cal, cK, bK, rotation_seed=args.rotation_seed)
                entry["K"]["qjl_scale"] = alpha_K
                print(f"  L{L:>2}: K range [{cK[0]:+.4f},{cK[-1]:+.4f}]  "
                      f"alpha_K={alpha_K:.4f}  "
                      f"V range [{cV[0]:+.4f},{cV[-1]:+.4f}]", flush=True)
            else:
                print(f"  L{L:>2}: K range [{cK[0]:+.4f},{cK[-1]:+.4f}]  "
                      f"V range [{cV[0]:+.4f},{cV[-1]:+.4f}]", flush=True)
            out[str(L)] = entry
        else:
            samples = np.concatenate([K, V])
            samples = maybe_subsample(samples)
            c, b = lloyd_max(samples, n_levels=args.n_levels,
                             max_iter=args.max_iter, tol=args.tol)
            entry = {"centroids": c, "boundaries": b}
            if not args.skip_qjl_calibration:
                alpha = calibrate_qjl_scale(
                    K_vecs_cal, c, b, rotation_seed=args.rotation_seed)
                entry["qjl_scale"] = alpha
                print(f"  L{L:>2}: range [{c[0]:+.4f},{c[-1]:+.4f}]  "
                      f"alpha={alpha:.4f}  "
                      f"std={float(np.std(samples)):.4f}  "
                      f"n_samples={samples.size}", flush=True)
            else:
                print(f"  L{L:>2}: range [{c[0]:+.4f},{c[-1]:+.4f}]  "
                      f"std={float(np.std(samples)):.4f}  "
                      f"n_samples={samples.size}", flush=True)
            out[str(L)] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.out}  wall={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
