"""Recovery of the coordinated group on the summed multiplex layer SUM.

For one country and one null model, the script builds one score graph, SUM,
from the per-modality layers, and runs three methods on it:

  peel         size-agnostic signed peeling (peel_one_layer)
  fix_k        fixed-size heuristic with k = |truth| (peel_one_layer)
  fix_k_layer  the same with --k-mode layer: k = k_l, the operation's accounts present in SUM
  spectral     top eigenvector (spectral.spectral_layer)

The entry of SUM for the pair (u, v) is

  Y_uv = sum over the modalities l of c^(l)_uv,
  c^(l)_uv = min(-log p^(l)_uv - 1, c_max) if the pair is tested in modality l, else 0.

Outputs in --out-dir:
  peel/peeling_results_<code>.csv      one row, layer = SUM
  peel/members/<code>__SUM.txt
  fix_k/peeling_results_<code>.csv     one row, layer = SUM
  fix_k/members/<code>__SUM.txt
  fix_k_layer/peeling_results_<code>.csv   one row, layer = SUM
  fix_k_layer/members/<code>__SUM.txt
  spectral/spectral_<code>.csv        one row per spectral method, layer = SUM
  spectral/members/<method>/<code>__SUM.txt
  sum_graph/<code>.joblib              the summed graph, read again on a rerun

Usage
-----
  python multiplex_sum_layers.py --country 14189075 \
      --layers-dir pvalue_coord --out-dir multiplex_sum_out --null-b 100
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

import heuristics.common as hrc
import heuristics.fix_k as hfk
import heuristics.size_agnostic as hsa
import heuristics.spectral as hs

SUM_LAYER = "SUM"
METHODS = ("peel", "fix_k", "fix_k_layer", "spectral")
DEFAULT_METHODS = ("peel", "fix_k", "spectral")


def _merge_keys(keys: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add the weights of equal pair keys. Return the distinct keys and their sums."""
    uniq, inv = np.unique(keys, return_inverse=True)
    return uniq, np.bincount(inv, weights=w, minlength=len(uniq))


def build_sum_graph(layers_dir: str, code: str, c_max: float) -> dict[str, Any]:
    """Sum the per-modality score graphs of one country.

    Args:
        layers_dir: Directory with <code>_layers_fast for this null model.
        code: Country code.
        c_max: Cap on the score of one pair in one modality.

    Returns:
        dict with nodes (list), rows, cols (int arrays, rows < cols), weights
        (float array, the summed scores, nonzero only) and layers (the names
        of the summed modalities).
    """
    index: dict[Any, int] = {}
    nodes: list[Any] = []
    keys = np.empty(0, dtype=np.int64)
    w = np.empty(0, dtype=float)
    names: list[str] = []
    for name, adj in hrc.iter_score_graphs(
        layers_dir, code, c_max, include_global=False
    ):
        if str(name).strip().upper() == hfk.GLOBAL_LAYER.upper():
            raise ValueError(
                f"layer file of {code} holds a layer named '{name}'; GLOBAL must not be summed"
            )
        for u in adj:
            if u not in index:
                index[u] = len(nodes)
                nodes.append(u)
        if len(nodes) >= 2**31:
            raise ValueError("too many nodes for 32-bit pair keys")
        r, c, x = hrc.upper_edges(adj, index)  # each tested pair once, r < c
        del adj
        keys, w = _merge_keys(
            np.concatenate([keys, (r << 32) | c]), np.concatenate([w, x])
        )
        names.append(str(name))
        logger.info(
            f"    added modality '{name}': {len(x)} tested pairs, {len(keys)} pairs in SUM so far"
        )
    keep = np.abs(w) > 1e-15  # the same rule as layer_graph_direct
    keys, w = keys[keep], w[keep]
    return dict(
        nodes=nodes, rows=keys >> 32, cols=keys & 0xFFFFFFFF, weights=w, layers=names
    )


def load_or_build(args: argparse.Namespace) -> dict[str, Any]:
    """Read the summed graph of args.country from its cache, or build and cache it."""
    path = os.path.join(args.out_dir, "sum_graph", f"{args.country}.joblib")
    if os.path.isfile(path):
        logger.info(f"  reading the summed graph from {path}")
        return joblib.load(path)
    g = build_sum_graph(args.layers_dir, args.country, args.c_max)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(g, path)
    logger.info(f"  wrote the summed graph to {path}")
    return g


def run_peel(
    adj: dict, truth_ids: set[str], args: argparse.Namespace, out: str, n_summed: int
) -> None:
    """Size-agnostic peeling, as in peel_layers_parallel.py."""
    a = hsa.parse_args(
        [
            "--layers-dir",
            args.layers_dir,
            "--truth-root",
            args.truth_root,
            "--c-max",
            str(args.c_max),
            "--min-size",
            str(args.min_size),
            "--null-b",
            str(args.null_b),
            "--seed",
            str(args.seed),
        ]
    )
    row = hsa._peel_one_layer(
        SUM_LAYER, adj, args.country, a, truth_ids, np.random.default_rng(a.seed)
    )
    rows = [] if row is None else [dict(row, n_summed_layers=n_summed)]
    members_dir = os.path.join(os.path.dirname(out), "members")
    os.makedirs(members_dir, exist_ok=True)
    hsa._write_member_files(rows, args.country, members_dir)
    hsa.save_csv_snapshot(rows, out)


def run_fix_k(
    adj: dict,
    truth_ids: set[str],
    args: argparse.Namespace,
    out: str,
    n_summed: int,
    k_mode: str = "truth",
) -> None:
    """Fixed-size heuristic: k = |truth| for k_mode 'truth',
    k = k_l (the operation's accounts present in SUM) for 'layer'.
    """
    a = hfk.parse_args(
        [
            "--layers-dir",
            args.layers_dir,
            "--truth-root",
            args.truth_root,
            "--c-max",
            str(args.c_max),
            "--null-b",
            str(args.null_b),
            "--seed",
            str(args.seed),
            "--k-mode",
            k_mode,
        ]
    )
    row = hfk._peel_one_layer(
        SUM_LAYER, adj, args.country, a, truth_ids, np.random.default_rng(a.seed)
    )
    rows = [] if row is None else [dict(row, n_summed_layers=n_summed)]
    members_dir = os.path.join(os.path.dirname(out), "members")
    os.makedirs(members_dir, exist_ok=True)
    hfk._write_member_files(rows, args.country, members_dir)
    hfk.save_csv_snapshot(rows, out)


def run_spectral(
    adj: dict, truth_ids: set[str], args: argparse.Namespace, out: str, n_summed: int
) -> None:
    """Spectral method, as in spectral.py."""
    a = argparse.Namespace(min_size=args.min_size, null_b=args.null_b, seed=args.seed)
    rows, members = hs.spectral_layer(args.country, SUM_LAYER, adj, truth_ids, a)
    for method, S in members.items():
        hrc.write_members(
            os.path.join(os.path.dirname(out), "members", method),
            args.country,
            SUM_LAYER,
            S,
        )
    pd.DataFrame([dict(r, n_summed_layers=n_summed) for r in rows]).to_csv(
        out, index=False
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--country", required=True)
    ap.add_argument(
        "--layers-dir",
        required=True,
        help="Directory with <code>_layers_fast of one null model.",
    )
    ap.add_argument(
        "--truth-root", required=True, help="Directory of ground-truth files."
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated subset of peel,fix_k,fix_k_layer,spectral.",
    )
    ap.add_argument("--c-max", type=float, default=20.0)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--null-b", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        sys.exit(f"unknown methods: {sorted(unknown)}")
    outs = {
        "peel": os.path.join(
            args.out_dir, "peel", f"peeling_results_{args.country}.csv"
        ),
        "fix_k": os.path.join(
            args.out_dir, "fix_k", f"peeling_results_{args.country}.csv"
        ),
        "fix_k_layer": os.path.join(
            args.out_dir, "fix_k_layer", f"peeling_results_{args.country}.csv"
        ),
        "spectral": os.path.join(
            args.out_dir, "spectral", f"spectral_{args.country}.csv"
        ),
    }
    todo = [m for m in methods if not os.path.isfile(outs[m])]
    for m in set(methods) - set(todo):
        logger.info(f"  {m}: {outs[m]} exists; skipped")
    if not todo:
        return

    logger.info(f"=== {args.country}, layers {args.layers_dir} ===")
    truth_ids, id_column = hfk.load_ground_truth(args.truth_root, args.country)
    logger.info(f"  ground truth: {len(truth_ids)} accounts (id column '{id_column}')")
    g = load_or_build(args)
    logger.info(
        f"  SUM = sum of {len(g['layers'])} modalities: {', '.join(g['layers'])}"
    )
    adj = hrc.adj_from_edges(g["nodes"], g["rows"], g["cols"], g["weights"])
    logger.info(
        f"  SUM: {len(adj)} nodes, {len(g['weights'])} pairs with a nonzero summed score"
    )

    run = {
        "peel": run_peel,
        "fix_k": run_fix_k,
        "fix_k_layer": lambda *a: run_fix_k(*a, k_mode="layer"),
        "spectral": run_spectral,
    }
    for m in todo:
        os.makedirs(os.path.dirname(outs[m]), exist_ok=True)
        logger.info(f"  --- {m} ---")
        run[m](adj, truth_ids, args, outs[m], len(g["layers"]))
        logger.info(f"  wrote {outs[m]}")


if __name__ == "__main__":
    main()
