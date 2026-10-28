"""Spectral recovery .

The matrix has Y_uv = min(-log p_uv - 1, c_max) for a tested pair,
0 for an untested pair, 0 on the diagonal.

Returned sets (one CSV row each, column `method`):
    spec_topk    the k_l nodes with the largest entries in the top eigenvector;
    spec_clean   rank every node by its summed score to spec_topk, keep the top k_l;
    spec_prefix  size-agnostic: the prefix of the ranking by top eigenvectorv_1
                 having the largest rho(S) = W(S) = sum_{i<j in S} Y_ij / |S|.
k_l = labelled accounts present in the layer.

Usage
-----
  python spectral.py --country 14189075 --out-dir spectral_out --null-b 100
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

import heuristics.common as hrc

DENSE_MAX_N = 400  # below this, a dense eigh is cheaper and never fails to converge


def sym_matrix(n: int, r: np.ndarray, c: np.ndarray, w: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.concatenate([w, w]), (np.concatenate([r, c]), np.concatenate([c, r]))),
        shape=(n, n),
    )


def top_eigs(
    A: sp.csr_matrix, num: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """The `num` largest algebraic eigenpairs, in decreasing order."""
    n = A.shape[0]
    num = min(num, n)
    if n <= DENSE_MAX_N or num >= n - 1:
        vals, vecs = np.linalg.eigh(A.toarray())
        return vals[::-1][:num], vecs[:, ::-1][:, :num]
    v0 = rng.uniform(0.5, 1.5, n)  # positive start: overlaps any positive block
    try:
        vals, vecs = eigsh(A, k=num, which="LA", v0=v0, maxiter=20 * n, tol=1e-8)
    except ArpackNoConvergence as e:
        logger.warning(
            f"      ARPACK did not converge ({e}); using the converged pairs"
        )
        vals, vecs = e.eigenvalues, e.eigenvectors
        if len(vals) == 0:
            vals, vecs = np.linalg.eigh(A.toarray())
    order = np.argsort(vals)[::-1][:num]
    return vals[order], vecs[:, order]


def lambda_null(
    n: int,
    r: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """lambda_1 under B permutations of the edge weights, topology fixed."""
    out = np.empty(B)
    for b in range(B):
        out[b] = top_eigs(sym_matrix(n, r, c, rng.permutation(w)), 1, rng)[0][0]
    return out


def best_prefix(
    order: np.ndarray,
    n: int,
    r: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    min_size: int,
) -> tuple[int, float]:
    """Size m maximising rho(first m of `order`), m >= min_size, in O(edges)."""
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    enters = np.maximum(pos[r], pos[c])  # edge counted once both ends are in
    inc = np.bincount(enters, weights=w, minlength=n)
    total = np.cumsum(inc)  # total[m-1] = weight inside first m
    sizes = np.arange(1, n + 1)
    rho = total / sizes
    rho[: max(min_size, 1) - 1] = -np.inf
    m = int(np.argmax(rho)) + 1
    return m, float(rho[m - 1])


def total_weight(idx: np.ndarray, A: sp.csr_matrix) -> float:
    sub = A[idx][:, idx]
    return float(sub.sum()) / 2.0


def spectral_layer(
    code: str,
    layer: str,
    adj: dict[Any, dict[Any, float]],
    truth_ids: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, set[Any]]]:
    nodes = list(adj.keys())
    n = len(nodes)
    index = {u: i for i, u in enumerate(nodes)}
    r, c, w = hrc.upper_edges(adj, index)
    K = hrc.labelled_in_layer(nodes, truth_ids)
    k_l = len(K)
    rng = hrc.layer_rng(args.seed, code, layer)
    base = dict(
        code=code, layer=layer, n_l=n, n_edges=len(w), k_l=k_l, k_all=len(truth_ids)
    )
    if n < max(args.min_size, 3) or k_l < 2:
        logger.warning(f"    layer '{layer}': n={n} k_l={k_l}, too small; skipped")
        return [], {}

    A = sym_matrix(n, r, c, w)
    vals, vecs = top_eigs(A, 2, rng)
    lam1 = float(vals[0])
    lam2 = float(vals[1]) if len(vals) > 1 else float("nan")
    v = vecs[:, 0]
    # Label-free sign choice: the orientation whose top-k_l set is heavier.
    top_pos = np.argsort(-v, kind="stable")[:k_l]
    top_neg = np.argsort(v, kind="stable")[:k_l]
    if total_weight(top_neg, A) > total_weight(top_pos, A):
        v = -v
    order = np.argsort(-v, kind="stable")
    ipr = float(np.sum(v**4))
    sigma2 = float(2 * np.sum(w**2) / (n * (n - 1)))

    S_topk_idx = order[:k_l]
    deg_to_S = np.asarray(A[:, S_topk_idx].sum(axis=1)).ravel()
    S_clean_idx = np.argsort(-deg_to_S, kind="stable")[:k_l]
    m, _ = best_prefix(
        order, n, r, c, w, args.min_size
    )  # its rho is rho_hat of spec_prefix
    S_prefix_idx = order[:m]

    ranking = [nodes[i] for i in order]
    ap = hrc.average_precision(ranking, K)
    rho_star = hrc.rho_of(K, adj)

    detect = dict(
        lambda1=lam1,
        lambda2=lam2,
        lambda_edge_iid=2 * math.sqrt(n * sigma2),
        ipr=ipr,
        eff_support=1 / ipr if ipr > 0 else float("nan"),
        ap_ranking=ap,
        rho_star=rho_star,
    )
    if args.null_b > 0:
        null = lambda_null(n, r, c, w, args.null_b, rng)
        sd = float(null.std())
        detect.update(
            null_B=args.null_b,
            null_lambda_mean=float(null.mean()),
            null_lambda_sd=sd,
            null_lambda_z=(lam1 - null.mean()) / sd if sd > 0 else float("nan"),
            null_lambda_p=float((1 + (null >= lam1).sum()) / (1 + args.null_b)),
        )
    else:
        detect.update(
            null_B=0,
            null_lambda_mean=np.nan,
            null_lambda_sd=np.nan,
            null_lambda_z=np.nan,
            null_lambda_p=np.nan,
        )

    rows, members = [], {}
    for method, idx in (
        ("spec_topk", S_topk_idx),
        ("spec_clean", S_clean_idx),
        ("spec_prefix", S_prefix_idx),
    ):
        S = {nodes[i] for i in idx}
        members[method] = S
        row = dict(base, method=method)
        row.update(hrc.set_metrics(S, K, n))
        row["rho_hat"] = hrc.rho_of(S, adj)
        row.update(detect)
        rows.append(row)

    p_txt = f" p={detect['null_lambda_p']:.3f}" if args.null_b > 0 else ""
    logger.info(
        f"    layer '{layer}': n={n} k_l={k_l} lambda1={lam1:.1f} "
        f"(iid edge {detect['lambda_edge_iid']:.1f}{p_txt}) 1/ipr={detect['eff_support']:.1f} "
        f"AP={ap:.2f} | "
        + " ".join(f"{r_['method']}: F1={r_['f1']:.2f} |S|={r_['size']}" for r_ in rows)
    )
    return rows, members


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--layers-dir", required=True, help="Directory of layer files.")
    ap.add_argument(
        "--truth-root", required=True, help="Directory of ground-truth files."
    )
    ap.add_argument("--out-dir", default="./spectral_out")
    ap.add_argument(
        "--country", default=None, help="One country code; default: all 26."
    )
    ap.add_argument("--c-max", type=float, default=20.0)
    ap.add_argument(
        "--min-size", type=int, default=5, help="Smallest set spec_prefix may return."
    )
    ap.add_argument(
        "--null-b",
        type=int,
        default=100,
        help="Permutations for the lambda_1 null; 0 skips.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    codes = [args.country] if args.country else hrc.COUNTRY_CODES
    os.makedirs(args.out_dir, exist_ok=True)
    for code in codes:
        logger.info(f"=== {code} ===")
        try:
            truth_ids, _ = hrc.load_ground_truth(args.truth_root, code)
        except Exception as e:
            logger.error(f"  [skip] no ground truth: {e!r}")
            continue
        rows = []
        for layer, adj in hrc.iter_score_graphs(args.layers_dir, code, args.c_max):
            layer_rows, members = spectral_layer(code, layer, adj, truth_ids, args)
            rows.extend(layer_rows)
            for method, S in members.items():
                hrc.write_members(
                    os.path.join(args.out_dir, "members", method), code, layer, S
                )
            del adj
        out = os.path.join(args.out_dir, f"spectral_{code}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.info(f"  wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
