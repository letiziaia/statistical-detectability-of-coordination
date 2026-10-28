"""Signed Charikar peeling on every layer of every joblib dump, one
layer at a time, plus the GLOBAL layer.
Score the peeling results against the ground-truth (precision/recall/AP/rho).

  * c_{uv} = min(-log(p_uv) - 1, c_max), c_max = 20.0, capped on the positive
    side only.
  * The peel is signed_peel: minimum weighted degree, epoch-based lazy heap,
    best prefix by rho(S) = sum of the edge weights inside S over |S|.
  * Only observed edges carry weight. A pair with no edge contributes nothing.
    This is the "untested" convention: a pair that never co-acted is treated as
    a pair that was never tested, not as a pair tested with p = 1.

S is the peeled set, rho(S) is its density, k_star = |S*| is the true operation size.

Usage
-----
    python size_agnostic.py --country 14141550     # one country
    python size_agnostic.py                        # every country
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import cohen_kappa_score

from heuristics.common import (
    DETECTABILITY_OUT_SUBDIR,
    GLOBAL_ATTR,
    GLOBAL_CACHE_SUFFIX,
    GLOBAL_LAYER,
    PVALUE_ATTR,
    average_precision,
    layer_graph_direct,
    load_ground_truth,
    log_macro_average_summary,
    normalize_account_id,
    rho_of,
    signed_peel,
    signed_peel_ordered,
    strip_self_loops,
)


def permutation_null(
    adj: dict[Any, dict[Any, float]],
    min_size: int,
    num_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null distribution of the peeled rho, by shuffling the edge weights across
    pairs while holding the topology fixed, and re-peeling.

    Run it at the SAME min_size as the analysis. At min_size 2 the best prefix is
    just the largest single weight, which permutation cannot change, so every
    replicate returns the same number and the null is degenerate.

    Args:
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        min_size: Smallest |S| the best prefix is taken from.
        num_replicates: Number of shuffled replicates to draw.
        rng: Random generator, for reproducibility.

    Returns:
        num_replicates peeled rho values under the shuffled null.
    """
    edge_pairs, edge_weights = [], []
    for u, neighbors in adj.items():
        for v, w in neighbors.items():
            if str(u) < str(v):
                edge_pairs.append((u, v))
                edge_weights.append(w)
    edge_weights = np.asarray(edge_weights, dtype=float)
    null_rho_samples = np.empty(num_replicates, dtype=float)
    for replicate in range(num_replicates):
        permuted_weights = rng.permutation(edge_weights)
        permuted_adj: dict[Any, dict[Any, float]] = {}
        for (u, v), w in zip(edge_pairs, permuted_weights):
            permuted_adj.setdefault(u, {})[v] = float(w)
            permuted_adj.setdefault(v, {})[u] = float(w)
        _, rho = signed_peel(list(permuted_adj.keys()), permuted_adj, min_size=min_size)
        null_rho_samples[replicate] = rho
    return null_rho_samples


def score_against_truth(
    S: set[Any],
    order: list[Any],
    adj: dict[Any, dict[Any, float]],
    nodes: list[Any],
    truth_ids: set[str],
    min_size: int,
    num_null_replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Every number the scoring table holds, for one layer of one country.

    Args:
        S: The peeled set.
        order: Removal order from signed_peel_ordered, for this layer.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        nodes: Every node of this layer's graph.
        truth_ids: Normalised account ids of the true operation (see
            normalize_account_id and load_ground_truth).
        min_size: Smallest |S| the best prefix is taken from.
        num_null_replicates: Permutation replicates for the null distribution
            of rho. 0 skips it.
        rng: Random generator, for the permutation null.

    Returns:
        One dict of scores, keyed as in the CSV: precision, recall,
        recall_reachable, f1, kappa, lift, precision_at_reachable,
        average_precision, rho_star, rho_hat, rho_ratio, rho_gap, and (when
        num_null_replicates > 0) the null_* columns.
    """
    normalized_id_by_node = {node: normalize_account_id(node) for node in nodes}
    reachable_positives = {
        node for node in nodes if normalized_id_by_node[node] in truth_ids
    }
    k_star = len(truth_ids)
    reachable = len(reachable_positives)

    true_positives = len(set(S) & reachable_positives)
    # Precision is undefined, not zero, when the layer holds no member of the
    # operation at all: the layer cannot get any right, so a 0 here would be
    # averaged in and would read as a failed detection. When the layer does hold
    # members and the peel misses them, 0 is the true answer and is kept.
    precision = true_positives / len(S) if (S and reachable) else float("nan")
    # Two recalls, because they answer different questions and only one is fair
    # to a sparse modality. `recall` is against the whole operation.
    # `recall_reachable` is against the members the layer actually contains: an
    # account that never posted a hashtag cannot be recovered from the hashtag
    # layer, so that is the ceiling the layer should be judged on.
    recall = true_positives / k_star if k_star else float("nan")
    recall_reachable = true_positives / reachable if reachable else float("nan")
    # F1 is undefined, not zero, when either input is undefined. NaN is truthy
    # in Python and NaN > 0 is False, so a plain `if precision and recall and
    # ...` guard silently reports 0.0 for a country with no ground truth and
    # drags the macro-average down. Test it explicitly.
    if not (np.isfinite(precision) and np.isfinite(recall)):
        f1 = float("nan")
    elif precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    chance_precision = reachable / len(nodes) if nodes else float("nan")

    # Cohen's kappa: corrects for chance agreement using all four confusion
    # cells, unlike balanced accuracy or G-mean, which both reduce to a
    # rescaling of recall here -- the class imbalance (k_star/n ~ 1/60,
    # |S| << n) pins specificity near 1 for almost any S, so those two carry
    # no information recall doesn't already have. Kappa doesn't collapse the
    # same way. Undefined under the same convention as precision: a layer
    # with no member of the operation has nothing to measure agreement
    # against.
    if reachable:
        is_true_positive = [node in reachable_positives for node in nodes]
        is_predicted_positive = [node in S for node in nodes]
        kappa = cohen_kappa_score(
            is_true_positive, is_predicted_positive, labels=[False, True]
        )
    else:
        kappa = float("nan")

    ranking = list(reversed(order))
    precision_at_reachable = (
        (len(set(ranking[:reachable]) & reachable_positives) / reachable)
        if reachable
        else float("nan")
    )
    average_precision_value = average_precision(ranking, reachable_positives)

    rho_star = rho_of(reachable_positives, adj)
    rho_hat = rho_of(S, adj)
    # The ratio is only readable when the truth itself is dense. When there is no
    # signal, rho(S*) sits at or below zero and the ratio explodes or changes
    # sign, so it is left undefined there and rho_gap carries the comparison.
    rho_ratio = (
        (rho_hat / rho_star)
        if (np.isfinite(rho_star) and rho_star > 0)
        else float("nan")
    )
    rho_gap = (rho_hat - rho_star) if np.isfinite(rho_star) else float("nan")

    scores = dict(
        k_star=k_star,
        reachable=reachable,
        ceiling=(reachable / k_star if k_star else float("nan")),
        tp=true_positives,
        precision=precision,
        recall=recall,
        recall_reachable=recall_reachable,
        f1=f1,
        kappa=kappa,
        chance_precision=chance_precision,
        lift=(
            (precision / chance_precision)
            if (
                np.isfinite(precision)
                and np.isfinite(chance_precision)
                and chance_precision > 0
            )
            else float("nan")
        ),
        precision_at_reachable=precision_at_reachable,
        average_precision=average_precision_value,
        rho_star=rho_star,
        rho_hat=rho_hat,
        rho_ratio=rho_ratio,
        rho_gap=rho_gap,
    )

    if num_null_replicates > 0:
        null_rho_samples = permutation_null(adj, min_size, num_null_replicates, rng)
        null_std = float(null_rho_samples.std())
        scores.update(
            null_B=num_null_replicates,
            null_mean=float(null_rho_samples.mean()),
            null_sd=null_std,
            null_z=(
                (rho_hat - null_rho_samples.mean()) / null_std
                if null_std > 0
                else float("nan")
            ),
            null_p=float(
                (1 + (null_rho_samples >= rho_hat).sum()) / (1 + num_null_replicates)
            ),
        )
    else:
        scores.update(
            null_B=0,
            null_mean=float("nan"),
            null_sd=float("nan"),
            null_z=float("nan"),
            null_p=float("nan"),
        )
    return scores


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def country_code_from_filename(layer_file_path: str) -> str | None:
    """Extract a country code from a layer file's name.

    Args:
        layer_file_path: e.g. ".../14141550_layers_fast" or
            ".../armenia_layers_fast".

    Returns:
        The leading run of digits in the basename (e.g. "14141550"), or, if
        the basename doesn't start with digits, whatever comes before
        "_layers_fast" (e.g. "armenia"). None if neither pattern matches.
    """
    file_basename = os.path.basename(layer_file_path)
    leading_digits_match = re.match(r"(\d+)", file_basename)
    if leading_digits_match:
        return leading_digits_match.group(1)
    name_before_suffix_match = re.search(r"([^/_]+)_layers_fast", file_basename)
    return name_before_suffix_match.group(1) if name_before_suffix_match else None


# ---------------------------------------------------------------------------
# One country
# ---------------------------------------------------------------------------
def _peel_one_layer(
    layer_name: str,
    graph: dict[Any, dict[Any, float]],
    country_code: str,
    args: argparse.Namespace,
    truth_ids: set[str] | None,
    rng: np.random.Generator,
    num_modalities: int | None = None,
) -> dict[str, Any] | None:
    """Peel one already-scored layer graph and log the result.

    Args:
        layer_name: The layer's name, or GLOBAL_LAYER.
        graph: graph[u][v] = c_uv, as returned by layer_graph_direct.
        country_code: This country's code, stored on the row.
        args: Parsed CLI args; reads min_size, c_max, null_b.
        truth_ids: This country's true operation ids, or None to peel
            without scoring.
        rng: Random generator, for the permutation null.
        num_modalities: Pass the number of per-country modalities to log
            layer_name as the cached GLOBAL combine; None for an ordinary
            layer.

    Returns:
        One row (see CSV_COLS-shaped keys in the module docstring), or None
        if this layer has fewer than args.min_size nodes.
    """
    nodes = list(graph.keys())
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    log_prefix = f"    layer '{layer_name}':"
    if num_modalities is not None:
        log_prefix = (
            f"    layer '{layer_name}': cached Fisher combine over "
            f"{num_modalities} modalities ('{GLOBAL_ATTR}'),"
        )
    if len(nodes) < args.min_size:
        logger.warning(f"{log_prefix} {len(nodes)} nodes, below --min-size; skipped")
        return None

    S, rho, order = signed_peel_ordered(nodes, graph, min_size=args.min_size)
    row: dict[str, Any] = dict(
        code=country_code,
        layer=str(layer_name),
        n_nodes=len(nodes),
        n_edges=num_edges,
        best_size=len(S),
        best_rho=rho,
        c_max=args.c_max,
        members=S,
    )
    score_summary = ""
    if truth_ids:
        truth_scores = score_against_truth(
            S, order, graph, nodes, truth_ids, args.min_size, args.null_b, rng
        )
        row.update(truth_scores)
        score_summary = (
            f"  P={truth_scores['precision']:.2f} R={truth_scores['recall']:.2f} "
            f"Rr={truth_scores['recall_reachable']:.2f} "
            f"AP={truth_scores['average_precision']:.2f} "
            f"rho/rho*={truth_scores['rho_ratio']:.2f}"
        )
    logger.info(
        f"{log_prefix} nodes={len(nodes)} edges={num_edges} "
        f"-> |S|={len(S)} rho={rho:.4f}{score_summary}"
    )
    return row


def peel_country(
    layers_by_name: dict[str, nx.Graph],
    country_code: str,
    args: argparse.Namespace,
    truth_ids: set[str] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Peel every layer of one country, then the GLOBAL layer. Returns rows.

    When truth_ids is given, every row also carries the scoring block: set
    recovery against the operation, the oracle density rho(S*), and optionally
    a permutation null for rho.

    Args:
        layers_by_name: This country's layer graphs, keyed by layer name.
            Mutated in place (self-loops are stripped).
        country_code: This country's code, stored on every row.
        args: Parsed CLI args; reads min_size, c_max, null_b, global_dir.
        truth_ids: This country's true operation ids, or None to peel
            without scoring.
        rng: Random generator, for the permutation null. A fresh
            default_rng(0) is used if not given.

    Returns:
        One row per layer that reached min_size, including GLOBAL when its
        cache is found.
    """
    rows: list[dict[str, Any]] = []
    rng = rng if rng is not None else np.random.default_rng(0)

    num_self_loops_removed = strip_self_loops(layers_by_name)
    if num_self_loops_removed:
        logger.info(f"  removed {num_self_loops_removed} self-loop(s) before peeling")

    # One layer at a time, so only one layer's scores are ever in memory.
    for layer_name in list(layers_by_name.keys()):
        try:
            graph = layer_graph_direct(
                layers_by_name[layer_name], PVALUE_ATTR, args.c_max
            )
        except TypeError as e:
            logger.warning(f"    layer '{layer_name}': [skip] {e}")
            continue
        if not graph:
            logger.warning(f"    layer '{layer_name}': no scored edges; skipped")
            continue
        row = _peel_one_layer(layer_name, graph, country_code, args, truth_ids, rng)
        if row is not None:
            rows.append(row)
        del graph

    # GLOBAL: read the precomputed graph (detectability.py's
    # build_global_pvalue_graph, cached to <global-dir>/<code>_global.joblib)
    # and use its p_global edge attribute directly. No Fisher combine is done
    # in this script; the cache is the only source.
    try:
        cache_path = os.path.join(
            args.global_dir, f"{country_code}{GLOBAL_CACHE_SUFFIX}"
        )
        if not os.path.isfile(cache_path):
            logger.warning(
                f"    layer '{GLOBAL_LAYER}': [skip] no cached graph at {cache_path}"
            )
        else:
            global_graph = joblib.load(cache_path)
            graph = layer_graph_direct(global_graph, GLOBAL_ATTR, args.c_max)
            del global_graph
            if graph:
                row = _peel_one_layer(
                    GLOBAL_LAYER,
                    graph,
                    country_code,
                    args,
                    truth_ids,
                    rng,
                    num_modalities=len(layers_by_name),
                )
                if row is not None:
                    rows.append(row)
            else:
                logger.warning(f"    layer '{GLOBAL_LAYER}': no scored edges; skipped")
    except Exception as e:
        logger.warning(f"    layer '{GLOBAL_LAYER}': [skip] {e!r}")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
CSV_DROP: tuple[str, ...] = ("members",)


def save_csv_snapshot(rows: list[dict[str, Any]], csv_path: str) -> pd.DataFrame:
    """Write every peeled row collected so far to csv_path (overwriting it),
    and return that same table -- so the caller never has to rebuild it.

    Args:
        rows: One dict per peeled layer, collected so far in the run. Each
            row's "members" set (the peeled account ids) is dropped before
            writing, since a CSV cell cannot hold a Python set.
        csv_path: Where to write the CSV. Nothing is written if rows is
            empty.

    Returns:
        rows as a DataFrame, with "members" dropped.
    """
    peeling_results = pd.DataFrame(
        [{k: v for k, v in row.items() if k not in CSV_DROP} for row in rows]
    )
    if rows:
        peeling_results.to_csv(csv_path, index=False)
    return peeling_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument strings to parse. None (the default) makes argparse
            read from sys.argv.

    Returns:
        The parsed arguments, plus two derived, non-CLI attributes:
        global_dir and coords_csv (see DETECTABILITY_OUT_SUBDIR).
    """
    ap = argparse.ArgumentParser(
        description="Signed Charikar peeling on each layer and on the Fisher-combined "
        "GLOBAL layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--layers-dir", required=True)
    ap.add_argument("--layers-glob", default="*_layers_fast*")
    ap.add_argument("--out-dir", default="./peeling_out")
    ap.add_argument(
        "--c-max",
        type=float,
        default=20.0,
        help="Winsorization cap on the score, as in compute_c_scores. Default "
        "20.0, which is p ~ 2e-9. Note that a 100-draw resampling null "
        "floors ONE object's p-value at 1/101, but p_modality and p_global "
        "are Fisher combinations of many such p-values and go far below "
        "that, so do not set the cap from the single-test floor.",
    )
    ap.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Smallest set the best prefix is taken from.",
    )
    ap.add_argument(
        "--truth-root",
        required=True,
        help="Root holding one sub-directory per country code with the "
        "ground-truth parquet.",
    )
    ap.add_argument(
        "--no-score",
        action="store_true",
        help="Peel only, do not score against the ground truth.",
    )
    ap.add_argument(
        "--null-b",
        type=int,
        default=0,
        help="Permutation replicates for the null distribution of rho. 0 (the "
        "default) skips it. 200 is what you want for a reported number; "
        "at ~300k edges one peel is about 0.4s, so 200 replicates cost "
        "roughly 1.2 min per layer per country. The empirical p is "
        "floored at 1/(B+1).",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="Seed for the permutation null."
    )
    ap.add_argument(
        "--country",
        default=None,
        help="Process only the layer file whose country_code_from_filename "
        "equals this code, instead of every file matched by --layers-glob. "
        "Output goes to <out-dir>/peeling_results_<country>.csv rather than "
        "the shared peeling_results.csv, and the cross-country macro-average "
        "and performance plots are skipped (run peel_layers.py, or a merge "
        "step, once every array task has finished). Used by "
        "peel_layers_parallel.sbatch to run one country per array task.",
    )
    args = ap.parse_args(argv)
    # Not CLI flags: detectability.py always leaves these under this fixed
    # subdirectory of --layers-dir, so there is nothing to configure.
    args.global_dir = os.path.join(args.layers_dir, DETECTABILITY_OUT_SUBDIR)
    args.coords_csv = os.path.join(args.global_dir, "layer_coordinates.csv")
    return args


def _write_member_files(
    country_rows: list[dict[str, Any]], country_code: str, members_dir: str
) -> None:
    """Write one members_*.txt per row (its peeled account ids, sorted), and
    record its path on the row as "members_file".

    Args:
        country_rows: This country's rows, as returned by peel_country.
            Mutated in place: each row gains a "members_file" key.
        country_code: This country's code, used in every filename.
        members_dir: Directory to write into.
    """
    for row in country_rows:
        safe_layer_name = re.sub(r"[^0-9A-Za-z._-]+", "_", str(row["layer"]))
        members_path = os.path.join(
            members_dir, f"{country_code}__{safe_layer_name}.txt"
        )
        with open(members_path, "w") as member_file:
            member_file.write(
                "\n".join(str(m) for m in sorted(row["members"], key=str))
            )
            member_file.write("\n" if row["members"] else "")
        row["members_file"] = members_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    members_dir = os.path.join(args.out_dir, "members")
    os.makedirs(members_dir, exist_ok=True)
    layer_files = sorted(glob.glob(os.path.join(args.layers_dir, args.layers_glob)))
    if not layer_files:
        sys.exit(
            f"No layer files under {args.layers_dir} matching '{args.layers_glob}'."
        )

    if args.country is not None:
        layer_files = [
            f for f in layer_files if country_code_from_filename(f) == args.country
        ]
        if not layer_files:
            sys.exit(
                f"No layer file under {args.layers_dir} matching '{args.layers_glob}' "
                f"has country code '{args.country}'."
            )

    rows: list[dict[str, Any]] = []
    csv_path = (
        os.path.join(args.out_dir, f"peeling_results_{args.country}.csv")
        if args.country is not None
        else os.path.join(args.out_dir, "peeling_results.csv")
    )

    for layer_file_path in layer_files:
        country_code = country_code_from_filename(layer_file_path)
        if country_code is None:
            logger.error(
                f"[skip] could not extract a country code from {layer_file_path}"
            )
            continue
        logger.info(
            f"=== {os.path.basename(layer_file_path)}  (code {country_code}) ==="
        )
        try:
            layers_by_name = joblib.load(layer_file_path)
        except Exception as e:
            logger.error(f"  [skip] could not load: {e!r}")
            save_csv_snapshot(rows, csv_path)
            continue
        if not isinstance(layers_by_name, dict):
            logger.error(
                f"  [skip] expected dict of layer->graph, got {type(layers_by_name)}"
            )
            del layers_by_name
            continue

        truth_ids = None
        if not args.no_score:
            try:
                truth_ids, id_column = load_ground_truth(args.truth_root, country_code)
                logger.info(
                    f"  ground truth: {len(truth_ids)} in-group accounts "
                    f"(id column '{id_column}')"
                )
            except Exception as e:
                logger.warning(f"  [warn] no scoring for this country: {e!r}")

        country_rows = peel_country(
            layers_by_name,
            country_code,
            args,
            truth_ids=truth_ids,
            rng=np.random.default_rng(args.seed),
        )
        _write_member_files(country_rows, country_code, members_dir)
        rows.extend(country_rows)
        del layers_by_name
        save_csv_snapshot(rows, csv_path)

    if not rows:
        sys.exit("Nothing was peeled.")
    peeling_results = save_csv_snapshot(rows, csv_path)
    logger.info(f"Wrote {csv_path} with {len(rows)} rows, members under {members_dir}/")

    if args.country is not None:
        logger.info(
            "--country was given: skipping the macro-average summary and the "
            "performance plots, both of which pool every country. Once every "
            "array task has finished, run merge_peeling_results.py to combine "
            "every peeling_results_*.csv and reproduce them."
        )
        return

    if "average_precision" not in peeling_results:
        return
    log_macro_average_summary(peeling_results)


if __name__ == "__main__":
    main()
