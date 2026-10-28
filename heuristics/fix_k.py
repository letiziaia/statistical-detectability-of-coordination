"""Every peeled set here is searched at exactly one fixed size, k = |truth_ids|,
the number of distinct accounts the ground truth puts in the operation for that country
(every account with one is_control=False row; see load_ground_truth).
k is read off the ground truth per country
--k-mode layer instead sets k per layer to k_l, the number of operation accounts
present in that layer
--k-mode truth, is the k = |truth_ids| described here.

Because enumerating every size-k subset (C(n, k), n in the thousands) is
infeasible, the fixed-size peel is a heuristic: starts from a seed, refines,
and keeps the best of a few random-restart.

S is the peeled set, rho(S) is its density, k_star = |S*| is the operation size.

Usage
-----
    python fix_k.py --country 14141550  # one country, for one array task
    python fix_k.py                     # every country with ground truth
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import combine_pvalues
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
    signed_peel_ordered,
    strip_self_loops,
)


# ---------------------------------------------------------------------------
# Fixed-size-k search: greedy seed + local-search swap refinement
#
# signed_peel finds the best-scoring PREFIX over every size from min_size to
# n. The fix-k analysis wants exactly one size: k = |truth_ids|, the number
# of distinct accounts the ground truth puts in the operation for this
# country (see load_ground_truth). Enumerating every k-subset (C(n, k), n in
# the thousands) is infeasible, so this is a heuristic: seed with the greedy
# peel at |current| == k -- order[-k:] from signed_peel_ordered. The seed is then
# locally improved by repeatedly swapping out its weakest member for the
# strongest node outside it, accepting only swaps that increase the total
# edge weight inside S. A few random-restart seeds (uniform k-subsets,
# refined the same way) are tried alongside the greedy one, and the best restart wins.
# ---------------------------------------------------------------------------
def _local_search_swap(
    all_nodes: list[Any],
    adj: dict[Any, dict[Any, float]],
    S0: set[Any],
    max_iters: int,
) -> tuple[set[Any], float]:
    """Single-swap hill-climbing at fixed |S| = |S0|.

    Each iteration removes the weakest member (least weight to the rest of
    S) from S and adds the strongest node outside S (most weight to S).
    Iterations stop when that swap would not increase the total edge weight inside S
    (a local optimum for single swaps) or max_iters is reached.

    Args:
        all_nodes: Every node the search may place in or out of S.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        S0: Starting set, held at this fixed size throughout.
        max_iters: Cap on the number of swaps.

    Returns:
        (S, total): the refined set and the sum of edge weights inside it
        (i.e. total = rho(S) * |S|).
    """
    S = set(S0)
    deg_S: dict[Any, float] = {}
    for s in S:
        for t, w in adj.get(s, {}).items():
            deg_S[t] = deg_S.get(t, 0.0) + w
    total = sum(deg_S.get(s, 0.0) for s in S) / 2.0

    for _ in range(max_iters):
        worst = min(S, key=lambda s: deg_S.get(s, 0.0), default=None)
        best_outside = max(
            (v for v in all_nodes if v not in S),
            key=lambda v: deg_S.get(v, 0.0),
            default=None,
        )
        if worst is None or best_outside is None:
            break
        w_swap = adj.get(worst, {}).get(best_outside, 0.0)
        delta = deg_S.get(best_outside, 0.0) - w_swap - deg_S.get(worst, 0.0)
        if delta <= 1e-12:  # no single swap improves S any further
            break
        S.remove(worst)
        S.add(best_outside)
        total += delta
        for t, w in adj.get(worst, {}).items():
            deg_S[t] = deg_S.get(t, 0.0) - w
        for t, w in adj.get(best_outside, {}).items():
            deg_S[t] = deg_S.get(t, 0.0) + w

    return S, total


def _densest_fixed_k_subset(
    all_nodes: list[Any],
    adj: dict[Any, dict[Any, float]],
    k: int,
    rng: np.random.Generator,
    seed_order: list[Any] | None = None,
    max_iters: int = 200,
    num_restarts: int = 3,
) -> tuple[set[Any], float]:
    """Heuristic search for the densest exactly-size-k subset.

    See the section comment above: the greedy seed (order[-k:], the
    signed_peel_ordered survivors at |current| == k) is refined by
    _local_search_swap, plus num_restarts - 1 random-subset restarts refined
    the same way. The restart with the highest total edge weight wins.

    Args:
        all_nodes: Every node the search may place in or out of S.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        k: Target |S|. Capped at len(all_nodes) if larger.
        rng: Random generator, for the random-restart seeds.
        seed_order: removal order in signed_peel_ordered for this graph, if
            the caller already computed it (avoids recomputing it here).
            None computes it here.
        max_iters: Cap on the number of swaps per restart.
        num_restarts: Number of restarts (the greedy seed, plus
            num_restarts - 1 random ones). At least 1 is used.

    Returns:
        (S, rho): |S| == min(k, len(all_nodes)), rho = rho(S).
    """
    n = len(all_nodes)
    if k <= 0 or n == 0:
        return set(), float("nan")
    if k >= n:
        S = set(all_nodes)
        return S, rho_of(S, adj)

    if seed_order is None:
        _, _, seed_order = signed_peel_ordered(all_nodes, adj, min_size=2)
    seeds = [set(seed_order[-k:])]
    for _ in range(max(num_restarts, 1) - 1):
        random_idx = rng.permutation(n)[:k]
        seeds.append({all_nodes[i] for i in random_idx})

    best_S: set[Any] = seeds[0]
    best_total = -math.inf
    for seed in seeds:
        S, total = _local_search_swap(all_nodes, adj, seed, max_iters)
        if total > best_total:
            best_S, best_total = S, total

    return best_S, best_total / k


def permutation_null(
    adj: dict[Any, dict[Any, float]],
    k: int,
    num_replicates: int,
    rng: np.random.Generator,
    ls_max_iters: int = 200,
    ls_restarts: int = 3,
) -> np.ndarray:
    """Null distribution of the fixed-size-k rho, by shuffling the edge weights
    across pairs while holding the topology fixed, and re-running the same
    _densest_fixed_k_subset heuristic used for the observed statistic.

    Needs to run at the SAME k, ls_max_iters and ls_restarts it gets compared to:
    a weaker search here than on the observed graph would systematically bias the
    null distribution of rho_hat regardless of any real signal, and
    inflate null_z/null_p.

    Args:
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        k: Target |S|, same as the observed statistic's.
        num_replicates: Number of shuffled replicates to draw.
        rng: Random generator, for reproducibility.
        ls_max_iters: Passed through to _densest_fixed_k_subset.
        ls_restarts: Passed through to _densest_fixed_k_subset.

    Returns:
        num_replicates fixed-size-k rho values under the shuffled null.
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
        _, rho = _densest_fixed_k_subset(
            list(permuted_adj.keys()),
            permuted_adj,
            k,
            rng,
            max_iters=ls_max_iters,
            num_restarts=ls_restarts,
        )
        null_rho_samples[replicate] = rho
    return null_rho_samples


def score_against_truth(
    S: set[Any],
    order: list[Any],
    adj: dict[Any, dict[Any, float]],
    nodes: list[Any],
    truth_ids: set[str],
    num_null_replicates: int,
    rng: np.random.Generator,
    ls_max_iters: int = 200,
    ls_restarts: int = 3,
    k: int | None = None,
) -> dict[str, float]:
    """All scores, for one layer of one country.

    Args:
        S: The fixed-size-k peeled set (see _densest_fixed_k_subset), searched
            at size k.
        order: Removal order from signed_peel_ordered, for this layer.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.
        nodes: Every node of this layer's graph.
        truth_ids: Normalised account ids of the true operation (see
            normalize_account_id and load_ground_truth).
        num_null_replicates: Permutation replicates for the null distribution
            of rho. 0 skips it.
        rng: Random generator, for the permutation null.
        ls_max_iters: Passed through to permutation_null's
            _densest_fixed_k_subset calls, so the null uses the same search
            budget as the observed statistic.
        ls_restarts: Passed through the same way.
        k: The size S was searched at, and so the size the permutation null
            is searched at. None means len(truth_ids) (--k-mode truth).

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
    # Precision is undefined when the layer holds no member of the operation at all.
    # When the layer does hold members and the peel misses them, precision is 0.
    precision = true_positives / len(S) if (S and reachable) else float("nan")
    # `recall` is against the whole operation.
    # `recall_reachable` is against the members of the operations contained in the layer:
    # an account that never posted a hashtag cannot be recovered from the hashtag layer.
    recall = true_positives / k_star if k_star else float("nan")
    recall_reachable = true_positives / reachable if reachable else float("nan")
    # F1 is undefined when either input is undefined.
    # NaN is truthy in Python and NaN > 0 is False
    if not (np.isfinite(precision) and np.isfinite(recall)):
        f1 = float("nan")
    elif precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    chance_precision = reachable / len(nodes) if nodes else float("nan")

    # Cohen's kappa: corrects for chance agreement
    # It is undefined under the same convention as precision
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
    # When there is no signal, rho(S*) is <= 0. This makes the ratio explode or change
    # sign, so we don't compute it.
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
        null_rho_samples = permutation_null(
            adj,
            k if k is not None else k_star,
            num_null_replicates,
            rng,
            ls_max_iters,
            ls_restarts,
        )
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


def build_global_pvalue_graph(
    layers_by_name: dict[str, nx.Graph],
    p_attr: str = "p_values",
    absent_layer_p: float | None = None,
    min_p: float = 1e-300,
    block_cells: int = 4_000_000,
) -> nx.Graph:
    """Build the graph whose edge attribute p_global is the Fisher combination of a
    pair's p-values across the layers.

    p_attr may hold a single p-value per edge or a list of them, one per shared
    object.

    absent_layer_p = None (the default, the "untested" option) combines only
    the layers a pair appears in. A pair seen in a single modality with a single
    shared object then keeps that p-value unchanged, since Fisher over one test
    is the identity. The number of tests varies from pair to pair, which leaves
    the null intact, because a Fisher-combined p-value is uniform under H0
    whatever the degrees of freedom, and changes only the power.

    absent_layer_p = 1.0 (corresponds to the "p1" option) adds ONE entry of that
    value for each LAYER a pair is missing from, so all pairs are combined over the
    same number of layers. The pad is per missing layer, not per missing object, since
    the object counts are a property of the pair. Padding costs two degrees of freedom
    per missing layer while adding no evidence, so a pair present in one modality of
    seven comes out with a LARGER p_global than its single observed p-value.

    Args:
        layers_by_name: layer graphs, keyed by layer name.
        p_attr: edge attribute to pool: a single p-value, or a list of them.
        absent_layer_p: See above. None (default) is the "untested"
            option; 1.0 is the "p1" option.
        min_p: floor applied to every p-value before combining.
        block_cells: Target size, in numbers, of each dense block handed to
            scipy.stats.combine_pvalues.

    Returns:
        A graph with every input node, and one edge per pair with at least
        one finite p-value, carrying GLOBAL_ATTR (p_global), "n_tests" (how
        many p-values were pooled) and "n_layers" (how many layers saw this
        pair).
    """

    def _pair_key(node_a: Any, node_b: Any) -> tuple[Any, Any]:
        return (node_a, node_b) if str(node_a) <= str(node_b) else (node_b, node_a)

    def _coerce_to_float_list(value: object) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [float(value)]
        try:
            return [float(x) for x in value]
        except TypeError:
            return []

    layer_names = list(layers_by_name.keys())
    num_layers = len(layer_names)
    pair_row_index: dict[tuple[Any, Any], int] = {}
    pairs_by_row: list[tuple[Any, Any]] = []
    pooled_row_ids: list[int] = []
    pooled_layer_indices: list[int] = []
    pooled_p_values: list[float] = []
    all_nodes: set[Any] = set()

    for layer_index, layer_name in enumerate(layer_names):
        graph = layers_by_name[layer_name]
        all_nodes.update(graph.nodes())
        for node_a, node_b, edge_attrs in graph.edges(data=True):
            if node_a == node_b:
                continue
            p_values_for_pair = [
                p
                for p in _coerce_to_float_list(edge_attrs.get(p_attr))
                if math.isfinite(p)
            ]
            if not p_values_for_pair:
                continue
            pair_key = _pair_key(node_a, node_b)
            row_id = pair_row_index.get(pair_key)
            if row_id is None:
                row_id = pair_row_index[pair_key] = len(pairs_by_row)
                pairs_by_row.append(pair_key)
            for p in p_values_for_pair:
                pooled_row_ids.append(row_id)
                pooled_layer_indices.append(layer_index)
                pooled_p_values.append(min(max(p, min_p), 1.0))

    global_graph = nx.Graph()
    global_graph.add_nodes_from(all_nodes)
    global_graph.graph["layer_names"] = layer_names
    if not pair_row_index:
        return global_graph

    pooled_row_ids_arr = np.asarray(pooled_row_ids, dtype=np.int64)
    pooled_layer_indices_arr = np.asarray(pooled_layer_indices, dtype=np.int64)
    pooled_p_values_arr = np.asarray(pooled_p_values, dtype=float)
    n_pairs = len(pairs_by_row)

    # p1: one pad per MISSING LAYER
    if absent_layer_p is not None:
        pad = float(min(max(absent_layer_p, min_p), 1.0))
        seen_matrix = np.zeros((n_pairs, num_layers), dtype=bool)
        seen_matrix[pooled_row_ids_arr, pooled_layer_indices_arr] = True
        n_layers_seen_per_pair = seen_matrix.sum(axis=1)
        missing_row_ids, _ = np.nonzero(~seen_matrix)
        pooled_row_ids_arr = np.concatenate([pooled_row_ids_arr, missing_row_ids])
        pooled_p_values_arr = np.concatenate(
            [pooled_p_values_arr, np.full(missing_row_ids.size, pad, dtype=float)]
        )
        del seen_matrix, missing_row_ids
    else:
        n_layers_seen_per_pair = (
            np.bincount(
                pooled_row_ids_arr * num_layers + pooled_layer_indices_arr,
                minlength=n_pairs * num_layers,
            )
            .reshape(n_pairs, num_layers)
            .astype(bool)
            .sum(axis=1)
        )
    del pooled_layer_indices_arr

    sort_order_by_row = np.argsort(pooled_row_ids_arr, kind="stable")
    sorted_row_ids = pooled_row_ids_arr[sort_order_by_row]
    sorted_p_values = pooled_p_values_arr[sort_order_by_row]
    row_start_offsets = np.searchsorted(sorted_row_ids, np.arange(n_pairs + 1))
    n_observed_tests_per_pair = np.diff(row_start_offsets)

    p_global_per_pair = np.empty(n_pairs, dtype=float)
    for group_observed_count in np.unique(
        n_observed_tests_per_pair[n_observed_tests_per_pair > 0]
    ):
        group_observed_count = int(group_observed_count)
        pair_indices_in_group = np.flatnonzero(
            n_observed_tests_per_pair == group_observed_count
        )
        block_size = max(1, block_cells // group_observed_count)
        offsets_within_group = np.arange(group_observed_count)
        for group_start in range(0, len(pair_indices_in_group), block_size):
            group_pair_indices = pair_indices_in_group[
                group_start : group_start + block_size
            ]
            gather_indices = np.repeat(
                row_start_offsets[group_pair_indices], group_observed_count
            ) + np.tile(offsets_within_group, len(group_pair_indices))
            observed_p_matrix = sorted_p_values[gather_indices].reshape(
                len(group_pair_indices), group_observed_count
            )
            p_global_per_pair[group_pair_indices] = combine_pvalues(
                observed_p_matrix, method="fisher", axis=1
            )[1]

    for row_id, (node_a, node_b) in enumerate(pairs_by_row):
        global_graph.add_edge(
            node_a,
            node_b,
            **{
                GLOBAL_ATTR: float(p_global_per_pair[row_id]),
                "n_tests": int(n_observed_tests_per_pair[row_id]),
                "n_layers": int(n_layers_seen_per_pair[row_id]),
            },
        )
    return global_graph


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def country_code_from_filename(layer_file_path: str) -> str | None:
    """Extract a country code from a layer file's name.

    Args:
        layer_file_path: e.g. ".../14141550" or
            ".../armenia".

    Returns:
        The leading run of digits in the basename (e.g. "14141550"), or, if
        the basename doesn't start with digits, whatever comes before
        "" (e.g. "armenia"). None if neither pattern matches.
    """
    file_basename = os.path.basename(layer_file_path)
    leading_digits_match = re.match(r"(\d+)", file_basename)
    if leading_digits_match:
        return leading_digits_match.group(1)
    name_before_suffix_match = re.search(r"([^/_]+)", file_basename)
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
    """Peel one already-scored layer graph to exactly k = |truth_ids| nodes
    (see _densest_fixed_k_subset) and log the result.

    Args:
        layer_name: The layer's name, or GLOBAL_LAYER.
        graph: graph[u][v] = c_uv, as returned by layer_graph_direct.
        country_code: This country's code, stored on the row.
        args: Parsed CLI args; reads c_max, null_b, ls_max_iters, ls_restarts.
        truth_ids: This country's true operation ids. k = len(truth_ids) is
            the fixed size every peeled set is searched at, so this is
            required, not optional: a layer is skipped without it.
        rng: Random generator, for the random-restart seeds and the
            permutation null.
        num_modalities: Pass the number of per-country modalities to log
            layer_name as the cached GLOBAL combine; None for an ordinary
            layer.

    Returns:
        One row (see CSV_COLS-shaped keys in the module docstring), or None
        if truth_ids is empty or this layer has no scored edges.
    """
    nodes = list(graph.keys())
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    log_prefix = f"    layer '{layer_name}':"
    if num_modalities is not None:
        log_prefix = (
            f"    layer '{layer_name}': cached Fisher combine over "
            f"{num_modalities} modalities ('{GLOBAL_ATTR}'),"
        )
    if not truth_ids:
        logger.warning(
            f"{log_prefix} no ground-truth accounts; k is undefined, skipped"
        )
        return None
    if args.k_mode == "layer":
        # k_l: the operation's accounts present in this layer's score graph,
        # the same set score_against_truth calls reachable_positives.
        k = sum(normalize_account_id(node) in truth_ids for node in nodes)
        if k < 2 or k >= len(nodes):
            logger.warning(
                f"{log_prefix} nodes={len(nodes)} k_l={k}; fixed-k is trivial, skipped"
            )
            return None
    else:
        k = len(truth_ids)

    # The full removal order is still a ranking over every size at once
    # (used below for AP and precision_at_reachable, which don't depend on
    # a cut point), and doubles as _densest_fixed_k_subset's greedy seed.
    _, _, order = signed_peel_ordered(nodes, graph, min_size=2)
    S, rho = _densest_fixed_k_subset(
        nodes,
        graph,
        k,
        rng,
        seed_order=order,
        max_iters=args.ls_max_iters,
        num_restarts=args.ls_restarts,
    )
    row: dict[str, Any] = dict(
        code=country_code,
        layer=str(layer_name),
        n_nodes=len(nodes),
        n_edges=num_edges,
        k_mode=args.k_mode,
        k_target=k,
        best_size=len(S),
        best_rho=rho,
        c_max=args.c_max,
        members=S,
    )
    truth_scores = score_against_truth(
        S,
        order,
        graph,
        nodes,
        truth_ids,
        args.null_b,
        rng,
        args.ls_max_iters,
        args.ls_restarts,
        k=k,
    )
    row.update(truth_scores)
    score_summary = (
        f"  P={truth_scores['precision']:.2f} R={truth_scores['recall']:.2f} "
        f"Rr={truth_scores['recall_reachable']:.2f} "
        f"AP={truth_scores['average_precision']:.2f} "
        f"rho/rho*={truth_scores['rho_ratio']:.2f}"
    )
    logger.info(
        f"{log_prefix} nodes={len(nodes)} edges={num_edges} k={k} "
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
    """Peel every layer of one country, then the GLOBAL layer, to exactly
    k = |truth_ids| nodes each (see _densest_fixed_k_subset). Returns rows.

    Every row carries the scoring block: set recovery against the
    operation, the oracle density rho(S*), and optionally a permutation
    null for rho.

    Args:
        layers_by_name: This country's layer graphs, keyed by layer name.
            Mutated in place (self-loops are stripped).
        country_code: This country's code, stored on every row.
        args: Parsed CLI args; reads c_max, null_b, ls_max_iters,
            ls_restarts, global_dir.
        truth_ids: This country's true operation ids. k = len(truth_ids)
            fixes the size every peeled set is searched at, so a country
            with no ground truth cannot be peeled at all here: an empty or
            None truth_ids skips the whole country.
        rng: Random generator, for the random-restart seeds and the
            permutation null. A fresh default_rng(0) is used if not given.

    Returns:
        One row per layer with scored edges, including GLOBAL when its
        cache is found. Empty if truth_ids is empty or None.
    """
    rows: list[dict[str, Any]] = []
    if not truth_ids:
        logger.warning(
            "  no ground-truth accounts for this country; k is undefined, "
            "the whole country is skipped"
        )
        return rows
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
    ap.add_argument("--layers-glob", default="**")
    # Distinct from peel_layers.py/peel_layers_parallel.py's "./peeling_out"
    # default: that run's best-any-size peel and this one's fixed-size-k peel
    # are different statistics, so they must never land in the same CSVs.
    ap.add_argument("--out-dir", default="./peeling_out_fix_k")
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
        "--truth-root",
        required=True,
        help="Root holding one sub-directory per country code with the "
        "ground-truth parquet.",
    )
    ap.add_argument(
        "--ls-max-iters",
        type=int,
        default=200,
        help="Cap on the number of single-swap local-search iterations per "
        "restart, in the fixed-size-k search (see _densest_fixed_k_subset). "
        "Also used for every permutation-null replicate, so it should not "
        "be set differently for a re-run whose null is meant to be "
        "compared against an earlier run's rho_hat.",
    )
    ap.add_argument(
        "--ls-restarts",
        type=int,
        default=3,
        help="Number of restarts in the fixed-size-k search: the greedy "
        "peel-order seed, plus this many minus one random k-subset seeds, "
        "each refined by local search. The best restart is kept. Also used "
        "for every permutation-null replicate, for the same reason as "
        "--ls-max-iters.",
    )
    ap.add_argument(
        "--null-b",
        type=int,
        default=0,
        help="Permutation replicates for the null distribution of the "
        "fixed-size-k rho. 0 (the default) skips it. Each replicate reruns "
        "the same fixed-k heuristic search (--ls-max-iters, --ls-restarts) "
        "on a shuffled graph, so it costs roughly ls_restarts times one "
        "peel's worth of work; set null-b with that multiplier in mind. "
        "The empirical p is floored at 1/(B+1).",
    )
    ap.add_argument(
        "--k-mode",
        choices=["truth", "layer"],
        default="truth",
        help="How the fixed size k is set. 'truth' (the default, and every "
        "run before this flag existed): k = |truth_ids|, the whole operation, "
        "for every layer. 'layer': k = k_l, the operation's accounts present "
        "in that layer's score graph (the 'reachable' set of "
        "score_against_truth); a layer "
        "with k_l < 2 or k_l >= n is skipped. The permutation null always "
        "uses the same k as the observed statistic.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the local-search random restarts and the permutation null.",
    )
    ap.add_argument(
        "--country",
        default=None,
        help="Process only the layer file whose country_code_from_filename "
        "equals this code. If not given, every layer file under --layers-dir is processed.",
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

        try:
            truth_ids, id_column = load_ground_truth(args.truth_root, country_code)
        except Exception as e:
            logger.error(
                f"  [skip] no ground truth for this country, so k (the fixed "
                f"peeled-set size) cannot be determined: {e!r}"
            )
            del layers_by_name
            continue
        if not truth_ids:
            logger.error(
                "  [skip] ground truth has zero in-group accounts; k=0, nothing to peel"
            )
            del layers_by_name
            continue
        k_description = (
            f"k = {len(truth_ids)}"
            if args.k_mode == "truth"
            else "k = k_l, those of them present in each layer"
        )
        logger.info(
            f"  ground truth: {len(truth_ids)} in-group accounts "
            f"(id column '{id_column}'); {k_description}"
        )

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
