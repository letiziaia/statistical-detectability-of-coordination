"""Scoring and ground truth"""

from __future__ import annotations

import glob
import heapq
import math
import os
import re
import zlib
from collections.abc import Iterator
from typing import Any

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from scipy.stats import hypergeom
from sklearn.metrics import average_precision_score

COUNTRY_CODES = [
    "14141550",
    "14188947",
    "14188959",
    "14188970",
    "14188975",
    "14188984",
    "14189015",
    "14189018",
    "14189028",
    "14189037",
    "14189038",
    "14189041",
    "14189047",
    "14189048",
    "14189053",
    "14189058",
    "14189061",
    "14189072",
    "14189075",
    "14189078",
    "14189081",
    "14189086",
    "14189095",
    "14189098",
    "14189107",
    "14189110",
]

# The edge attribute for single layers
PVALUE_ATTR = "p_modality"

GLOBAL_LAYER = "GLOBAL"
GLOBAL_ATTR = "p_global"
GLOBAL_CACHE_SUFFIX = "_global.joblib"

# Outpus from detectability.py
DETECTABILITY_OUT_SUBDIR = "detectability_out"


def safe_layer_name(layer: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(layer))


def layer_rng(seed: int, code: str, layer: str) -> np.random.Generator:
    """Each (seed, country, layer) gets its own random number generator, so
    the random numbers used for one layer don't depend on which other layers
    were computed before it.
    """
    return np.random.default_rng(
        [
            seed,
            int(code) if code.isdigit() else zlib.crc32(code.encode()),
            zlib.crc32(str(layer).encode()),
        ]
    )


def iter_score_graphs(
    layers_dir: str, code: str, c_max: float, include_global: bool = True
) -> Iterator[tuple[str, dict[Any, dict[Any, float]]]]:
    """Yield (layer_name, adj) for every modality of a country, then GLOBAL.

    adj[u][v] = c_uv as built by layer_graph_direct. Layers that hold no
    scored edge, or whose attribute is not a single p-value, are skipped with
    a warning.
    """
    layer_path = os.path.join(layers_dir, f"{code}_layers_fast")
    layers_by_name = joblib.load(layer_path)
    strip_self_loops(layers_by_name)
    for name in list(layers_by_name.keys()):
        try:
            adj = layer_graph_direct(layers_by_name[name], PVALUE_ATTR, c_max)
        except TypeError as e:
            logger.warning(f"    layer '{name}': [skip] {e}")
            continue
        del layers_by_name[name]
        if not adj:
            logger.warning(f"    layer '{name}': no scored edges; skipped")
            continue
        yield str(name), adj
    del layers_by_name

    if not include_global:
        return
    cache = os.path.join(
        layers_dir, DETECTABILITY_OUT_SUBDIR, f"{code}{GLOBAL_CACHE_SUFFIX}"
    )
    if not os.path.isfile(cache):
        logger.warning(f"    layer '{GLOBAL_LAYER}': [skip] no cached graph at {cache}")
        return
    global_graph = joblib.load(cache)
    adj = layer_graph_direct(global_graph, GLOBAL_ATTR, c_max)
    del global_graph
    if adj:
        yield GLOBAL_LAYER, adj
    else:
        logger.warning(f"    layer '{GLOBAL_LAYER}': no scored edges; skipped")


def strip_self_loops(layers_by_name: dict[str, nx.Graph]) -> int:
    """Remove self-loops in place and return how many were dropped.

    They are removed because the three parts of this script disagree about
    them: compute_c_scores keeps a self-loop, rho_of counts its weight and then
    halves it, and permutation_null drops it. Rather than patch the verbatim
    copies, take them out before anything sees them. Real layer dumps normally
    have none, so this usually reports 0.

    Args:
        layers_by_name: This country's layer graphs, mutated in place.

    Returns:
        The number of self-loop edges removed, across every layer.
    """
    num_removed = 0
    for graph in layers_by_name.values():
        loops = list(getattr(graph, "selfloop_edges", list)())
        if not loops:
            loops = [(u, v) for u, v in graph.edges() if u == v]
        if loops:
            graph.remove_edges_from(loops)
            num_removed += len(loops)
    return num_removed


def labelled_in_layer(nodes: list[Any], truth_ids: set[str]) -> set[Any]:
    """K_l: the layer's nodes that are labelled accounts."""
    return {u for u in nodes if normalize_account_id(u) in truth_ids}


def set_metrics(S: set[Any], K: set[Any], n_l: int) -> dict[str, float]:
    """Recovery of the returned set S against K_l, as in the paper (Sec. 3.4).

    Recall is reachable recall (against K_l). beats_chance is the one-sided
    hypergeometric test Pr(T >= t) < 0.05 for a random set of |S| nodes;
    majority = P >= 0.5, R >= 0.5 and P > k_l / n_l.
    """
    s, k_l = len(S), len(K)
    t = len(S & K)
    nan = float("nan")
    precision = t / s if (s and k_l) else nan
    recall = t / k_l if k_l else nan
    if not (np.isfinite(precision) and np.isfinite(recall)):
        f1 = nan
    else:
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    chance = k_l / n_l if n_l else nan
    hyper_p = float(hypergeom.sf(t - 1, n_l, k_l, s)) if (t > 0 and k_l and s) else 1.0
    return dict(
        size=s,
        tp=t,
        precision=precision,
        recall=recall,
        f1=f1,
        jaccard=t / (s + k_l - t) if (s + k_l - t) else nan,
        chance_precision=chance,
        lift=precision / chance if (np.isfinite(precision) and chance > 0) else nan,
        hyper_p=hyper_p,
        beats_chance=bool(t > 0 and hyper_p < 0.05),
        ef1_random=2 * s * k_l / (n_l * (s + k_l)) if (n_l and s + k_l) else nan,
        majority=bool(
            np.isfinite(precision)
            and precision >= 0.5
            and recall >= 0.5
            and precision > chance
        ),
    )


def upper_edges(
    adj: dict[Any, dict[Any, float]], index: dict[Any, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each undirected edge appears once, as integer arrays (row < col, weight)."""
    rows, cols, w = [], [], []
    for u, nbrs in adj.items():
        iu = index[u]
        for v, c in nbrs.items():
            iv = index[v]
            if iu < iv:
                rows.append(iu)
                cols.append(iv)
                w.append(c)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(w, dtype=float),
    )


def adj_from_edges(
    nodes: list[Any], r: np.ndarray, c: np.ndarray, w: np.ndarray
) -> dict[Any, dict[Any, float]]:
    """Inverse of upper_edges: a dict-of-dicts score graph over `nodes`."""
    adj: dict[Any, dict[Any, float]] = {}
    for i, j, x in zip(r.tolist(), c.tolist(), w.tolist()):
        u, v = nodes[i], nodes[j]
        adj.setdefault(u, {})[v] = x
        adj.setdefault(v, {})[u] = x
    return adj


def write_members(members_dir: str, code: str, layer: str, S: set[Any]) -> None:
    os.makedirs(members_dir, exist_ok=True)
    with open(
        os.path.join(members_dir, f"{code}__{safe_layer_name(layer)}.txt"), "w"
    ) as fh:
        fh.write("\n".join(str(m) for m in sorted(S, key=str)))
        fh.write("\n" if S else "")


# ---------------------------------------------------------------------------
# Edge scoring with Winsorization
# ---------------------------------------------------------------------------
def compute_c_scores(
    layers: dict[str, nx.Graph], p_attr: str, c_max: float = 20.0
) -> tuple[dict[Any, dict[Any, dict[str, float]]], list[str]]:
    """
    c_{uvl} = min(-log(p) - 1, c_max).

    Null-centered (E[c]=0 under H0), Winsorized at c_max to prevent
    extreme p-values from dominating.

    The cap is applied ONLY on the positive side.  Negative scores
    (evidence against coordination) are left uncapped because they
    are bounded by construction: c >= -1 (when p = 1).

    Args:
        layers: layer name -> layer graph.
        p_attr: Edge attribute with the p-value.
        c_max: Winsorization, positive side only.

    Returns:
        (adj, layer_names): adj[u][v][layer_name] = c_{uvl}; layer_names is
        list(layers.keys()).
    """
    adj: dict = {}
    layer_names = list(layers.keys())

    for l_name, G in layers.items():
        for u, v, d in G.edges(data=True):
            p = d.get(p_attr, 1.0)
            c = -math.log(max(p, 1e-300)) - 1.0
            c = min(c, c_max)  # Winsorize
            adj.setdefault(u, {}).setdefault(v, {})[l_name] = c
            adj.setdefault(v, {}).setdefault(u, {})[l_name] = c

    return adj, layer_names


def build_layer_graph(
    adj_scores: dict[Any, dict[Any, dict[str, float]]],
    layer_subset: frozenset[str] | set[str],
) -> dict[Any, dict[Any, float]]:
    """
    Edge weights c_{uv}(L') = sum_{l in L'} c_{uvl}.
    Under H0: E[c_{uv}(L')] = 0 for any L'.

    Args:
        adj_scores: adj[u][v][layer_name] = c_{uvl}, as returned by
            compute_c_scores.
        layer_subset: L', the layer names to sum over.

    Returns:
        graph[u][v] = c_{uv}(L'), for every pair with a nonzero weight.
    """
    graph: dict = {}
    seen: set = set()

    for u, neighbors in adj_scores.items():
        for v, ldict in neighbors.items():
            if (u, v) in seen or (v, u) in seen:
                continue
            seen.add((u, v))

            w = sum(ldict.get(l, 0.0) for l in layer_subset)
            if abs(w) > 1e-15:
                graph.setdefault(u, {})[v] = w
                graph.setdefault(v, {})[u] = w

    return graph


# ---------------------------------------------------------------------------
# Building one layer's graph in a single pass
#
# compute_c_scores returns the scores for every layer as nested dicts, and
# build_layer_graph loops over that whole structure once per layer. The total
# work grows with (number of layers) x (number of edges), and every layer is
# held in memory at once.
#
# When only one layer is needed, it is more efficient to build the graph in a single pass.
# The weight of a pair in a layer is just its score in that layer.
# This function returns the same dict in time proportional to the number of edges.
# ---------------------------------------------------------------------------
def layer_graph_direct(
    G: nx.Graph, p_attr: str, c_max: float = 20.0
) -> dict[Any, dict[Any, float]]:
    """build_layer_graph(compute_c_scores({l: G})[0], {l}), in one pass.

    Args:
        G: The single layer's graph.
        p_attr: Edge attribute holding the p-value to score.
        c_max: Winsorization cap, positive side only.

    Returns:
        graph[u][v] = c_uv, for every pair with a nonzero weight.

    Raises:
        TypeError: p_attr holds a list rather than a single p-value.
    """
    graph: dict = {}
    for u, v, d in G.edges(data=True):
        if u == v:
            continue  # self-loops are stripped before the peel
        p = d.get(p_attr, 1.0)
        if isinstance(p, (list, tuple, set, np.ndarray)):
            raise TypeError(
                f"edge attribute '{p_attr}' holds a list, not a number. The "
                f"per-layer peel needs one p-value per edge (PVALUE_ATTR = "
                f"'{PVALUE_ATTR}')."
            )
        c = -math.log(max(float(p), 1e-300)) - 1.0
        c = min(c, c_max)  # Winsorize, positive side only
        if abs(c) > 1e-15:
            graph.setdefault(u, {})[v] = c
            graph.setdefault(v, {})[u] = c
    return graph


# ---------------------------------------------------------------------------
# Signed Charikar peeling (epoch-based heap)
# ---------------------------------------------------------------------------
def signed_peel(
    all_nodes: list[Any], adj: dict[Any, dict[Any, float]], min_size: int = 2
) -> tuple[set[Any], float]:
    """
    Signed greedy peeling.  Returns (best_S, best_rho).
    rho = sum(edge weights in S) / |S|.

    Args:
        all_nodes: Every node the peel may keep or remove.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct or
            build_layer_graph.
        min_size: Smallest |S| the best prefix is taken from.

    Returns:
        (best_S, best_rho): the best-scoring prefix and its rho. best_rho is
        -inf if no prefix reaches min_size.
    """
    current = set(all_nodes)
    if len(current) < min_size:
        return set(), -math.inf

    wdeg: dict = {}
    for u in all_nodes:
        wdeg[u] = sum(w for v, w in adj.get(u, {}).items() if v in current)

    epoch: dict = {u: 0 for u in all_nodes}
    heap = [(wdeg[u], 0, u) for u in all_nodes]
    heapq.heapify(heap)

    total = sum(wdeg.values()) / 2.0
    n = len(current)
    best_rho = total / n if n >= min_size else -math.inf
    best_S = set(current) if n >= min_size else set()
    removed: set = set()

    while len(current) > 1:
        while heap:
            d, ep, u = heapq.heappop(heap)
            if u not in removed and ep == epoch[u]:
                break
        else:
            break

        removed.add(u)
        current.discard(u)
        total -= wdeg[u]
        n -= 1

        for v, w in adj.get(u, {}).items():
            if v not in current:
                continue
            wdeg[v] -= w
            epoch[v] += 1
            heapq.heappush(heap, (wdeg[v], epoch[v], v))

        if n >= min_size:
            rho = total / n
            if rho > best_rho:
                best_rho = rho
                best_S = set(current)

    return best_S, best_rho


def signed_peel_ordered(
    all_nodes: list[Any], adj: dict[Any, dict[Any, float]], min_size: int = 2
) -> tuple[set[Any], float, list[Any]]:
    """signed_peel, returning also the order in which nodes were deleted, so
    list(reversed(order)) ranks the accounts from most core to least.

    Args:
        all_nodes: nodes the peel may keep or remove.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct or
            build_layer_graph.
        min_size: smallest |S| the best prefix is taken from.

    Returns:
        (best_S, best_rho, order): as in signed_peel, and removal
        order.
    """
    current = set(all_nodes)
    order: list = []
    if len(current) < min_size:
        return set(), -math.inf, sorted(current, key=str)

    wdeg: dict = {}
    for u in all_nodes:
        wdeg[u] = sum(w for v, w in adj.get(u, {}).items() if v in current)

    epoch: dict = {u: 0 for u in all_nodes}
    heap = [(wdeg[u], 0, u) for u in all_nodes]
    heapq.heapify(heap)

    total = sum(wdeg.values()) / 2.0
    n = len(current)
    best_rho = total / n if n >= min_size else -math.inf
    best_S = set(current) if n >= min_size else set()
    removed: set = set()

    while len(current) > 1:
        while heap:
            d, ep, u = heapq.heappop(heap)
            if u not in removed and ep == epoch[u]:
                break
        else:
            break

        removed.add(u)
        current.discard(u)
        order.append(u)
        total -= wdeg[u]
        n -= 1

        for v, w in adj.get(u, {}).items():
            if v not in current:
                continue
            wdeg[v] -= w
            epoch[v] += 1
            heapq.heappush(heap, (wdeg[v], epoch[v], v))

        if n >= min_size:
            rho = total / n
            if rho > best_rho:
                best_rho = rho
                best_S = set(current)

    order.extend(sorted(current, key=str))  # the survivors are the most core
    return best_S, best_rho, order


def normalize_account_id(raw_id: object) -> str | None:
    """Normalize one account id to a stripped string.

    Graph node ids and parquet ids can differ in type (int64 vs str).

    Args:
        raw_id: A graph node id or a parquet column value.

    Returns:
        The id as a stripped string, or None if raw_id is None.
    """
    if raw_id is None:
        return None
    if isinstance(raw_id, float) and raw_id.is_integer():
        raw_id = int(raw_id)
    return str(raw_id).strip()


def load_ground_truth(truth_root: str, country_code: str) -> tuple[set[str], str]:
    """Accounts of the operation for one country: any account with a row
    where is_control is False.

    Args:
        truth_root: Directory containing one subdirectory per country code,
            each holding that country's ground-truth parquet file(s).
        country_code: The country's numeric code, matching a subdirectory
            name under truth_root.

    Returns:
        (operation_account_ids, id_column): a set of normalised account ids
        (see normalize_account_id), and the name of the column they were
        read from.

    Raises:
        FileNotFoundError: truth_root has no subdirectory for country_code.
        KeyError: the parquet has no 'is_control' column, or no column that
            looks like an account id.
    """
    truth_dir = os.path.join(truth_root, str(country_code))
    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f"no ground-truth directory: {truth_dir}")

    try:
        truth_df = pd.read_parquet(truth_dir)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
        # Parts of one directory can disagree on a column's type (an all-null part is
        # typed `null`), which breaks the single-schema read: read part by part instead.
        truth_part_paths = [
            path
            for path in sorted(glob.glob(os.path.join(truth_dir, "*.parquet")))
            if not os.path.basename(path).startswith(("_", "."))
        ]
        if not truth_part_paths:
            raise
        truth_df = pd.concat(
            (pd.read_parquet(path) for path in truth_part_paths), ignore_index=True
        )
    if "is_control" not in truth_df.columns:
        raise KeyError(
            f"'is_control' absent in {truth_dir}; columns {list(truth_df.columns)}"
        )

    # Candidate column names for the account id
    # same as in ACCOUNT_ID_COLUMN_NAME from detectability.py.
    id_column = "accountid"

    is_control_column = truth_df["is_control"]
    if is_control_column.dtype == bool:
        is_control = is_control_column
    else:
        is_control = is_control_column.map(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "t", "yes")
        )

    account_ids = truth_df[id_column].map(normalize_account_id)
    is_in_operation_by_account = (~is_control).groupby(account_ids).any()
    operation_account_ids = set(
        is_in_operation_by_account[is_in_operation_by_account].index.tolist()
    )
    return operation_account_ids, id_column


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rho_of(S: set[Any], adj: dict[Any, dict[Any, float]]) -> float:
    """
    Args:
        S: The set to score, e.g. the peeled set or the true operation.
        adj: adj[u][v] = c_uv, as returned by layer_graph_direct.

    Returns:
        rho(S), or NaN if |S| < 2.
    """
    S = set(S)
    if len(S) < 2:
        return float("nan")
    total_weight = sum(w for u in S for v, w in adj.get(u, {}).items() if v in S)
    return total_weight / 2.0 / len(S)


def average_precision(ranking: list[Any], positives: set[Any]) -> float:
    """Average precision over a ranking from most core to least.

    Args:
        ranking: Nodes from most to least core (see signed_peel_ordered).
        positives: The operation's account ids, in the same id space as
            ranking (both graph node ids, both raw, or both normalised).

    Returns:
        Average precision, or NaN if positives is empty
    """
    if not positives:
        return float("nan")
    is_positive = [node in positives for node in ranking]
    descending_score = list(range(len(ranking), 0, -1))
    return float(average_precision_score(is_positive, descending_score))


# ---------------------------------------------------------------------------
# The GLOBAL layer: Fisher over the per-pair p-value lists
# ---------------------------------------------------------------------------
def resolve_global_attr(
    layers_by_name: dict[str, nx.Graph], forced_attr: str | None = None
) -> str:
    """Edge attribute GLOBAL is combined from.

    Prefer p_values, the raw list with one p-value per shared object. Fall back to
    p_modality only if there is no list.

    Args:
        layers_by_name: This country's layer graphs, keyed by layer name.
        forced_attr: Skip detection and return this instead, if given.

    Returns:
        "p_values" if any edge of any layer carries it, else "p_modality".
    """
    if forced_attr:
        return forced_attr
    for graph in layers_by_name.values():
        for _, _, edge_attrs in graph.edges(data=True):
            return "p_values" if "p_values" in edge_attrs else "p_modality"
    return "p_modality"


def log_macro_average_summary(peeling_results: pd.DataFrame) -> None:
    """Log the macro-average of every score column, by layer.

    Args:
        peeling_results: DataFrame with one row per country-layer, holding
            the scores of that layer against the ground truth.
    """
    score_columns = [
        "precision",
        "recall",
        "recall_reachable",
        "f1",
        "kappa",
        "lift",
        "precision_at_reachable",
        "average_precision",
        "rho_gap",
    ]
    if peeling_results.get("null_B", pd.Series([0])).max() > 0:
        score_columns.append("null_z")
    score_columns = [c for c in score_columns if c in peeling_results]

    summary = peeling_results.groupby("layer")[score_columns].mean(numeric_only=True)
    # rho_ratio is undefined wherever rho(S*) <= 0, and otherwise skewed, so compute the median.
    if "rho_ratio" in peeling_results:
        summary["rho_ratio_med"] = peeling_results.groupby("layer")[
            "rho_ratio"
        ].median()
    summary.insert(0, "n_countries", peeling_results.groupby("layer")["code"].nunique())
    logger.info(
        "Macro-average over countries, by layer:\n" + summary.round(3).to_string()
    )
