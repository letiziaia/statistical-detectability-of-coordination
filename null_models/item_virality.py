"""Item-virality null model.

Same pipeline (observed nearest-subsequent delta-t per user pair and object,
empirical p-value against null replicates, Fisher combination across objects).

Each user's null timestamps are drawn from the pool of all timestamps seen for
the object (any user).

For an object, a user with L observed timestamps gets L null timestamps per replicate,
sampled uniformly WITH REPLACEMENT (i.i.d.) from that pool of t timestamps. The pool
includes the user's own timestamps and keeps multiplicity (one entry per post), so
a user may draw the same timestamp more than once within a replicate.
"""

from __future__ import annotations

import gc
import hashlib
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import click
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from loguru import logger
from scipy.stats import combine_pvalues
from tqdm import tqdm

from null_models.common import (
    INSTRUMENT_OBJECT_SIZE_THRESHOLD,
    MODALITIES,
    NULL_MEANS_VECTORIZE_PRODUCT_CAP,
    compute_pvalue,
    iter_object_records,
    load_parquet,
    matched_delta_t_list,
    null_means_from_replicates,
    prepare_modality_df,
)

logger = logger
logger.configure(
    handlers=[
        {
            "sink": sys.stderr,
            "format": "<cyan>{extra}</cyan> | "
            + "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
            + "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            + "<level>{message}</level>",
            "serialize": False,
        }
    ]
)


def _stable_object_user_seed(
    user: object, modality: str, object_key: object, salt: int = 0
) -> int:
    """Seed a user's null resampling by (modality, object, user) so it can be
    computed once per object and reused across every partner sharing that object."""
    payload = f"{user!r}|{modality!r}|{object_key!r}|{salt}".encode(
        "utf-8", errors="surrogatepass"
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _sorted_datetime64_array(timestamps: list[datetime] | np.ndarray) -> np.ndarray:
    array = np.asarray(timestamps, dtype="datetime64[ns]")
    if array.size == 0:
        return array
    return np.sort(array)


def _subsequent_deltas(
    source: np.ndarray,
    target: np.ndarray,
    strict: bool,
) -> np.ndarray:
    side = "right" if strict else "left"
    idx = np.searchsorted(target, source, side=side)
    mask = idx < target.size
    if not np.any(mask):
        return np.empty(0, dtype=float)

    matched_target = target[idx[mask]]
    matched_source = source[mask]
    return (matched_target - matched_source) / np.timedelta64(1, "m")


def resample_from_pool_vectorized(
    pool: np.ndarray,
    length: int,
    n_tries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_tries null copies of a user's `length` timestamps at once (shape (n_tries, length)).

    Item-based null model: each null timestamp is drawn uniformly (with replacement)
    from `pool`, the timestamps of every post that used this object, by any user.
    """
    if length == 0 or pool.size == 0:
        return np.empty((n_tries, 0), dtype="datetime64[ns]")
    idx = rng.integers(0, pool.size, size=(n_tries, length))
    return pool[idx]


def precompute_object_replicates(
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    n_tries: int,
    modality: str,
    object_key: object,
) -> dict[object, np.ndarray]:
    """Resample each user's timestamps once per object, reused across all its pairs.

    The pool is all timestamps seen for this object (across all users); each user
    gets as many null timestamps as they have observed ones.
    """
    pool = np.concatenate(
        [np.asarray(timestamps_by_user[user], dtype="datetime64[ns]") for user in users]
    )
    replicates: dict[object, np.ndarray] = {}
    for user in users:
        length = len(timestamps_by_user[user])
        seed = _stable_object_user_seed(user, modality, object_key)
        rng = np.random.default_rng(seed)
        replicates[user] = resample_from_pool_vectorized(pool, length, n_tries, rng)
    return replicates


def _null_means_from_replicates_scalar(
    reps_u: np.ndarray,
    reps_v: np.ndarray,
) -> np.ndarray:
    n_tries = reps_u.shape[0]
    null_means = np.empty(n_tries, dtype=float)
    for i in range(n_tries):
        gaps = matched_delta_t_list(reps_u[i], reps_v[i])
        null_means[i] = float(gaps.mean()) if gaps.size > 0 else float(np.inf)
    return null_means


def _null_means_from_replicates_vectorized(
    reps_u: np.ndarray,
    reps_v: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of calling matched_delta_t_list per try, for all
    tries at once. Mathematically equivalent to _subsequent_deltas: sorting is
    only an implementation detail of searchsorted, not part of its definition
    (the nearest-subsequent element is the same regardless of how it's found).
    """
    u_ns = reps_u.astype("int64")  # epoch-ns, exact
    v_ns = reps_v.astype("int64")
    diff = (v_ns[:, None, :] - u_ns[:, :, None]).astype(np.float64)  # (n_tries, Lu, Lv)

    pos = np.where(diff > 0, diff, np.inf)
    min_pos = pos.min(axis=2)  # (n_tries, Lu): nearest strictly-subsequent v per u
    neg = np.where(diff <= 0, -diff, np.inf)
    min_neg = neg.min(axis=1)  # (n_tries, Lv): nearest at-or-after u per v

    combined = np.concatenate([min_pos, min_neg], axis=1)  # (n_tries, Lu+Lv), ns
    finite_mask = np.isfinite(combined)
    counts = finite_mask.sum(axis=1)
    # Must mask before summing: a plain combined.sum(axis=1) would collapse any
    # row containing an unmatched (inf) entry to inf entirely, silently biasing
    # every null mean with at least one unmatched element.
    sums = np.where(finite_mask, combined, 0.0).sum(axis=1)
    null_means_ns = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
    return null_means_ns / 6e10  # ns -> minutes


def _process_object(
    object_key: object,
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    replicates = precompute_object_replicates(
        users, timestamps_by_user, n_tries, modality, object_key
    )

    k = len(users)
    pair_count = k * (k - 1) // 2
    instrument = k >= INSTRUMENT_OBJECT_SIZE_THRESHOLD
    if instrument:
        # logger.contextualize's contextvars don't cross the joblib "processes"
        # backend's process boundary, so bind explicitly using our own args.
        bound_logger = logger.bind(
            modality=modality, object_key=object_key, k=k, pair_count=pair_count
        )
        bound_logger.info("Starting large object")
        start_time = time.perf_counter()
        last_heartbeat = start_time
        vectorized_pairs = 0
        scalar_pairs = 0
    scored = 0

    results: list[tuple[object, object, float | None]] = []
    for left_idx in range(len(users) - 1):
        user_left = users[left_idx]
        ts_left = timestamps_by_user[user_left]
        for right_idx in range(left_idx + 1, len(users)):
            user_right = users[right_idx]
            ts_right = timestamps_by_user[user_right]

            observed_iat = matched_delta_t_list(ts_left, ts_right)
            if observed_iat.size == 0:
                results.append((user_left, user_right, None))
            else:
                reps_left = replicates[user_left]
                reps_right = replicates[user_right]
                null_means = null_means_from_replicates(reps_left, reps_right)
                results.append(
                    (user_left, user_right, compute_pvalue(observed_iat, null_means))
                )
                if instrument:
                    if (
                        reps_left.shape[1] * reps_right.shape[1]
                        <= NULL_MEANS_VECTORIZE_PRODUCT_CAP
                    ):
                        vectorized_pairs += 1
                    else:
                        scalar_pairs += 1

            if instrument:
                scored += 1
                if scored % 10_000 == 0:
                    now = time.perf_counter()
                    if now - last_heartbeat >= 60:
                        bound_logger.info(
                            "Large object heartbeat: {}/{} pairs scored "
                            "({} vectorized, {} scalar), {:.0f}s elapsed",
                            scored,
                            pair_count,
                            vectorized_pairs,
                            scalar_pairs,
                            now - start_time,
                        )
                        last_heartbeat = now

    if instrument:
        bound_logger.info(
            "Finished large object: {:.0f}s elapsed ({} vectorized, {} scalar)",
            time.perf_counter() - start_time,
            vectorized_pairs,
            scalar_pairs,
        )
    return results


def _process_object_batch(
    batch: list[tuple[object, list[object], dict[object, list[datetime]]]],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    results: list[tuple[object, object, float | None]] = []
    for object_key, users, timestamps_by_user in batch:
        results.extend(
            _process_object(object_key, users, timestamps_by_user, n_tries, modality)
        )
    return results


def build_coordination_network(
    mod_df: pd.DataFrame,
    modality: str,
    n_tries: int = 100,
    n_jobs: int = -1,
    pair_budget_per_task: int = 20_000,
) -> nx.Graph:
    """Build a coordination layer for one modality using per-object batched dispatch.

    Work is dispatched at the object-batch level (not per pair): each object's users
    are resampled once (from the object's pooled timestamps) and reused across all of that object's C(k,2) pairs, and each
    joblib task handles a whole batch of objects bounded by `pair_budget_per_task`
    pairs, instead of one process-pool round trip per pair.
    """
    edge_pvalues: dict[tuple[object, object], list[float]] = defaultdict(list)

    effective_n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    group_capacity = max(effective_n_jobs * 4, 1)

    sub_batch: list[tuple[object, list[object], dict[object, list[datetime]]]] = []
    sub_batch_pairs = 0
    dispatch_group: list[
        list[tuple[object, list[object], dict[object, list[datetime]]]]
    ] = []

    def run_group() -> None:
        if not dispatch_group:
            return
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_process_object_batch)(group_batch, n_tries, modality)
            for group_batch in dispatch_group
        )
        for batch_results in results:
            for user1, user2, pvalue in batch_results:
                if pvalue is not None:
                    edge_pvalues[(user1, user2)].append(pvalue)
        dispatch_group.clear()

    object_iter = iter_object_records(mod_df, modality)
    with tqdm(total=None, desc=f"Scoring {modality}", unit="pair") as progress:
        for object_key, users, timestamps_by_user in object_iter:
            k = len(users)
            pair_count = k * (k - 1) // 2
            sub_batch.append((object_key, users, timestamps_by_user))
            sub_batch_pairs += pair_count
            progress.update(pair_count)

            if sub_batch_pairs >= pair_budget_per_task:
                dispatch_group.append(sub_batch)
                sub_batch = []
                sub_batch_pairs = 0
                if len(dispatch_group) >= group_capacity:
                    run_group()

        if sub_batch:
            dispatch_group.append(sub_batch)
        run_group()

    graph = nx.Graph()
    for (user1, user2), pvalues in edge_pvalues.items():
        res = combine_pvalues(pvalues, method="fisher")
        graph.add_edge(
            user1,
            user2,
            p_values=pvalues,
            weight=1 - res.pvalue,
            p_modality=res.pvalue,
        )

    return graph


def build_layers(
    df: pd.DataFrame,
    n_tries: int,
    n_jobs: int,
    pair_budget_per_task: int,
) -> dict[str, nx.Graph]:
    layers: dict[str, nx.Graph] = {}
    for modality in MODALITIES:
        with logger.contextualize(modality=modality):
            logger.info("Preparing modality frame")
            mod_df = prepare_modality_df(df, modality)
            logger.info(f"Building coordination network for {modality}")
            layers[modality] = build_coordination_network(
                mod_df=mod_df,
                modality=modality,
                n_tries=n_tries,
                n_jobs=n_jobs,
                pair_budget_per_task=pair_budget_per_task,
            )
            del mod_df
            gc.collect()
    return layers


@click.command()
@click.option("--path", type=str, required=True, help="Path to the parquet input file")
@click.option(
    "--output", type=str, default=None, help="Output path for the dumped layers"
)
@click.option(
    "--n-tries",
    type=int,
    default=100,
    show_default=True,
    help="Null replicates per pair",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Parallel workers for pair scoring",
)
@click.option(
    "--pair-budget-per-task",
    type=int,
    default=20_000,
    show_default=True,
    help="Approximate number of pairs bundled into a single parallel task",
)
def main(
    path: str, output: str | None, n_tries: int, n_jobs: int, pair_budget_per_task: int
):
    logger.info(f"Loading data from {path}")
    df = load_parquet(path)

    layers = build_layers(
        df=df,
        n_tries=n_tries,
        n_jobs=n_jobs,
        pair_budget_per_task=pair_budget_per_task,
    )

    if output is None:
        output = f"{Path(path).name}_itemvirality"
    dump(layers, output)
    logger.info(f"Wrote layers to {output}")


if __name__ == "__main__":
    main()
