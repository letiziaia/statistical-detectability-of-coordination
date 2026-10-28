from __future__ import annotations

import gc
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
    stable_object_user_seed,
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


def empirical_posting_rates(data: pd.DataFrame) -> dict[object, np.ndarray]:
    """Compute 168-bin posting profiles for each user."""
    slot = data["post_time"].dt.dayofweek * 24 + data["post_time"].dt.hour
    counts = (
        data.assign(_slot=slot)
        .pivot_table(
            index="accountid",
            columns="_slot",
            values="post_time",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=range(168), fill_value=0)
    )

    rates: dict[object, np.ndarray] = {}
    totals = counts.sum(axis=1).to_numpy(dtype=float)
    values = counts.to_numpy(dtype=float)
    for user, row, total in zip(counts.index, values, totals, strict=False):
        if total > 0:
            rates[user] = row / total
    return rates


def _daily_profile_slices(profile: np.ndarray) -> list[np.ndarray]:
    slices: list[np.ndarray] = []
    profile = np.asarray(profile, dtype=float)
    for day in range(7):
        day_slice = profile[day * 24 : (day + 1) * 24].astype(float, copy=False)
        total = day_slice.sum()
        if total > 0:
            day_slice = day_slice / total
        else:
            day_slice = np.full(24, 1.0 / 24.0, dtype=float)
        slices.append(day_slice)
    return slices


def resample_timestamps_vectorized(
    timestamps: np.ndarray,
    daily_profiles: list[np.ndarray],
    n_tries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_tries resampled copies of `timestamps` at once (shape (n_tries, L)).

    Only hour/minute/second are redrawn (per the stratified-shuffling null model);
    the calendar date, and hence day-of-week, of each timestamp is preserved.
    """
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    length = timestamps.size
    if length == 0:
        return np.empty((n_tries, 0), dtype="datetime64[ns]")

    dow = pd.DatetimeIndex(timestamps).dayofweek.to_numpy()
    hours = np.empty((n_tries, length), dtype=np.int64)
    for day in range(7):
        mask = dow == day
        count = int(mask.sum())
        if count == 0:
            continue
        hours[:, mask] = rng.choice(24, size=(n_tries, count), p=daily_profiles[day])

    minutes = rng.integers(0, 60, size=(n_tries, length))
    seconds = rng.integers(0, 60, size=(n_tries, length))

    day_floor = timestamps.astype("datetime64[D]").astype("datetime64[ns]")
    seconds_of_day = (hours * 3600 + minutes * 60 + seconds).astype("timedelta64[s]")
    return day_floor[np.newaxis, :] + seconds_of_day


def precompute_object_replicates(
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    daily_profiles_by_user: dict[object, list[np.ndarray]],
    n_tries: int,
    modality: str,
    object_key: object,
) -> dict[object, np.ndarray]:
    """Resample each user's timestamps once per object, reused across all its pairs."""
    replicates: dict[object, np.ndarray] = {}
    for user in users:
        timestamps = np.asarray(timestamps_by_user[user], dtype="datetime64[ns]")
        seed = stable_object_user_seed(user, modality, object_key)
        rng = np.random.default_rng(seed)
        replicates[user] = resample_timestamps_vectorized(
            timestamps, daily_profiles_by_user[user], n_tries, rng
        )
    return replicates


def _process_object(
    object_key: object,
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    daily_profiles_by_user: dict[object, list[np.ndarray]],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    replicates = precompute_object_replicates(
        users, timestamps_by_user, daily_profiles_by_user, n_tries, modality, object_key
    )

    k = len(users)
    pair_count = k * (k - 1) // 2
    instrument = k >= INSTRUMENT_OBJECT_SIZE_THRESHOLD
    if instrument:
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


def _users_in_batch(
    batch: list[tuple[object, list[object], dict[object, list[datetime]]]],
) -> set[object]:
    users: set[object] = set()
    for _, batch_users, _ in batch:
        users.update(batch_users)
    return users


def _process_object_batch(
    batch: list[tuple[object, list[object], dict[object, list[datetime]]]],
    daily_profiles_by_user: dict[object, list[np.ndarray]],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    results: list[tuple[object, object, float | None]] = []
    for object_key, users, timestamps_by_user in batch:
        results.extend(
            _process_object(
                object_key,
                users,
                timestamps_by_user,
                daily_profiles_by_user,
                n_tries,
                modality,
            )
        )
    return results


def build_coordination_network(
    mod_df: pd.DataFrame,
    modality: str,
    empirical_rates: dict[object, np.ndarray],
    n_tries: int = 100,
    n_jobs: int = -1,
    pair_budget_per_task: int = 20_000,
) -> nx.Graph:
    """Build a coordination layer for one modality using per-object batched dispatch.

    Work is dispatched at the object-batch level (not per pair): each object's users
    are resampled once and reused across all of that object's C(k,2) pairs, and each
    joblib task handles a whole batch of objects bounded by `pair_budget_per_task`
    pairs, instead of one process-pool round trip per pair.
    """
    daily_profiles_by_user = {
        user: _daily_profile_slices(rate) for user, rate in empirical_rates.items()
    }
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
        users_needed: set[object] = set()
        for group_batch in dispatch_group:
            users_needed.update(_users_in_batch(group_batch))
        profiles_subset = {user: daily_profiles_by_user[user] for user in users_needed}

        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_process_object_batch)(
                group_batch, profiles_subset, n_tries, modality
            )
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
            logger.info("Computing empirical posting rates")
            empirical_rates = empirical_posting_rates(mod_df)
            logger.info(f"Building coordination network for {modality}")
            layers[modality] = build_coordination_network(
                mod_df=mod_df,
                modality=modality,
                empirical_rates=empirical_rates,
                n_tries=n_tries,
                n_jobs=n_jobs,
                pair_budget_per_task=pair_budget_per_task,
            )
            del mod_df
            del empirical_rates
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
        output = f"{Path(path).name}"
    dump(layers, output)
    logger.info(f"Wrote layers to {output}")


if __name__ == "__main__":
    main()
