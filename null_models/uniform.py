"""Uniform-time null

Same pipeline (observed nearest-subsequent delta-t per user pair and object,
empirical p-value against null replicates, Fisher combination across objects).

Each null timestamp is drawn uniformly from a time grid.

The grid has one point per action: with N actions between tmin and tmax it is
linspace(tmin, tmax, N), i.e. step (tmax - tmin) / (N - 1).
E.g. tmin=03:00, tmax=06:00, N=7 gives 03:00, 03:30, ..., 06:00,
each drawn with probability 1/7.

For an object, a user with L observed timestamps still gets L null timestamps per
replicate, drawn i.i.d. (with replacement) from their grid.

--null sets where tmin, tmax and N come from is:
  global    over every action in every modality, by every user (one grid);
  modality  over every action in the modality being scored, by every user
            (one grid per modality);
  user      over every action by that user, in every modality (one grid per user).
"""

from __future__ import annotations

import gc
import os
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

NULL_MODES = ("global", "modality", "user")

# A grid is (tmin in epoch-ns, step in ns, number of points).
Grid = tuple[int, float, int]


def _grid_from_stats(tmin_ns: int, tmax_ns: int, n_actions: int) -> Grid:
    step = (tmax_ns - tmin_ns) / (n_actions - 1) if n_actions > 1 else 0.0
    return (int(tmin_ns), float(step), int(n_actions))


def collect_time_stats(
    df: pd.DataFrame,
) -> tuple[dict[str, tuple[int, int, int]], pd.DataFrame]:
    """One pass over the modality frames: (tmin, tmax, count) per modality and per user.

    Returns {modality: (tmin_ns, tmax_ns, n)} and a per-user frame (index accountid,
    columns tmin, tmax, n) aggregated over every modality.
    """
    modality_stats: dict[str, tuple[int, int, int]] = {}
    per_user_parts: list[pd.DataFrame] = []
    for modality in MODALITIES:
        mod_df = prepare_modality_df(df, modality)
        if mod_df.empty:
            logger.warning(f"{modality}: no actions, no grid")
            continue
        post_ns = pd.Series(
            np.asarray(mod_df["post_time"], dtype="datetime64[ns]").astype("int64")
        )
        modality_stats[modality] = (
            int(post_ns.min()),
            int(post_ns.max()),
            len(post_ns),
        )
        per_user_parts.append(
            post_ns.groupby(mod_df["accountid"].to_numpy()).agg(
                tmin="min", tmax="max", n="size"
            )
        )
        del mod_df, post_ns
        gc.collect()

    per_user = (
        pd.concat(per_user_parts)
        .groupby(level=0)
        .agg(tmin=("tmin", "min"), tmax=("tmax", "max"), n=("n", "sum"))
    )
    return modality_stats, per_user


def resample_uniform_grid_vectorized(
    grid: Grid,
    length: int,
    n_tries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_tries null copies of a user's `length` timestamps at once (shape (n_tries, length)).

    Uniform null: each null timestamp is a uniformly random point of the grid
    tmin + i * step, i = 0..n-1.
    """
    tmin_ns, step_ns, n_points = grid
    if length == 0:
        return np.empty((n_tries, 0), dtype="datetime64[ns]")
    idx = rng.integers(0, n_points, size=(n_tries, length))
    offsets = np.rint(idx * step_ns).astype("int64")
    return (tmin_ns + offsets).astype("datetime64[ns]")


def precompute_object_replicates(
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    grids_by_user: dict[object, Grid],
    n_tries: int,
    modality: str,
    object_key: object,
) -> dict[object, np.ndarray]:
    """Resample each user's timestamps once per object, reused across all its pairs.

    Each user gets as many null timestamps as they have observed ones for the object,
    drawn from that user's grid.
    """
    replicates: dict[object, np.ndarray] = {}
    for user in users:
        length = len(timestamps_by_user[user])
        seed = stable_object_user_seed(user, modality, object_key)
        rng = np.random.default_rng(seed)
        replicates[user] = resample_uniform_grid_vectorized(
            grids_by_user[user], length, n_tries, rng
        )
    return replicates


def _process_object(
    object_key: object,
    users: list[object],
    timestamps_by_user: dict[object, list[datetime]],
    grids_by_user: dict[object, Grid],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    replicates = precompute_object_replicates(
        users, timestamps_by_user, grids_by_user, n_tries, modality, object_key
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


ObjectRecord = tuple[
    object, list[object], dict[object, list[datetime]], dict[object, Grid]
]


def _process_object_batch(
    batch: list[ObjectRecord],
    n_tries: int,
    modality: str,
) -> list[tuple[object, object, float | None]]:
    results: list[tuple[object, object, float | None]] = []
    for object_key, users, timestamps_by_user, grids_by_user in batch:
        results.extend(
            _process_object(
                object_key, users, timestamps_by_user, grids_by_user, n_tries, modality
            )
        )
    return results


def build_coordination_network(
    mod_df: pd.DataFrame,
    modality: str,
    grid_for_user,
    n_tries: int = 100,
    n_jobs: int = -1,
    pair_budget_per_task: int = 20_000,
) -> nx.Graph:
    """Build a coordination layer for one modality using per-object batched dispatch.

    Same dispatch as pvalue_coord_fast_itemnull.build_coordination_network; each
    object additionally carries the grid of each of its users (grid_for_user(user)).
    """
    edge_pvalues: dict[tuple[object, object], list[float]] = defaultdict(list)

    effective_n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    group_capacity = max(effective_n_jobs * 4, 1)

    sub_batch: list[ObjectRecord] = []
    sub_batch_pairs = 0
    dispatch_group: list[list[ObjectRecord]] = []

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
            grids_by_user = {user: grid_for_user(user) for user in users}
            sub_batch.append((object_key, users, timestamps_by_user, grids_by_user))
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
    null_mode: str,
    n_tries: int,
    n_jobs: int,
    pair_budget_per_task: int,
) -> dict[str, nx.Graph]:
    logger.info("Collecting tmin/tmax/count per modality and per user")
    modality_stats, per_user = collect_time_stats(df)

    modality_grids = {
        m: _grid_from_stats(*stats) for m, stats in modality_stats.items()
    }
    global_grid = _grid_from_stats(
        min(s[0] for s in modality_stats.values()),
        max(s[1] for s in modality_stats.values()),
        sum(s[2] for s in modality_stats.values()),
    )
    user_grids = {
        user: _grid_from_stats(tmin, tmax, n)
        for user, tmin, tmax, n in zip(
            per_user.index,
            per_user["tmin"],
            per_user["tmax"],
            per_user["n"],
            strict=True,
        )
    }
    del per_user
    step_minutes = global_grid[1] / 6e10
    logger.info(
        f"Global grid: {global_grid[2]} actions, step {step_minutes:.3f} min; "
        f"{len(user_grids)} users"
    )
    for modality, grid in modality_grids.items():
        logger.info(
            f"{modality} grid: {grid[2]} actions, step {grid[1] / 6e10:.3f} min"
        )

    layers: dict[str, nx.Graph] = {}
    for modality in MODALITIES:
        if null_mode == "global":
            grid_for_user = lambda user: global_grid
        elif null_mode == "modality":
            modality_grid = modality_grids.get(modality)
            grid_for_user = lambda user: modality_grid
        else:
            grid_for_user = user_grids.__getitem__

        with logger.contextualize(modality=modality):
            logger.info("Preparing modality frame")
            mod_df = prepare_modality_df(df, modality)
            logger.info(
                f"Building coordination network for {modality} ({null_mode} uniform null)"
            )
            layers[modality] = build_coordination_network(
                mod_df=mod_df,
                modality=modality,
                grid_for_user=grid_for_user,
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
    "--null",
    "null_mode",
    type=click.Choice(NULL_MODES),
    required=True,
    help="Where tmin, tmax and the action count of the uniform grid come from",
)
@click.option(
    "--output",
    type=str,
    default=None,
    help="Output path for the dumped layers "
    "[default: <output-root>/pvalue_coord_uniform_<null>/<input name>]",
)
@click.option("--output-root", type=str, required=True)
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
    path: str,
    null_mode: str,
    output: str | None,
    output_root: str,
    n_tries: int,
    n_jobs: int,
    pair_budget_per_task: int,
):
    if output is None:
        out_dir = Path(output_root) / f"pvalue_coord_uniform_{null_mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"{Path(path).name}")

    logger.info(f"Loading data from {path}")
    df = load_parquet(path)

    layers = build_layers(
        df=df,
        modalities=MODALITIES,
        null_mode=null_mode,
        n_tries=n_tries,
        n_jobs=n_jobs,
        pair_budget_per_task=pair_budget_per_task,
    )

    # Write to a temp name and rename, so the sbatch skip check never sees a
    # half-written file after a crash or timeout.
    tmp_output = f"{output}.tmp"
    dump(layers, tmp_output)
    os.replace(tmp_output, output)
    logger.info(f"Wrote layers to {output}")


if __name__ == "__main__":
    main()
