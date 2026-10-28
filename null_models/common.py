# The vectorized broadcast path builds an (n_tries, Lu, Lv) tensor per pair --
# O(Lu*Lv) work per try, vs. the scalar path's O((Lu+Lv) log L) sort+search
# per try. For the common case (an object used once each by many distinct
# users, Lu=Lv=1-ish) the vectorized path wins by orders of magnitude on call
# overhead alone. But once users post the same object repeatedly (e.g.
# repeat/state-media posters), Lu*Lv can grow enough that the vectorized
# path's worse complexity loses to the scalar path's better one, even though
# the tensor still fits comfortably in memory -- this is a complexity
# crossover, not a memory limit, so gate on Lu*Lv directly rather than total
# element count against a memory budget.
NULL_MEANS_VECTORIZE_PRODUCT_CAP = 400

# Objects with at least this many users get start/finish/heartbeat logging in
# _process_object, to help spot straggler "hub" objects in future runs.
INSTRUMENT_OBJECT_SIZE_THRESHOLD = 500

MODALITIES = [
    "hashtags",
    "urls",
    "account_mentions",
    "in_reply_to_postid",
    "in_reply_to_accountid",
    "reposted_accountid",
    "reposted_postid",
]

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger


def stable_object_user_seed(
    user: object, modality: str, object_key: object, salt: int = 0
) -> int:
    """Seed a user's null resampling by (modality, object, user) so it can be
    computed once per object and reused across every partner sharing that object."""
    payload = f"{user!r}|{modality!r}|{object_key!r}|{salt}".encode(
        "utf-8", errors="surrogatepass"
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def compute_pvalue(
    observed_deltas: np.ndarray,
    null_means: np.ndarray,
) -> float | None:
    if observed_deltas.size == 0:
        return None

    t_obs = float(observed_deltas.mean())
    n_extreme = int(np.sum(null_means <= t_obs))
    return (1 + n_extreme) / (len(null_means) + 1)


def iter_object_records(mod_df: pd.DataFrame, modality: str):
    """Yield per-object user groupings (object_key, users, timestamps_by_user)."""
    for object_key, object_df in mod_df.groupby(modality, sort=False):
        per_user = (
            object_df.groupby("accountid", sort=True)["post_time"]
            .agg(list)
            .sort_index()
        )
        users = per_user.index.to_list()
        if len(users) < 2:
            continue
        timestamps_by_user = dict(zip(users, per_user.to_list(), strict=False))
        yield object_key, users, timestamps_by_user


def load_parquet(path: str) -> pd.DataFrame:
    """Read a parquet file or a directory of parquet parts.

    pd.read_parquet on a directory infers one schema from the first part, so it fails
    when that part has an all-null column typed `null` while later parts have the real
    type (e.g. reposted_accountid in 14188975). In that case fall back to reading the
    parts one by one, so each keeps its own schema, and concatenating them.
    """
    try:
        return pd.read_parquet(path)
    except (pa.ArrowNotImplementedError, pa.ArrowInvalid) as exc:
        part_files = sorted(
            part
            for part in Path(path).glob("*.parquet")
            if not part.name.startswith((".", "_"))
        )
        if not part_files:
            raise
        logger.warning(
            f"pd.read_parquet({path}) failed ({exc}); "
            f"reading {len(part_files)} part files one by one"
        )
        return pd.concat(
            (pd.read_parquet(part) for part in part_files), ignore_index=True
        )


def matched_delta_t_list(
    timestamps_u: list[datetime] | np.ndarray,
    timestamps_v: list[datetime] | np.ndarray,
) -> np.ndarray:
    """Compute all nearest-subsequent gaps in minutes for U->V and V->U."""
    times_u = sorted_datetime64_array(timestamps_u)
    times_v = sorted_datetime64_array(timestamps_v)

    if times_u.size == 0 or times_v.size == 0:
        return np.empty(0, dtype=float)

    delta_uv = _subsequent_deltas(times_u, times_v, strict=True)
    delta_vu = _subsequent_deltas(times_v, times_u, strict=False)
    if delta_uv.size == 0:
        return delta_vu
    if delta_vu.size == 0:
        return delta_uv
    return np.concatenate([delta_uv, delta_vu])


def sorted_datetime64_array(timestamps: list[datetime] | np.ndarray) -> np.ndarray:
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


def null_means_from_replicates(
    reps_u: np.ndarray,
    reps_v: np.ndarray,
) -> np.ndarray:
    n_tries, length_u = reps_u.shape
    length_v = reps_v.shape[1]
    if length_u == 0 or length_v == 0:
        return np.full(n_tries, np.inf, dtype=float)
    if length_u * length_v <= NULL_MEANS_VECTORIZE_PRODUCT_CAP:
        return _null_means_from_replicates_vectorized(reps_u, reps_v)
    return _null_means_from_replicates_scalar(reps_u, reps_v)


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


def prepare_modality_df(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if modality in ("hashtags", "urls", "account_mentions"):
        mod_df = df[["accountid", "post_time", modality]].explode(modality)
        mod_df[modality] = mod_df[modality].astype("string").str.lower().str.strip()
        mod_df = mod_df.dropna(subset=[modality]).copy()
    elif modality in (
        "in_reply_to_postid",
        "in_reply_to_accountid",
        "reposted_accountid",
        "reposted_postid",
    ):
        mod_df = (
            df[["accountid", "post_time", modality]].dropna(subset=[modality]).copy()
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return mod_df
