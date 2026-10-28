"""Place each network layer in the detectability diagrams.

  n and k come from the ground truth
  x_ij   = -log p_ij                  (absent pair -> p = 1 -> x = 0)
  x is winsorized: x is capped at c_max + 1
  (--c-max, default 20), so one extreme pair cannot dominate
  m_in, s_in = mean and SD of x over the in-group pairs
  m0, s0     = mean and SD of x over the control-control pairs
  beta   = log k / log n

The signal, mu_hat, is computed from m_in, s_in and the pair counts.

USAGE
-----
  python detectability.py                  # full run

Flags worth knowing:
  --null-from {control,mixed}     which pairs give m0 and s0
  --absent-pairs {p1,untested}    see the note on --absent-pairs
=============================================================================
"""

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
from loguru import logger
from scipy.stats import combine_pvalues

# The edge attribute every individual layer is placed from. build_coordination_
# network writes this on every edge of every layer, in every country's joblib
# dump, as that layer's own Fisher-combined p-value -- there is nothing to
# configure here. The GLOBAL layer is different: it pools the raw `p_values`
# list instead (see build_global_pvalue_graph's docstring for why).
PVALUE_ATTR = "p_modality"

# Column order for the output CSV (layer_coordinates.csv) and for the rows
# built by compute_layer_coords().
CSV_COLS: list[str] = [
    "code",
    "layer",
    "n",
    "k",
    "k_ctrl",
    "beta",
    "m0",
    "s0",
    "m_in",
    "s_in",
    "q_obs",
    "n_in_pairs_obs",
    "n_in_pairs_total",
    "n_ctrl_pairs_obs",
    "n_ctrl_pairs_total",
    "c_max",
    "n_capped",
    "frac_capped",
    "mu_ach",
    "mu_ach_rate",
    "rho",
    "mu_it_floor",
    "mu_spec",
    "mu_sum",
    "mu_comp",
    "mu_stat",
    "alpha_ach_exact",
    "alpha_comp_exact",
    "flag_k_too_large",
    "flag_few_pairs",
    "flag_null_weak",
]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument strings to parse. None (the default) makes argparse
            read from sys.argv, which is what every real invocation wants;
            tests pass an explicit list instead.

    Returns:
        The parsed arguments as an argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Place coordination layers in the planted-submatrix detectability diagrams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--layers-dir", required=True)
    parser.add_argument("--truth-root", required=True)
    parser.add_argument("--layers-glob", default="**")
    parser.add_argument("--out-dir", default="./layer_analysis_out")
    parser.add_argument(
        "--null-from",
        default="control",
        choices=["control", "mixed", "both"],
        help="Which pairs estimate H0. 'control' = control-control (default, "
        "cleanest). 'mixed' = operation-account x control-account pairs. "
        "'both' reports each.",
    )
    parser.add_argument(
        "--absent-pairs",
        default="p1",
        choices=["p1", "untested"],
        help="'p1': a pair with no edge was tested and got p=1. "
        "'untested': the pair was never tested, so it is excluded from the "
        "means m_in and m0.",
    )
    parser.add_argument(
        "--c-max",
        type=float,
        default=20.0,
        help="Winsorization cap on the score c = -log p - 1, so x = -log p is "
        "capped at c_max + 1. Same meaning and same default as compute_c_scores "
        "Set 0 or below to disable.",
    )
    parser.add_argument(
        "--min-in-pairs",
        type=int,
        default=45,
        help="Minimum C(k,2) for a layer to count as usable (default 45, i.e. k>=10).",
    )
    parser.add_argument(
        "--min-ctrl-pairs",
        type=int,
        default=200,
        help="Minimum control pairs for the null estimate to count as usable.",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=2.0 / 3.0,
        help="Flag layers with beta above this.",
    )
    return parser.parse_args(argv)


def _log_binom(n: float, k: float) -> float:
    """Return log(C(n, k)), computed via log-gamma for numerical stability."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _lambda_of_w(w: float, k: float, m: float) -> float:
    """Lambda(w). Closed-form optimum t = wk / (m + wk). Used by w_star_cgf's bisection search for w."""
    t = w * k / (m + w * k)
    return -k * t * w + m * (-t - math.log1p(-t))


def w_star_cgf(n: float, k: float, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Finite-n scan threshold w*_CGF.

    The equation is  min_{0<theta<k} [ -theta w + C(k,2) psi(theta/k) ]
                     = -log C(n,k),   psi(t) = -t - log(1-t).

    The inner minimisation has a closed form. With m = C(k,2) and t = theta/k,
    d/dtheta gives (m/k) t/(1-t) = w, hence t = wk / (m + wk), so only a
    one-dimensional root find in w is left. Lambda(w) starts at 0 and decreases
    monotonically, so bisection is safe once the bracket is found by doubling.

    Returns:
        w*_CGF, or NaN where the equation has no meaning (k < 2 or k >= n).
    """
    if k < 2 or k >= n:
        return float("nan")
    m = k * (k - 1) / 2.0
    target = -_log_binom(n, k)

    hi = 1.0
    for _ in range(max_iter):
        if _lambda_of_w(hi, k, m) <= target:
            break
        hi *= 2.0
    else:
        return float("nan")
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _lambda_of_w(mid, k, m) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def boundaries_mu(
    n: float | np.ndarray, k: float | np.ndarray
) -> dict[str, np.ndarray]:
    """Compute every detectability threshold, in mu units, for population n
    and group size k.

    n and k may be scalars or arrays (elementwise). Entries with k < 2 or
    k >= n give NaN, because log(n/k) and 1/(k-1) are undefined there and the
    theory needs k << n.

    Returns:
        A dict of threshold name -> array of mu values, with keys:
            ach        mu_scan, the valid finite-(n,k) scan threshold
            ach_rate   leading-order rate for mu_scan
            rho        ach / ach_rate, how far the rate underestimates scan
            it_tail    the IT tail branch
            it_floor   the IT floor, min(it_tail, sum)
            spec       the spectral threshold
            sum        the sum/linear threshold
            comp       the computational boundary, min(spec, sum)
            stat       the statistical boundary, min(scan, sum)
    """
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)
    bad = (k < 2) | (k >= n)
    kk = np.where(bad, np.nan, k)
    lr = np.log(n / kk)

    ach_rate = 2.0 * np.sqrt(lr / (kk - 1.0))  # eq. mu-scan-threshold
    it_tail = np.sqrt(2.0 * lr / (kk - 1.0))  # eq. it-tail

    # The scan threshold. w-cgf has no closed form, so it is solved
    # per (n, k). w_star_cgf is scalar.
    n_broadcast = np.broadcast_to(n, kk.shape)
    w_cgf = np.array(
        [
            w_star_cgf(float(nv), float(kv)) if np.isfinite(kv) else np.nan
            for nv, kv in zip(np.ravel(n_broadcast), np.ravel(kk))
        ]
    ).reshape(kk.shape)
    ach = 2.0 * w_cgf / (kk - 1.0)
    # rho: how far the leading-order rate falls short of the valid one
    rho = np.asarray(ach) / ach_rate
    spec = np.sqrt(n) / (kk - 1.0)  # eq. spectral-threshold
    summ = np.sqrt(2.0 * n * (n - 1.0)) / (kk * (kk - 1.0))  # eq. mulin

    # lower-bound: the boundary is the SMALLER of the two branches. It
    # coincides with the sum test up to the absolute constant c, so we
    # take the sum test itself (c = sqrt 2).
    it_floor = np.minimum(it_tail, summ)
    comp = np.minimum(spec, summ)
    stat = np.minimum(ach, summ)

    return dict(
        ach=ach,
        ach_rate=ach_rate,
        rho=rho,
        it_tail=it_tail,
        it_floor=it_floor,
        spec=spec,
        sum=summ,
        comp=comp,
        stat=stat,
    )


def alpha_of_mu(mu: np.ndarray, n: float | np.ndarray) -> np.ndarray:
    """Compute Brennan alpha for a signal mu at population n, using
    d_SKL = mu^2.

    Returns:
        alpha for each entry of mu, or NaN where mu <= 0 (alpha is undefined
        there since d_SKL = mu^2 would need a signless log).
    """
    mu = np.asarray(mu, dtype=float)
    out = np.full(mu.shape, np.nan)
    ok = mu > 0
    out[ok] = -np.log(mu[ok] ** 2) / np.log(n if np.isscalar(n) else np.asarray(n)[ok])
    return out


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------
ACCOUNT_ID_COLUMN_NAME = "accountid"


def normalize_account_id(raw_id: object) -> str | None:
    """Normalize one account id to a stripped string.

    Account ids appear as int64 in the ground-truth parquet and as str in the
    graphs (or the reverse). A silent type mismatch between the two would
    make every intersection empty and k = 0, so both sides must be forced
    through this same function before being compared or hashed together.

    Returns:
        The id as a stripped string, or None if raw_id is None.
    """
    if raw_id is None:
        return None
    if isinstance(raw_id, float) and float(raw_id).is_integer():
        raw_id = int(raw_id)
    return str(raw_id).strip()


class GroundTruthMembership(NamedTuple):
    """Which accounts are in the planted coordinated operation, and which
    are controls, in one country.

    control_account_ids is everyone else, so callers wanting every known
    account id can take operation_account_ids | control_account_ids.
    """

    operation_account_ids: set[str]
    control_account_ids: set[str]


def load_ground_truth_membership(
    truth_root: str, country_code: str
) -> GroundTruthMembership:
    """Load the ground-truth parquet for one country and split its accounts
    into the planted coordinated operation and everyone else (the controls).

    An account is IN the operation if any of its rows has is_control == False.
    An account is a CONTROL if it is never in the operation.

    Args:
        truth_root: Directory containing one subdirectory per country code,
            each holding that country's ground-truth parquet file(s).
        country_code: The country's numeric code, matching a subdirectory
            name under truth_root (e.g. "14141550").

    Returns:
        A GroundTruthMembership with every account id as a stripped string
        (see normalize_account_id).

    Raises:
        FileNotFoundError: truth_root has no subdirectory for country_code.
        KeyError: the parquet has no 'is_control' column, or no
            ACCOUNT_ID_COLUMN_NAME column.
    """
    truth_dir = os.path.join(truth_root, str(country_code))
    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f"No ground-truth directory: {truth_dir}")

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
            f"'is_control' absent in {truth_dir}. Columns: {list(truth_df.columns)}"
        )
    if ACCOUNT_ID_COLUMN_NAME not in truth_df.columns:
        raise KeyError(
            f"'{ACCOUNT_ID_COLUMN_NAME}' absent in {truth_dir}. "
            f"Columns: {list(truth_df.columns)}"
        )

    is_control_column = truth_df["is_control"]
    if is_control_column.dtype == bool:
        is_control = is_control_column
    else:
        is_control = is_control_column.map(
            lambda v: str(v).strip().lower() in ("true", "1", "1.0", "t", "yes")
        )

    account_ids = truth_df[ACCOUNT_ID_COLUMN_NAME].map(normalize_account_id)
    is_in_operation_by_account = (~is_control).groupby(account_ids).any()
    operation_account_ids = set(
        is_in_operation_by_account[is_in_operation_by_account].index.tolist()
    )
    all_account_ids = set(account_ids.unique().tolist())
    control_account_ids = all_account_ids - operation_account_ids
    return GroundTruthMembership(
        operation_account_ids=operation_account_ids,
        control_account_ids=control_account_ids,
    )


# ----------------------------------------------------------------------------
# One pass over the edges, sums over ALL pairs
# ----------------------------------------------------------------------------
@dataclass
class LayerPairSums:
    """Sums of x = -log p, accumulated per pair class, over one layer's edges.

    Each pair class tracks a first-moment sum (sum_x_*), a second-moment sum
    (sum_x_squared_*, for computing variance later), and how many of that
    class's pairs actually had an edge (n_*_pairs_observed). A pair with no
    edge is not counted here at all: it contributes x = 0 implicitly, by
    being part of the full pair count used later in _mean_and_std(), not by
    adding a zero term to these sums.
    """

    sum_x_in_group: float = 0.0
    sum_x_squared_in_group: float = 0.0
    n_in_group_pairs_observed: int = 0

    sum_x_control: float = 0.0
    sum_x_squared_control: float = 0.0
    n_control_pairs_observed: int = 0

    sum_x_mixed: float = 0.0
    sum_x_squared_mixed: float = 0.0
    n_mixed_pairs_observed: int = 0

    n_negative_values_clamped: int = 0
    n_values_observed: int = 0
    n_values_capped: int = 0


def accumulate_layer(
    graph: nx.Graph,
    operation_account_ids: set[str],
    control_account_ids: set[str],
    attr_name: str,
    c_max: float = 20.0,
) -> LayerPairSums:
    """Make one pass over edges, summing x = -log p by pair class
    (in-group/in-group, control/control, or mixed; "in-group" here means a
    pair where BOTH accounts are in the planted operation).

    Absent pairs have p = 1 and x = 0 exactly, so they add nothing to either
    sum.

    Args:
        graph: One layer's graph.
        operation_account_ids: Normalized ids (see normalize_account_id) of
            accounts in the planted coordinated operation.
        control_account_ids: Normalized ids of control accounts.
        attr_name: Which edge attribute to read (e.g. "p_modality"). Its
            values are p-values.
        c_max: Winsorization cap on the score c.
            0 or below disables capping.

    Returns:
        The accumulated sums.
    """
    sums = LayerPairSums()
    # c = x - 1, so capping the score c at c_max caps x at c_max + 1.
    x_cap = (c_max + 1.0) if c_max and c_max > 0 else float("inf")
    for node_a, node_b, edge_attrs in graph.edges(data=True):
        if node_a == node_b:
            continue  # the diagonal is excluded from the model
        if attr_name not in edge_attrs:
            continue
        raw_value = edge_attrs[attr_name]
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            continue
        raw_value = float(raw_value)
        if not math.isfinite(raw_value):
            continue

        p_value = min(max(raw_value, 1e-300), 1.0)
        x = -math.log(p_value)
        if x < 0:
            sums.n_negative_values_clamped += 1
            x = 0.0  # p > 1 is not a p-value; clamp and count
        sums.n_values_observed += 1
        if x > x_cap:
            sums.n_values_capped += 1
            x = x_cap

        id_a = normalize_account_id(node_a)
        id_b = normalize_account_id(node_b)
        a_in_operation = id_a in operation_account_ids
        b_in_operation = id_b in operation_account_ids
        a_is_control, b_is_control = (
            id_a in control_account_ids,
            id_b in control_account_ids,
        )
        if a_in_operation and b_in_operation:
            sums.sum_x_in_group += x
            sums.sum_x_squared_in_group += x * x
            sums.n_in_group_pairs_observed += 1
        elif a_is_control and b_is_control:
            sums.sum_x_control += x
            sums.sum_x_squared_control += x * x
            sums.n_control_pairs_observed += 1
        elif (a_in_operation and b_is_control) or (a_is_control and b_in_operation):
            sums.sum_x_mixed += x
            sums.sum_x_squared_mixed += x * x
            sums.n_mixed_pairs_observed += 1
    return sums


def _mean_and_std(
    sum_x: float, sum_x_squared: float, n_total_pairs: float
) -> tuple[float, float]:
    """Mean and SD of x over n_total_pairs pairs, given sums over the
    non-zero (i.e. actual edges).

    Returns:
        (mean, standard deviation), or (NaN, NaN) if n_total_pairs <= 1.
    """
    if n_total_pairs <= 1:
        return float("nan"), float("nan")
    mean = sum_x / n_total_pairs
    variance = sum_x_squared / n_total_pairs - mean * mean
    return mean, math.sqrt(max(variance, 0.0))


@dataclass
class LayerCoordinates:
    """Position of one layer in the detectability diagrams."""

    n: int
    k: int
    k_ctrl: int
    beta: float
    m0: float
    s0: float
    m_in: float
    s_in: float
    q_obs: float
    n_in_pairs_obs: int
    n_in_pairs_total: int
    n_ctrl_pairs_obs: int
    n_ctrl_pairs_total: int
    c_max: float
    n_capped: int
    frac_capped: float
    mu_ach: float
    mu_ach_rate: float
    rho: float
    mu_it_floor: float
    mu_spec: float
    mu_sum: float
    mu_comp: float
    mu_stat: float
    alpha_ach_exact: float
    alpha_comp_exact: float
    flag_k_too_large: bool
    flag_few_pairs: bool
    flag_null_weak: bool


def compute_layer_coords(
    pair_sums: LayerPairSums,
    population_size: int,
    operation_size: int,
    control_group_size: int,
    args: argparse.Namespace,
) -> LayerCoordinates | None:
    """Means and SDs of x over the in-group and control pairs of one layer,
    pair counts, and the boundaries at the population (n, k).

    Args:
        pair_sums: This layer's accumulated sums, from accumulate_layer().
        population_size: n, the number of accounts in the ground truth.
        operation_size: k, the number of planted operation accounts.
        control_group_size: k_ctrl, the number of control accounts.
        args: Parsed CLI args; reads absent_pairs, null_from, min_ctrl_pairs,
            beta_max, min_in_pairs, c_max.

    Returns:
        The layer's coordinates and flags, or None only if operation_size < 2
        or population_size < operation_size. A layer whose control SD is zero
        or not finite is kept (flag_null_weak = True).
    """
    n, k, k_ctrl = population_size, operation_size, control_group_size
    if k < 2 or n < k:
        return None

    n_in_group_pairs_total = k * (k - 1) // 2
    n_control_pairs_total = k_ctrl * (k_ctrl - 1) // 2
    n_mixed_pairs_total = k * k_ctrl

    if args.absent_pairs == "untested":
        # Pairs without an edge are treated as never tested: condition on the
        # observed ones. The partial-block box then requires the achievability
        # boundary to be divided by the fill fraction q.
        n_in_group_pairs_total = max(pair_sums.n_in_group_pairs_observed, 1)
        n_control_pairs_total = max(pair_sums.n_control_pairs_observed, 1)
        n_mixed_pairs_total = max(pair_sums.n_mixed_pairs_observed, 1)

    m_in, s_in = _mean_and_std(
        pair_sums.sum_x_in_group,
        pair_sums.sum_x_squared_in_group,
        n_in_group_pairs_total,
    )
    m0_control, s0_control = _mean_and_std(
        pair_sums.sum_x_control, pair_sums.sum_x_squared_control, n_control_pairs_total
    )
    m0_mixed, s0_mixed = _mean_and_std(
        pair_sums.sum_x_mixed, pair_sums.sum_x_squared_mixed, n_mixed_pairs_total
    )

    if args.null_from == "mixed":
        m0, s0, n_null_pairs = m0_mixed, s0_mixed, n_mixed_pairs_total
    else:
        m0, s0, n_null_pairs = m0_control, s0_control, n_control_pairs_total

    null_weak = (not np.isfinite(s0)) or s0 <= 0 or n_null_pairs < args.min_ctrl_pairs

    q_obs = (
        pair_sums.n_in_group_pairs_observed / (k * (k - 1) / 2)
        if k >= 2
        else float("nan")
    )

    boundaries = boundaries_mu(n, k)
    # mu_ach/mu_ach_rate are the reference level for the scan test
    mu_ach = float(boundaries["ach"])
    mu_ach_rate = float(boundaries["ach_rate"])
    rho = float(boundaries["rho"])
    mu_it_floor = float(boundaries["it_floor"])
    mu_spec = float(boundaries["spec"])
    mu_sum = float(boundaries["sum"])
    mu_comp = float(boundaries["comp"])
    mu_stat = float(boundaries["stat"])

    beta = math.log(k) / math.log(n)

    flag_k_too_large = beta > args.beta_max
    flag_few_pairs = n_in_group_pairs_total < args.min_in_pairs

    return LayerCoordinates(
        n=int(n),
        k=int(k),
        k_ctrl=int(k_ctrl),
        beta=beta,
        m0=m0,
        s0=s0,
        m_in=m_in,
        s_in=s_in,
        q_obs=q_obs,
        n_in_pairs_obs=int(pair_sums.n_in_group_pairs_observed),
        n_in_pairs_total=int(k * (k - 1) // 2),
        n_ctrl_pairs_obs=int(pair_sums.n_control_pairs_observed),
        n_ctrl_pairs_total=int(k_ctrl * (k_ctrl - 1) // 2),
        c_max=float(args.c_max),
        n_capped=int(pair_sums.n_values_capped),
        frac_capped=pair_sums.n_values_capped / max(pair_sums.n_values_observed, 1),
        mu_ach=mu_ach,
        mu_ach_rate=mu_ach_rate,
        rho=rho,
        mu_it_floor=mu_it_floor,
        mu_spec=mu_spec,
        mu_sum=mu_sum,
        mu_comp=mu_comp,
        mu_stat=mu_stat,
        alpha_ach_exact=float(alpha_of_mu(np.array([mu_ach]), n)[0]),
        alpha_comp_exact=float(alpha_of_mu(np.array([mu_comp]), n)[0]),
        flag_k_too_large=bool(flag_k_too_large),
        flag_few_pairs=bool(flag_few_pairs),
        flag_null_weak=bool(null_weak),
    )


# ----------------------------------------------------------------------------
# The global layer
# ----------------------------------------------------------------------------
GLOBAL_LAYER = "GLOBAL"
GLOBAL_ATTR = "p_global"


def _fisher_combine_pooled_pvalues(
    n_observed_tests_per_pair: np.ndarray,
    row_start_offsets: np.ndarray,
    sorted_p_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Fisher's combined-probability test per pair, batched by how many
    p-values were observed for that pair. The GLOBAL layer pools only the
    layers a pair actually appears in. A pair absent from a layer is
    excluded from that pair's test, not counted as a p = 1 result.

    scipy.stats.combine_pvalues wants a rectangular array, but different
    pairs have different numbers of p-values. Grouping pairs by their
    (identical, within a group) observed count means every pair in a group
    has exactly the same number of values sitting contiguously in
    sorted_p_values, so they can be gathered by plain broadcasting into one
    rectangular matrix per group.

    Args:
        n_observed_tests_per_pair: How many p-values sorted_p_values holds
            for that pair.
        row_start_offsets: Index into sorted_p_values where each pair's
            values start; length n_pairs + 1. sorted_p_values must already
            be sorted by pair row id.
        sorted_p_values: Every p-value, sorted by pair row id.

    Returns:
        (p_global_per_pair, fisher_stat_per_pair): Fisher's combined p-value
        and test statistic for each pair, indexed the same way as
        n_observed_tests_per_pair. A pair with n_observed_tests_per_pair < 1
        is left uninitialized, since it belongs to no test group.
    """
    n_pairs = len(n_observed_tests_per_pair)
    p_global_per_pair = np.empty(n_pairs, dtype=float)
    fisher_stat_per_pair = np.empty(n_pairs, dtype=float)

    for group_observed_count in np.unique(n_observed_tests_per_pair):
        group_observed_count = int(group_observed_count)
        if group_observed_count < 1:
            continue
        pair_indices_in_group = np.flatnonzero(
            n_observed_tests_per_pair == group_observed_count
        )
        row_starts = row_start_offsets[pair_indices_in_group]
        observed_p_matrix = sorted_p_values[
            row_starts[:, None] + np.arange(group_observed_count)[None, :]
        ]
        fisher_stat_for_group, p_global_for_group = combine_pvalues(
            observed_p_matrix, method="fisher", axis=1
        )
        fisher_stat_per_pair[pair_indices_in_group] = fisher_stat_for_group
        p_global_per_pair[pair_indices_in_group] = p_global_for_group

    return p_global_per_pair, fisher_stat_per_pair


def _sorted_pair_key(node_a: object, node_b: object) -> tuple:
    """A pair key that hashes the same regardless of edge direction."""
    return (node_a, node_b) if str(node_a) <= str(node_b) else (node_b, node_a)


def _coerce_to_float_list(value: object) -> list[float]:
    """Normalize an edge attribute's value (scalar or list) to a list of floats."""
    if value is None:
        return []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    try:
        return [float(x) for x in value]
    except TypeError:
        return []


def build_global_pvalue_graph(
    layers_by_name: dict[str, nx.Graph],
    attr_name: str = "p_values",
    min_p_value: float = 1e-300,
) -> nx.Graph:
    """Build one graph whose edge attribute p_global is the Fisher
    combination of a pair's p-values, pooled across every layer that pair
    actually appears in. A pair absent from a layer is excluded from that
    pair's test.

    `p_values` is a list holding one p-value per shared object.

    Args:
        layers_by_name: This country's layer graphs, keyed by layer name.
        attr_name: Which edge attribute to pool (see above). Its values are
            p-values.
        min_p_value: Floor applied to every p-value before combining, to
            avoid log(0) inside Fisher's method.

    Returns:
        A new graph with one edge per pair that had at least one finite
        p-value in any layer, carrying GLOBAL_ATTR (the Fisher-combined
        p-value), "fisher_X" (the Fisher test statistic), "n_objects" (how
        many p-values were pooled), "n_layers" (how many layers saw this
        pair), and "layer_mask" (bitmask of which layers saw it). Its every
        node is the union of all input layers' nodes, even isolated ones.
    """
    layer_names = list(layers_by_name.keys())
    pair_row_index: dict[tuple, int] = {}
    pairs_by_row: list[tuple] = []
    pooled_row_ids: list[int] = []
    pooled_p_values: list[float] = []
    pooled_layer_indices: list[int] = []
    all_nodes = set()

    for layer_index, layer_name in enumerate(layer_names):
        graph = layers_by_name[layer_name]
        all_nodes.update(graph.nodes())
        for node_a, node_b, edge_attrs in graph.edges(data=True):
            if node_a == node_b:  # the model excludes the diagonal
                continue
            p_values_for_pair = _coerce_to_float_list(edge_attrs.get(attr_name))
            p_values_for_pair = [p for p in p_values_for_pair if math.isfinite(p)]
            if not p_values_for_pair:
                continue
            pair_key = _sorted_pair_key(node_a, node_b)
            row_id = pair_row_index.get(pair_key)
            if row_id is None:
                row_id = pair_row_index[pair_key] = len(pairs_by_row)
                pairs_by_row.append(pair_key)
            for p_value in p_values_for_pair:
                pooled_row_ids.append(row_id)
                pooled_p_values.append(min(max(p_value, min_p_value), 1.0))
                pooled_layer_indices.append(layer_index)

    global_graph = nx.Graph()
    global_graph.add_nodes_from(all_nodes)
    global_graph.graph["layer_names"] = layer_names
    global_graph.graph["fisher_attr"] = attr_name
    if not pair_row_index:
        return global_graph

    pooled_row_ids = np.asarray(pooled_row_ids, dtype=np.int64)
    pooled_p_values = np.asarray(pooled_p_values, dtype=float)
    pooled_layer_indices = np.asarray(pooled_layer_indices, dtype=np.int64)
    n_pairs = len(pairs_by_row)

    layers_seen_bitmask = np.zeros(n_pairs, dtype=np.int64)
    np.bitwise_or.at(
        layers_seen_bitmask, pooled_row_ids, (np.int64(1) << pooled_layer_indices)
    )
    try:
        n_layers_seen_per_pair = np.bitwise_count(layers_seen_bitmask).astype(
            np.int64
        )  # numpy >= 2.0
    except AttributeError:
        n_layers_seen_per_pair = np.array(
            [bin(int(m)).count("1") for m in layers_seen_bitmask], dtype=np.int64
        )

    sort_order_by_row = np.argsort(pooled_row_ids, kind="stable")
    sorted_row_ids = pooled_row_ids[sort_order_by_row]
    sorted_p_values = pooled_p_values[sort_order_by_row]
    row_start_offsets = np.searchsorted(sorted_row_ids, np.arange(n_pairs + 1))
    n_observed_tests_per_pair = np.diff(row_start_offsets)

    p_global_per_pair, fisher_stat_per_pair = _fisher_combine_pooled_pvalues(
        n_observed_tests_per_pair,
        row_start_offsets,
        sorted_p_values,
    )

    for row_id, (node_a, node_b) in enumerate(pairs_by_row):
        global_graph.add_edge(
            node_a,
            node_b,
            **{
                GLOBAL_ATTR: float(p_global_per_pair[row_id]),
                "fisher_X": float(fisher_stat_per_pair[row_id]),
                "n_objects": int(n_observed_tests_per_pair[row_id]),
                "n_layers": int(n_layers_seen_per_pair[row_id]),
                "layer_mask": int(layers_seen_bitmask[row_id]),
            },
        )
    return global_graph


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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def attach_global_layer(layers_by_name: dict[str, nx.Graph]) -> None:
    """Build the GLOBAL layer (the Fisher combination of every modality's
    p-values, per pair) and add it into layers_by_name under GLOBAL_LAYER,
    so the caller's per-layer loop places it exactly like a real layer.

    This pools "p_values", the raw per-object list -- not PVALUE_ATTR
    ("p_modality"), which is already Fisher-combined per layer. Pooling
    p_modality across layers would be a Fisher of Fishers, a different and
    less powerful test than one Fisher combination over every shared
    object; see build_global_pvalue_graph's docstring for the full reason.

    Mutates layers_by_name in place. Logs a warning and leaves
    layers_by_name unchanged if the GLOBAL layer cannot be built -- this is
    not fatal, since the real per-modality layers can still be placed.
    """
    try:
        global_graph = build_global_pvalue_graph(layers_by_name)
        n_modalities = len(layers_by_name)
        layers_by_name[GLOBAL_LAYER] = global_graph
        logger.info(
            f"{GLOBAL_LAYER}: {global_graph.number_of_nodes()} nodes, "
            f"{global_graph.number_of_edges()} edges, Fisher over {n_modalities} "
            f"modalities via 'p_values'"
        )
    except Exception as e:
        logger.warning(f"could not build the {GLOBAL_LAYER} layer: {e!r}")


def place_one_layer(
    graph: nx.Graph,
    layer_name: str,
    ground_truth: GroundTruthMembership,
    args: argparse.Namespace,
) -> LayerCoordinates | None:
    """Resolve one layer's edge attribute, accumulate its pair sums, and
    place it on the detectability diagrams.

    Any failure (a bad edge attribute, zero id overlap with the ground
    truth, a layer that can't be placed at all, ...) is logged as a warning
    or error here and reported as None, not raised -- one bad layer must
    never abort the rest of a country's layers.

    Args:
        graph: The layer's graph. May be the GLOBAL layer.
        layer_name: The layer's name, e.g. "hashtags" or GLOBAL_LAYER.
        ground_truth: This country's ground truth, from
            load_ground_truth_membership().
        args: Parsed CLI args.

    Returns:
        The layer's coordinates, or None if it could not be placed.
    """
    try:
        attr_name = GLOBAL_ATTR if layer_name == GLOBAL_LAYER else PVALUE_ATTR

        all_truth_account_ids = (
            ground_truth.operation_account_ids | ground_truth.control_account_ids
        )
        graph_account_ids = {normalize_account_id(node) for node in graph.nodes()}
        if graph_account_ids and not (graph_account_ids & all_truth_account_ids):
            logger.error(
                f"layer '{layer_name}': zero id overlap with the ground truth. "
                f"Sample node {next(iter(graph_account_ids))!r}. Fix the id types."
            )
            return None

        # n is the number of unique accounts in the ground truth.
        population = all_truth_account_ids
        accounts_missing_from_truth = graph_account_ids - all_truth_account_ids
        if accounts_missing_from_truth:
            logger.warning(
                f"layer '{layer_name}': {len(accounts_missing_from_truth)} graph node(s) "
                f"are absent from the ground truth and are not counted in n."
            )

        operation_accounts_in_population = (
            ground_truth.operation_account_ids & population
        )
        control_accounts_in_population = ground_truth.control_account_ids & population
        population_size = len(population)
        operation_size = len(operation_accounts_in_population)
        control_size = len(control_accounts_in_population)

        pair_sums = accumulate_layer(
            graph,
            operation_accounts_in_population,
            control_accounts_in_population,
            attr_name,
            c_max=args.c_max,
        )
        coords = compute_layer_coords(
            pair_sums, population_size, operation_size, control_size, args
        )
        if coords is None:
            logger.warning(
                f"layer '{layer_name}': n={population_size} k={operation_size} "
                f"kc={control_size} -> cannot place; skipped"
            )
            return None
        logger.info(
            f"layer '{layer_name}': n={population_size} k={operation_size} q={coords.q_obs:.3f} "
            f"m_in={coords.m_in:.4f} m0={coords.m0:.4f} s0={coords.s0:.4f} "
            f"beta={coords.beta:.3f} cap={coords.frac_capped:.3f}"
            f"{' null_weak' if coords.flag_null_weak else ''}"
        )
        return coords
    except Exception as e:
        logger.warning(f"layer '{layer_name}': [skip] {e!r}")
        return None


def process_one_country_file(
    layer_file_path: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Load one country's layer file, place every layer (plus the GLOBAL
    layer) on the detectability diagrams, and return one CSV row per placed
    layer.

    Any failure to load the file or its ground truth is logged, and an
    empty list is returned (country is skipped).

    Returns:
        A list of row dicts (LayerCoordinates fields plus "code", "layer"),
        one per successfully placed layer.
    """
    country_code = country_code_from_filename(layer_file_path)
    logger.info(f"=== {os.path.basename(layer_file_path)}  (code {country_code}) ===")
    try:
        layers_by_name = joblib.load(layer_file_path)
        logger.info(f"loaded {len(layers_by_name)} layers from {layer_file_path}")
    except Exception as e:
        logger.error(f"[skip] could not load: {e!r}")
        return []
    if not isinstance(layers_by_name, dict):
        logger.error(
            f"[skip] expected dict of layer->graph, got {type(layers_by_name)}"
        )
        return []

    try:
        ground_truth = load_ground_truth_membership(args.truth_root, country_code)
        logger.info(
            f"ground truth: {len(ground_truth.operation_account_ids)} in-operation, "
            f"{len(ground_truth.control_account_ids)} control"
        )
    except Exception as e:
        logger.error(f"[skip] ground truth error: {e!r}")
        return []

    global_cache_path = os.path.join(args.out_dir, f"{country_code}_global.joblib")
    if os.path.exists(global_cache_path):
        try:
            layers_by_name[GLOBAL_LAYER] = joblib.load(global_cache_path)
            logger.info(f"loaded cached {GLOBAL_LAYER} layer from {global_cache_path}")
        except Exception as e:
            logger.warning(
                f"could not load cached {global_cache_path} ({e!r}); "
                f"recomputing the {GLOBAL_LAYER} layer instead"
            )
            attach_global_layer(layers_by_name)
            joblib.dump(layers_by_name[GLOBAL_LAYER], global_cache_path)
            logger.info(f"saved {GLOBAL_LAYER} layer to {global_cache_path}")
    else:
        # attach_global_layer combines every layer's scores into one GLOBAL
        # layer. Since it is an expensive, memory-hungry step, it is cached
        # once per country. Everything below this (place_one_layer for every layer,
        # including GLOBAL, and therefore every row of layer_coordinates.csv)
        # is still recomputed every time.
        attach_global_layer(layers_by_name)
        joblib.dump(layers_by_name[GLOBAL_LAYER], global_cache_path)
        logger.info(f"saved {GLOBAL_LAYER} layer to {global_cache_path}")

    rows = []
    for layer_name, graph in layers_by_name.items():
        coords = place_one_layer(graph, layer_name, ground_truth, args)
        if coords is None:
            logger.warning(f"layer '{layer_name}': could not be placed; skipped")
            continue
        logger.info(f"layer '{layer_name}': placed successfully")
        rows.append(
            {
                **asdict(coords),
                "code": country_code,
                "layer": str(layer_name),
            }
        )
    return rows


def save_csv_snapshot(rows: list[dict[str, Any]], csv_path: str) -> pd.DataFrame:
    """Write every row placed so far to csv_path (overwriting it), and
    return that same table.

    Args:
        rows: One dict per placed layer, collected so far in the run.
        csv_path: Where to write the CSV. Nothing is written if rows is empty.

    Returns:
        rows as a DataFrame, columns reindexed to CSV_COLS.
    """
    layer_coordinates = pd.DataFrame(rows).reindex(columns=CSV_COLS)
    if rows:
        layer_coordinates.to_csv(csv_path, index=False)
    return layer_coordinates


def run_full_pipeline(args: argparse.Namespace) -> None:
    """Process every layer file under args.layers_dir, writing
    layer_coordinates.csv incrementally.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    layer_files = sorted(glob.glob(os.path.join(args.layers_dir, args.layers_glob)))
    if not layer_files:
        sys.exit(
            f"No layer files under {args.layers_dir} matching '{args.layers_glob}'."
        )

    rows: list[dict[str, Any]] = []
    csv_path = os.path.join(args.out_dir, "layer_coordinates.csv")

    for layer_file_path in layer_files:
        logger.info(f"=== {os.path.basename(layer_file_path)} ===")
        rows.extend(process_one_country_file(layer_file_path, args))
        save_csv_snapshot(rows, csv_path)
        logger.info(f"[progress] {len(rows)} layers placed -> {csv_path}")

    if not rows:
        sys.exit("No layers could be placed.")
    layer_coordinates = save_csv_snapshot(rows, csv_path)
    logger.info(f"Wrote {csv_path} with {len(layer_coordinates)} layers.")
    logger.info("Done.")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    run_full_pipeline(args)


if __name__ == "__main__":
    main()
