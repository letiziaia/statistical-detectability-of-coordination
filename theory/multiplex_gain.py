"""Empirical check of the M^{-1/2} gain."""

import itertools
import json
import math
import os
import time

import matplotlib
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import gammaln

matplotlib.use("Agg")
import matplotlib.pyplot as plt

n = 30
KS = [3, 4, 5]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NULL_REPS, ALT_REPS = 2000, 400
rng = np.random.default_rng(12345)

pair_id = -np.ones((n, n), dtype=np.int64)
c = 0
for i in range(n):
    for j in range(i + 1, n):
        pair_id[i, j] = pair_id[j, i] = c
        c += 1
NP = c


def scan_stat_factory(k):
    sets = np.array(list(itertools.combinations(range(n), k)), dtype=np.int32)
    prs = list(itertools.combinations(range(k), 2))
    P = np.stack([pair_id[sets[:, a], sets[:, b]] for a, b in prs], axis=1)
    return P


def draw(M, mu, k):
    x = rng.gamma(M, 1.0, NP) - M
    if mu > 0:
        blk = [pair_id[i, j] for i in range(k) for j in range(i + 1, k)]
        x[blk] = rng.gamma(M, 1.0 + mu, len(blk)) - M
    return x


def logit_fit(logmu, hits, tot):
    def nll(b):
        z = b[0] + b[1] * logmu
        return -np.sum(
            hits * (-np.logaddexp(0, -z)) + (tot - hits) * (-np.logaddexp(0, z))
        )

    r = minimize(nll, [0.0, 3.0], method="BFGS")
    b = r.x
    H = r.hess_inv
    # mu_50 = exp(-b0/b1); delta method for log mu_50
    g = np.array([-1 / b[1], b[0] / b[1] ** 2])
    se = float(np.sqrt(g @ H @ g))
    return math.exp(-b[0] / b[1]), se, b


out = {}
for k in KS:
    P = scan_stat_factory(k)
    t0 = time.time()
    for M in MS:
        nulls = np.empty(NULL_REPS)
        for r in range(NULL_REPS):
            nulls[r] = (draw(M, 0, k)[P].sum(1)).max() / k
        q95 = float(np.quantile(nulls, 0.95))
        g = 2 * q95 / ((k - 1) * M)
        grid = g * np.array([0.5, 0.62, 0.75, 0.88, 1.0, 1.15, 1.35, 1.6, 2.0])
        hits = []
        for mu in grid:
            h = 0
            for r in range(ALT_REPS):
                h += (draw(M, mu, k)[P].sum(1)).max() / k > q95
            hits.append(h)
        hits = np.array(hits, float)
        mu50, se, b = logit_fit(np.log(grid), hits, ALT_REPS)
        out[f"{k},{M}"] = dict(
            k=k,
            M=M,
            q95=q95,
            q95_se=float(np.std(nulls) / math.sqrt(NULL_REPS) * 1.0),
            mu50=mu50,
            se_log_mu50=se,
            grid=grid.tolist(),
            power=(hits / ALT_REPS).tolist(),
        )
        print(
            f"k={k} M={M} q95={q95:.3f} mu50={mu50:.4f} se={se:.3f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

json.dump(out, open("results.json", "w"))

# ---- Figure and slope table for the M^{-1/2} check ----
R = out
KS = sorted({v["k"] for v in R.values()})
MS = sorted({v["M"] for v in R.values()})
psi = lambda t: -t - np.log1p(-t)
logC = lambda n, k: gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def wcgf(n, k, M, alpha):
    m = k * (k - 1) / 2
    f = lambda w: (
        minimize_scalar(
            lambda t: -k * t * w + M * m * psi(t),
            bounds=(1e-9, 1 - 1e-9),
            method="bounded",
            options={"xatol": 1e-12},
        ).fun
        + logC(n, k)
        - math.log(alpha)
    )
    return brentq(f, 1e-6, 1e5)


def ols(x, y):
    r = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(x) - 2)
    return r.slope, t * r.stderr, r.intercept


rows = []
col = {3: "#4c78a8", 4: "#f58518", 5: "#54a24b", 6: "#b279a2"}
plt.rcParams.update(
    {"font.size": 7, "font.family": "DejaVu Sans", "axes.linewidth": 0.6}
)
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.45))

for k in KS:
    M = np.array(MS, float)
    q = np.array([R[f"{k},{m}"]["q95"] for m in MS])
    mu = np.array([R[f"{k},{m}"]["mu50"] for m in MS])
    se = np.array([R[f"{k},{m}"]["se_log_mu50"] for m in MS])
    s_mu, ci_mu, _ = ols(np.log(M), np.log(mu))
    hi = M >= 16
    s_mu_hi, ci_mu_hi, _ = ols(np.log(M[hi]), np.log(mu[hi]))
    s_q, ci_q, _ = ols(np.log(M), np.log(q))
    s_q_hi, ci_q_hi, _ = ols(np.log(M[hi]), np.log(q[hi]))
    rows.append((k, s_mu, ci_mu, s_mu_hi, ci_mu_hi, s_q, ci_q, s_q_hi, ci_q_hi))
    c = col[k]
    # (a) null threshold
    ax[0].loglog(M, q, "o", color=c, ms=3, label=f"$k={k}$")
    ax[0].loglog(M, [wcgf(n, k, m, 0.05) for m in M], "--", color=c, lw=0.8)
    # (b) mu50
    ax[1].errorbar(
        M, mu, yerr=1.96 * se * mu, fmt="o", color=c, ms=3, lw=0.6, capsize=0
    )
    b = np.polyfit(np.log(M), np.log(mu), 1)
    ax[1].loglog(
        M,
        np.exp(b[1]) * M ** b[0],
        "-",
        color=c,
        lw=0.8,
        label=f"$k={k}$: all $M$ ${s_mu:.2f}$, $M\\geq16$ ${s_mu_hi:.2f}$",
    )
ax[0].set_xlabel("$M$")
ax[0].set_ylabel(r"null quantile $q_{95}$")
ax[0].legend(frameon=False, fontsize=6, loc="upper left")
ax[0].set_title("threshold (dashed: Chernoff bound)", fontsize=6.5)
ax[1].set_xlabel("$M$")
ax[1].set_ylabel(r"$\hat\mu_{50}$")
ax[1].legend(frameon=False, fontsize=6, loc="lower left")
ax[1].set_title(r"detectable signal", fontsize=6.5)

for a in ax:
    a.tick_params(length=2.5, width=0.5)

fig.tight_layout()
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/multilayer_check.pdf")

print("k | slope mu50 (all M) | (M>=16) | slope q95 (all) | (M>=16)")

for r in rows:
    print("%d | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f | %.3f ± %.3f" % r)

json.dump(rows, open("slopes.json", "w"))
