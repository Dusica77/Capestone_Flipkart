"""Part 1: estimation, inference and a regression baseline (all 800 rows)."""
import math

import numpy as np

from common import VALUE, load_data, write_results

data = load_data()
y = data[:, VALUE]
x4 = data[:, 3]
n = len(y)
R = []


def add(k, v):
    R.append((k, v))


# ---- 1.1 estimate with an interval
mean_value = y.mean()
sd_value = y.std(ddof=1)
se_value = sd_value / math.sqrt(n)
add("n", n)
add("mean_value", mean_value)
add("sd_value", sd_value)
add("se_value", se_value)
add("ci95_low", mean_value - 1.96 * se_value)
add("ci95_high", mean_value + 1.96 * se_value)

# ---- 1.2 planning claim
z_stat = (mean_value - 8.0) / se_value
add("z_stat", z_stat)
add("reject_h0", abs(z_stat) > 1.96)

# ---- 1.3 sample size for E = 0.5
add("n_required", math.ceil((1.96 * sd_value / 0.5) ** 2))


# ---- helper: closed-form simple regression (own code)
def simple_ols(x, t):
    mx, mt = x.mean(), t.mean()
    cov = ((x - mx) * (t - mt)).sum() / (len(x) - 1)
    var = ((x - mx) ** 2).sum() / (len(x) - 1)
    slope = cov / var
    intercept = mt - slope * mx
    return slope, intercept, intercept + slope * x


# ---- 1.4 raw regression
slope, intercept, fitted = simple_ols(x4, y)
tss = ((y - y.mean()) ** 2).sum()
ess = ((fitted - y.mean()) ** 2).sum()
rss = ((y - fitted) ** 2).sum()
add("reg_slope", slope)
add("reg_intercept", intercept)
add("tss", tss)
add("ess", ess)
add("rss", rss)
add("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add("r2", 1 - rss / tss)
add("rmse", math.sqrt(rss / n))
add("mae", np.abs(y - fitted).mean())

# ---- 1.5 gradient descent on standardised X4
mx, sx = x4.mean(), x4.std(ddof=1)
z = (x4 - mx) / sx


def cost(m, c):
    e = y - (m * z + c)
    return (e ** 2).mean()


m = c = 0.0
add("gd_J0", cost(m, c))
saved = {}
for step in range(1, 501):
    e = y - (m * z + c)
    dm = -(2 / n) * (z * e).sum()
    dc = -(2 / n) * e.sum()
    m -= 0.1 * dm
    c -= 0.1 * dc
    if step in (1, 10, 100, 500):
        saved[step] = cost(m, c)
for step in (1, 10, 100, 500):
    add(f"gd_J{step}", saved[step])
gd_slope = m / sx
gd_intercept = c - m * mx / sx
add("gd_slope", gd_slope)
add("gd_intercept", gd_intercept)
add("gd_slope_gap", abs(gd_slope - slope))


# ---- 1.6 funnel check and log-log remedy
def half_sds(fit, resid):
    order = np.argsort(fit, kind="stable")
    lo, hi = order[:400], order[400:]
    return resid[lo].std(ddof=1), resid[hi].std(ddof=1)


lo, hi = half_sds(fitted, y - fitted)
add("resid_sd_lower_raw", lo)
add("resid_sd_upper_raw", hi)
add("resid_sd_ratio_raw", hi / lo)

ly, lx = np.log(y), np.log(x4)
ll_slope, ll_int, ll_fit = simple_ols(lx, ly)
ll_resid = ly - ll_fit
ll_rss = (ll_resid ** 2).sum()
ll_tss = ((ly - ly.mean()) ** 2).sum()
ll_r2 = 1 - ll_rss / ll_tss
add("loglog_intercept", ll_int)
add("loglog_slope", ll_slope)
add("loglog_r2", ll_r2)
add("loglog_rmse", math.sqrt(ll_rss / n))
lo, hi = half_sds(ll_fit, ll_resid)
add("resid_sd_lower_log", lo)
add("resid_sd_upper_log", hi)
add("resid_sd_ratio_log", hi / lo)

# ---- 1.7 multiple regression by the normal equation
X = np.column_stack([np.ones(n), lx, data[:, 0], data[:, 1], data[:, 2], data[:, 4], data[:, 5], data[:, 6]])
beta = np.linalg.solve(X.T @ X, X.T @ ly)
for i, b in enumerate(beta):
    add(f"mr_beta_{i}", b)
mr_rss = ((ly - X @ beta) ** 2).sum()
mr_r2 = 1 - mr_rss / ll_tss
add("mr_r2", mr_r2)
add("mr_adj_r2", 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1))
add("loglog_adj_r2", 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1))

write_results("part1_results.txt", R)
