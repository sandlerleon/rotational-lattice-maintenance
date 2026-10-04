# -*- coding: utf-8 -*-
"""Where in parameter space the mean-field reduction is bistable at all.

Proposition 2 states bistability "for cooperative feedback (h > 1 and beta above a threshold)". This
computes the threshold. The fold conditions follow the order-repair analysis of the companion model
(github.com/sandlerleon/order-repair-model): steady states satisfy rho = F(I) = mu(I)(1 - I)/I, and the
reduction is bistable exactly when F has an interior local minimum followed by a local maximum. For
each cooperativity h the smallest gain beta_c(h) at which that happens is found by bisection, and the
window (rho_-, rho_+) is reported at the paper's parameters.

For h <= 1 there is no threshold: F(I) = (1 - I)/I + beta I^(h-1)(1 - I)/(K^h + I^h) is then a sum of strictly
decreasing functions, so the steady state is unique for every beta. The scan below confirms this numerically up
to beta = 5000 and shows beta_c diverging as h -> 1+.

    python mf_threshold.py        ->  results/mf_threshold.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))


def turning_points(beta, h, K, n=40000):
    I = np.linspace(1e-5, 1 - 1e-5, n)
    mu = 1 + beta * I ** h / (K ** h + I ** h)
    F = mu * (1 - I) / I
    dF = np.diff(F)
    s = np.sign(dF)
    idx = np.where(s[:-1] != s[1:])[0]
    mins = [(float(I[i + 1]), float(F[i + 1])) for i in idx if s[i] < 0 < s[i + 1]]
    maxs = [(float(I[i + 1]), float(F[i + 1])) for i in idx if s[i] > 0 > s[i + 1]]
    return mins, maxs


def bistable(beta, h, K):
    mins, maxs = turning_points(beta, h, K)
    return bool(mins and maxs)


def beta_c(h, K, lo=0.0, hi=400.0):
    if not bistable(hi, h, K):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bistable(mid, h, K):
            hi = mid
        else:
            lo = mid
    return hi


out = {"K": 0.5, "beta_c": {}, "bistable_anywhere_up_to_beta_5000": {}}
for h in (0.5, 1.0):
    out["bistable_anywhere_up_to_beta_5000"]["%.1f" % h] = any(bistable(b, h, 0.5) for b in np.geomspace(0.1, 5000, 60))
for h in (1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
    b = beta_c(h, 0.5, hi=5000.0)
    out["beta_c"]["%.1f" % h] = b
mins, maxs = turning_points(20.0, 6.0, 0.5)
out["paper_point"] = {"beta": 20.0, "h": 6.0, "K": 0.5, "rho_minus": mins[0][1], "I_at_rho_minus": mins[0][0],
                      "rho_plus": maxs[0][1], "I_at_rho_plus": maxs[0][0],
                      "beta_over_beta_c": 20.0 / out["beta_c"]["6.0"]}
# how the window shrinks as the gain approaches its threshold, at h = 6
bc6 = out["beta_c"]["6.0"]
win = []
for b in (bc6 * 1.05, bc6 * 1.5, bc6 * 2, bc6 * 4, 20.0, 40.0):
    m, x = turning_points(b, 6.0, 0.5)
    win.append({"beta": b, "rho_minus": m[0][1] if m else None, "rho_plus": x[0][1] if x else None})
out["window_vs_beta_h6"] = win
json.dump(out, open(os.path.join(RES, "mf_threshold.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
