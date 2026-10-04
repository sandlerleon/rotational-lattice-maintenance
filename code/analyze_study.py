# -*- coding: utf-8 -*-
"""Reduce the raw Monte Carlo output to the numbers and figures the manuscript reports.

    python analyze_study.py      ->  results/mc_results.json, figures/Figure4.png, Figure5.png, Figure7.png

Statistics
  * integrated autocorrelation time by Sokal's automatic windowing (window W >= 6 tau_int)
  * interval estimates by bootstrap over independent seeds or disorder realizations (B = 2000,
    percentile intervals), so that between-run variation, not within-run correlation, sets the error
  * the hysteresis area is the signed area between the descending and the ascending branch, which is
    unbiased under noise (the absolute area is not)
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(20261003)
B = 2000


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- statistics
def autocorr(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def tau_int(x, c=6.0):
    rho = autocorr(x)
    tau = 0.5
    for W in range(1, len(rho)):
        tau += rho[W]
        if W >= c * tau:
            return float(tau), W
    return float(tau), len(rho) - 1


def boot_ci(samples, stat=np.mean, b=B):
    s = np.asarray(samples, float)
    idx = RNG.integers(0, len(s), (b, len(s)))
    vals = np.array([stat(s[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bimodality(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    s = x.std()
    g = np.mean((x - m) ** 3) / s ** 3
    k = np.mean((x - m) ** 4) / s ** 4 - 3.0
    return float((g * g + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


# ---------------------------------------------------------------- mean-field branches for overlays
def mf_branches(h=6.0, K=0.5, beta=20.0):
    I = np.linspace(1e-4, 1 - 1e-4, 20000)
    mu = 1 + beta * I ** h / (K ** h + I ** h)
    rho = mu * (1 - I) / I
    dmu = beta * h * K ** h * I ** (h - 1) / (K ** h + I ** h) ** 2
    stable = (dmu * (1 - I) - mu - rho) < 0
    lo = np.argmin(np.where((I > 0.1) & (I < 0.4), rho, np.inf))
    hi = np.argmax(np.where((I > 0.4) & (I < 0.8), rho, -np.inf))
    return I, rho, stable, float(rho[lo]), float(rho[hi])


I_MF, RHO_MF, STAB_MF, RHO_M, RHO_P = mf_branches()


def mf_static_loop_area():
    """Area of the quasi-static mean-field loop: between the saddle-nodes, U on the maintained branch minus U on the
    collapsed branch, integrated over rho. This is what a bistable system keeps as the sweep rate goes to zero."""
    rr = np.linspace(RHO_M + 1e-6, RHO_P - 1e-6, 2000)
    lo = (I_MF < I_MF[np.argmin(np.where((I_MF > 0.1) & (I_MF < 0.4), RHO_MF, np.inf))])
    hi = (I_MF > I_MF[np.argmax(np.where((I_MF > 0.4) & (I_MF < 0.8), RHO_MF, -np.inf))])
    Um = 1 - np.interp(rr, RHO_MF[lo][::-1], I_MF[lo][::-1])
    Uc = 1 - np.interp(rr, RHO_MF[hi][::-1], I_MF[hi][::-1])
    return float(np.trapezoid(Um - Uc, rr))


OUT = {"mean_field_window": [RHO_M, RHO_P], "mean_field_static_loop_area": mf_static_loop_area()}

# ================================================================ A. hysteresis
H = load("hysteresis_raw.json")
rg = np.array(H["rho_grid"])
asc = np.argsort(rg)
dwells = np.array(H["dwells"], float)


def signed_area(run):
    d = np.array(run["down"])[asc]
    u = np.array(run["up"])[asc]
    return float(np.trapezoid(d - u, rg[asc]))


def model(t, a, alpha):
    return a * t ** (-alpha)


PLATEAU = 64          # dwells at or beyond this are the slow-sweep plateau; the zero-rate area is estimated there
hyst = {}
for v, byL in H["runs"].items():
    hyst[v] = {}
    for L, byd in byL.items():
        areas = {d: np.array([signed_area(r) for r in runs]) for d, runs in byd.items()}
        ds = sorted(areas, key=float)
        mean = np.array([areas[d].mean() for d in ds])
        se = np.array([areas[d].std(ddof=1) / math.sqrt(len(areas[d])) for d in ds])
        x = np.array([float(d) for d in ds])
        # zero-rate limit, model-free: every run at a plateau dwell, bootstrapped over runs
        plateau = np.concatenate([areas[d] for d in ds if float(d) >= PLATEAU])
        # decay exponent over the decay region (from the maximum to the last dwell before the plateau)
        k0 = int(np.argmax(mean))
        dec = [i for i in range(k0, len(x)) if x[i] < PLATEAU]
        popt, pcov = curve_fit(model, x[dec], mean[dec], p0=[mean[k0] * x[k0], 1.0], sigma=np.maximum(se[dec], 1e-3),
                               absolute_sigma=True, maxfev=20000)
        chi2 = float(np.sum(((mean[dec] - model(x[dec], *popt)) / np.maximum(se[dec], 1e-3)) ** 2))
        hyst[v][L] = {"dwell": x.tolist(), "area_mean": mean.tolist(), "area_se": se.tolist(),
                      "max_area": float(mean[k0]), "dwell_at_max": float(x[k0]),
                      "A_zero_rate": float(plateau.mean()), "A_zero_rate_ci95": list(boot_ci(plateau)),
                      "plateau_runs": int(len(plateau)),
                      "decay_fit_range": [float(x[dec[0]]), float(x[dec[-1]])], "fit_a": float(popt[0]),
                      "fit_alpha": float(popt[1]), "fit_alpha_se": float(math.sqrt(pcov[1, 1])),
                      "fit_chi2": chi2, "fit_dof": len(dec) - 2}
OUT["hysteresis"] = hyst

# ================================================================ B. branches
Bd = load("branches_raw.json")
dt = Bd["save_every"]
half = Bd["steps"] // 2 // dt
branches = {}
for key, runs in Bd["runs"].items():
    o = np.array(runs["ordered"])
    c = np.array(runs["collapsed"])
    mo = o[:, half:].mean(axis=1)
    mc_ = c[:, half:].mean(axis=1)
    diff_boot = []
    for _ in range(B):
        diff_boot.append(mo[RNG.integers(0, len(mo), len(mo))].mean() - mc_[RNG.integers(0, len(mc_), len(mc_))].mean())
    avg_o, avg_c = o.mean(axis=0), c.mean(axis=0)
    # meeting time: first time the seed-averaged trajectories, in a 20-step moving average, agree within 0.02
    w = max(1, 20 // dt)
    sm = lambda x: np.convolve(x, np.ones(w) / w, "valid")
    gap = np.abs(sm(avg_o) - sm(avg_c))
    t_meet = int(np.argmax(gap < 0.02) * dt) if (gap < 0.02).any() else None
    # per run: first time the collapsed start comes within 0.05 of the stationary mean (same smoothing)
    stat = float(np.concatenate([mo, mc_]).mean())
    def relax(traj, up=True):
        x = sm(np.asarray(traj))
        hit = np.where(x >= stat - 0.05)[0] if up else np.where(x <= stat + 0.05)[0]
        return int(hit[0] * dt) if len(hit) else None
    tr_o = [relax(t, up=False) for t in o]
    tr_c = [relax(t, up=True) for t in c]
    pooled = np.concatenate([o[:, half:].ravel(), c[:, half:].ravel()])
    v, L, rho = key.split("|")
    branches.setdefault(v, {}).setdefault(rho, {})[L] = {
        "U_from_ordered": float(mo.mean()), "U_from_collapsed": float(mc_.mean()),
        "difference": float(mo.mean() - mc_.mean()),
        "difference_ci95": [float(np.percentile(diff_boot, 2.5)), float(np.percentile(diff_boot, 97.5))],
        "seed_sd": float(np.concatenate([mo, mc_]).std(ddof=1)),
        "t_meet_steps": t_meet,
        "relax_from_collapsed_median": float(np.median([x for x in tr_c if x is not None])) if any(x is not None for x in tr_c) else None,
        "relax_from_ordered_median": float(np.median([x for x in tr_o if x is not None])) if any(x is not None for x in tr_o) else None,
        "bimodality_coefficient": bimodality(pooled)}
# per-combination Welch t-test of ordered vs collapsed start, Holm-corrected over all combinations, and the
# combinations whose bootstrap interval excludes zero listed explicitly
from scipy.stats import ttest_ind
tests = []
for key, runs in Bd["runs"].items():
    o = np.array(runs["ordered"])[:, half:].mean(axis=1)
    c = np.array(runs["collapsed"])[:, half:].mean(axis=1)
    tests.append((key, float(ttest_ind(o, c, equal_var=False).pvalue)))
m = len(tests)
order = sorted(range(m), key=lambda i: tests[i][1])
holm, run = {}, 0.0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m - k) * tests[i][1]))
    holm[tests[i][0]] = run
exceptions = []
for key, pw in tests:
    v, L, rho = key.split("|")
    b = branches[v][rho][L]
    b["welch_p"] = pw
    b["holm_p"] = holm[key]
    if not (b["difference_ci95"][0] <= 0 <= b["difference_ci95"][1]):
        exceptions.append({"variant": v, "L": int(L), "rho": float(rho), "difference": b["difference"],
                           "ci95": b["difference_ci95"], "welch_p": pw, "holm_p": holm[key]})
OUT["branches"] = branches
OUT["branches_tests"] = {"n": m, "exceptions": sorted(exceptions, key=lambda e: (e["variant"], e["L"], e["rho"])),
                         "min_holm_p": min(holm.values()),
                         "positive_differences": int(sum(1 for v in branches for r in branches[v] for L in branches[v][r]
                                                         if branches[v][r][L]["difference"] > 0)),
                         "max_abs_difference": max(abs(branches[v][r][L]["difference"]) for v in branches for r in branches[v]
                                                   for L in branches[v][r])}

# ================================================================ C. autocorrelation and current balance
A = load("autocorr_raw.json")
disc = 2000
ac = {}
for key, runs in A["runs"].items():
    tU = [tau_int(np.array(r["U"])[disc:])[0] for r in runs]
    tF = [tau_int(np.array(r["intact"])[disc:])[0] for r in runs]
    jk = np.array([np.mean(r["kill"][disc:]) for r in runs])
    jr = np.array([np.mean(r["repair"][disc:]) for r in runs])
    d = jk - jr
    v, L, rho = key.split("|")
    ac.setdefault(rho, {})[L] = {"tau_int_U": float(np.mean(tU)), "tau_int_U_range": [float(min(tU)), float(max(tU))],
                                 "tau_int_intact": float(np.mean(tF)), "J_kill": float(jk.mean()), "J_repair": float(jr.mean()),
                                 "imbalance": float(d.mean()), "imbalance_ci95": list(boot_ci(d)),
                                 "relative_imbalance": float(abs(d.mean()) / jk.mean()),
                                 "acf_U": autocorr(np.array(runs[0]["U"])[disc:])[:400].tolist()}
OUT["autocorr"] = ac
OUT["autocorr_meta"] = {"steps": A["steps"], "discard": disc, "seeds": A["seeds"]}

# ================================================================ D. FSS
Fd = load("fss_raw.json")
T = Fd["T"]
NK = 2 * T / math.pi
fs = np.array(Fd["f"])
fss = {"T": T, "NK": NK, "sizes": Fd["sizes"], "f": fs.tolist(), "per_L": {}}
Yreal = {}
for L in map(str, Fd["sizes"]):
    Ym, Ys, Bn, tauY = [], [], [], []
    Yreal[L] = []
    for fk in ["%.2f" % f for f in fs]:
        runs = Fd["runs"][L][fk]
        yr = np.array([np.mean(r["Y"]) for r in runs])
        m2 = np.array([np.mean(np.array(r["M"]) ** 2) for r in runs])
        m4 = np.array([np.mean(np.array(r["M"]) ** 4) for r in runs])
        Yreal[L].append(yr)
        Ym.append(yr.mean())
        Ys.append(yr.std(ddof=1) / math.sqrt(len(yr)))
        Bn.append(1 - m4.mean() / (3 * m2.mean() ** 2))
        tauY.append(float(np.mean([tau_int(r["Y"])[0] for r in runs])) * Fd["sample_every"])
    fss["per_L"][L] = {"Y": Ym, "Y_se": Ys, "binder": Bn, "tau_int_Y_sweeps": tauY}


def crossing(y, level=NK):
    y = np.asarray(y)
    for i in range(len(fs) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(fs[i] + (level - y[i]) * (fs[i + 1] - fs[i]) / (y[i + 1] - y[i]))
    return float("nan")


for L in map(str, Fd["sizes"]):
    yr = np.array(Yreal[L])                       # f x realizations
    nre = yr.shape[1]
    boots = [crossing(yr[:, RNG.integers(0, nre, nre)].mean(axis=1)) for _ in range(B)]
    boots = [b for b in boots if np.isfinite(b)]
    fss["per_L"][L]["f_KT"] = crossing(yr.mean(axis=1))
    fss["per_L"][L]["f_KT_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

# extrapolation 1: f_KT(L) = f_inf + b / ln^2 L  (BKT finite-size form)
Ls = np.array([int(L) for L in Fd["sizes"]])
fk = np.array([fss["per_L"][str(L)]["f_KT"] for L in Ls])
X = 1.0 / np.log(Ls) ** 2
coef = np.polyfit(X, fk, 1)
boots = []
for _ in range(B):
    fb = []
    for L in Ls:
        yr = np.array(Yreal[str(L)])
        fb.append(crossing(yr[:, RNG.integers(0, yr.shape[1], yr.shape[1])].mean(axis=1)))
    fb = np.array(fb)
    if np.all(np.isfinite(fb)):
        boots.append(np.polyfit(X, fb, 1)[1])
fss["extrap_lnL2"] = {"f_inf": float(coef[1]), "slope": float(coef[0]),
                      "f_inf_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}

# extrapolation 2: Weber-Minnhagen, Upsilon_L(f*) = (2T/pi)(1 + 1/(2 ln L + C)) at the true f*
fgrid = np.arange(fs[0], fs[-1] + 1e-9, 0.0025)
use = [L for L in Ls if L >= 24]


CGRID = np.linspace(-2, 30, 641)
LNL = np.log(np.array(use, float))
PRED = NK * (1 + 1 / (2 * LNL[None, :] + CGRID[:, None]))          # C x L


def wm_chi2(f, Ymeans, Yses):
    y = np.array([np.interp(f, fs, Ymeans[str(L)]) for L in use])
    s = np.array([max(np.interp(f, fs, Yses[str(L)]), 1e-4) for L in use])
    chi = np.sum(((y[None, :] - PRED) / s[None, :]) ** 2, axis=1)
    k = int(np.argmin(chi))
    return float(chi[k]), float(CGRID[k])


Ymeans = {str(L): fss["per_L"][str(L)]["Y"] for L in Ls}
Yses = {str(L): fss["per_L"][str(L)]["Y_se"] for L in Ls}
chis = [wm_chi2(f, Ymeans, Yses) for f in fgrid]
i_best = int(np.argmin([c[0] for c in chis]))
wm_boot = []
for _ in range(300):
    Yb = {}
    for L in use:
        yr = np.array(Yreal[str(L)])
        Yb[str(L)] = yr[:, RNG.integers(0, yr.shape[1], yr.shape[1])].mean(axis=1).tolist()
    cb = [wm_chi2(f, Yb, Yses)[0] for f in fgrid]
    wm_boot.append(float(fgrid[int(np.argmin(cb))]))
fss["weber_minnhagen"] = {"f_KT": float(fgrid[i_best]), "C": float(chis[i_best][1]),
                          "chi2_min": float(chis[i_best][0]), "dof": len(use) - 1,
                          "f_KT_ci95": [float(np.percentile(wm_boot, 2.5)), float(np.percentile(wm_boot, 97.5))]}
# sensitivity of both extrapolations to the smallest sizes used
sens_fss = {"lnL2": {}, "weber_minnhagen": {}}
for Lmin in (16, 24, 32, 48):
    keep = Ls >= Lmin
    if keep.sum() < 3:
        continue
    cf = np.polyfit(X[keep], fk[keep], 1)
    bb = []
    for _ in range(1000):
        fb = []
        for L in Ls[keep]:
            yr = np.array(Yreal[str(L)])
            fb.append(crossing(yr[:, RNG.integers(0, yr.shape[1], yr.shape[1])].mean(axis=1)))
        if np.all(np.isfinite(fb)):
            bb.append(np.polyfit(X[keep], fb, 1)[1])
    sens_fss["lnL2"][str(Lmin)] = {"sizes": Ls[keep].tolist(), "f_inf": float(cf[1]),
                                   "f_inf_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
for Lmin in (24, 32, 48):
    useL = [L for L in Ls if L >= Lmin]
    lnl = np.log(np.array(useL, float))
    pred = NK * (1 + 1 / (2 * lnl[None, :] + CGRID[:, None]))

    def chi_at(f, Ym):
        y = np.array([np.interp(f, fs, Ym[str(L)]) for L in useL])
        sd = np.array([max(np.interp(f, fs, Yses[str(L)]), 1e-4) for L in useL])
        c = np.sum(((y[None, :] - pred) / sd[None, :]) ** 2, axis=1)
        return float(c.min())
    best = fgrid[int(np.argmin([chi_at(f, Ymeans) for f in fgrid]))]
    bb = []
    for _ in range(300):
        Yb = {}
        for L in useL:
            yr = np.array(Yreal[str(L)])
            Yb[str(L)] = yr[:, RNG.integers(0, yr.shape[1], yr.shape[1])].mean(axis=1).tolist()
        bb.append(float(fgrid[int(np.argmin([chi_at(f, Yb) for f in fgrid]))]))
    sens_fss["weber_minnhagen"][str(Lmin)] = {"sizes": [int(L) for L in useL], "f_KT": float(best),
                                              "f_KT_ci95": [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]}
fss["sensitivity"] = sens_fss
fss["one_minus_pc"] = 1 - 0.5927
OUT["fss"] = fss

# ================================================================ E. the closure D ~ I, tested directly
closure = None
if os.path.exists(os.path.join(RES, "closure_raw.json")):
    Cd = load("closure_raw.json")
    base = Cd["base"]
    beta, hh, KK = base["beta"], base["hill_h"], base["hill_K"]
    closure = {"L": Cd["L"], "seeds": Cd["seeds"]}
    for v in Cd["variants"]:
        rows = []
        for rho in Cd["rhos"]:
            for start in ("ordered", "collapsed"):
                rr = [r for r in Cd["runs"] if r["variant"] == v and r["rho"] == rho and r["start"] == start]
                if not rr:
                    continue
                I = np.array([r["I"] for r in rr])
                Dm = np.array([r["D_intact_mean"] for r in rr])
                mu = np.array([r["mu_lattice"] for r in rr])
                Ib = I.mean()
                rows.append({"rho": rho, "start": start, "I": float(Ib), "D_intact": float(Dm.mean()),
                             "D_minus_I": float((Dm - I).mean()), "D_minus_I_ci95": list(boot_ci(Dm - I)),
                             "mu_lattice": float(mu.mean()),
                             "mu_closure": float(1 + beta * Ib ** hh / (KK ** hh + Ib ** hh)),
                             "mu_at_mean_delta": float(1 + beta * Dm.mean() ** hh / (KK ** hh + Dm.mean() ** hh)),
                             "intact_fraction": float(np.mean([r["intact_fraction"] for r in rr]))})
        closure[v] = rows
    OUT["closure"] = closure
    # the ceiling of local disorder: mean of 1 - |average of n independent uniformly random unit vectors|
    rz = np.exp(1j * RNG.uniform(-np.pi, np.pi, (200000, 80)))
    closure["delta_of_random_neighbourhood"] = {str(n): float(np.mean(1 - np.abs(rz[:, :n].mean(axis=1)))) for n in (4, 8, 24, 80)}

# ================================================================ F. sensing range
sensing = None
if os.path.exists(os.path.join(RES, "sensing_raw.json")):
    Sd = load("sensing_raw.json")
    sensing = {"L": Sd["L"], "seeds": Sd["seeds"], "neighbours": {}, "sweep": {}, "two": {}}
    NB = {"nn": 4, "global": Sd["L"] ** 2 - 1}
    for sv in Sd["sensing"]:
        if sv.startswith("r"):
            NB[sv] = (2 * int(sv[1:]) + 1) ** 2 - 1
    sensing["neighbours"] = NB
    # the nearest-neighbour model at the same size, from studies A and B
    Lref = str(Sd["L"])
    ref_sweep = {repr(float(d)): H["runs"]["default"][Lref][str(d)] for d in Sd["dwells"] if str(d) in H["runs"]["default"].get(Lref, {})}
    allsweep = dict(Sd["runs"]["sweep"])
    if ref_sweep:
        allsweep["nn"] = ref_sweep
    for sv, byd in allsweep.items():
        sensing["sweep"][sv] = {}
        for d, runs in byd.items():
            a = np.array([signed_area(r) for r in runs])
            dn = np.mean([np.array(r["down"])[asc] for r in runs], axis=0)
            up = np.mean([np.array(r["up"])[asc] for r in runs], axis=0)
            sensing["sweep"][sv][str(int(float(d)))] = {"area": float(a.mean()), "area_ci95": list(boot_ci(a)), "n": int(len(a)),
                                                        "down": dn.tolist(), "up": up.tolist()}
    half = Sd["steps"] // 2 // Sd["save_every"]
    alltwo = dict(Sd["runs"]["two"])
    for sv, byr in alltwo.items():
        sensing["two"][sv] = {}
        for r, runs in byr.items():
            mo = np.array([np.mean(x["ordered"][half:]) for x in runs])
            mc_ = np.array([np.mean(x["collapsed"][half:]) for x in runs])
            db = [mo[RNG.integers(0, len(mo), len(mo))].mean() - mc_[RNG.integers(0, len(mc_), len(mc_))].mean() for _ in range(B)]
            sensing["two"][sv][str(float(r))] = {"U_ordered": float(mo.mean()), "U_collapsed": float(mc_.mean()),
                                                 "difference": float(mo.mean() - mc_.mean()),
                                                 "difference_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}
    if "default" in branches:
        sensing["two"]["nn"] = {}
        for r in branches["default"]:
            if Lref in branches["default"][r]:
                b = branches["default"][r][Lref]
                sensing["two"]["nn"][str(float(r))] = {"U_ordered": b["U_from_ordered"], "U_collapsed": b["U_from_collapsed"],
                                                       "difference": b["difference"], "difference_ci95": b["difference_ci95"],
                                                       "note": "from study B (3000 steps, 8 seeds)"}
    OUT["sensing"] = sensing

# ================================================================ G. extended-range bistability in size and time
extended = None
if os.path.exists(os.path.join(RES, "extended_raw.json")):
    Ed = load("extended_raw.json")
    dt_e = Ed["save_every"]
    w_e = max(1, 100 // dt_e)                      # 100-step moving average
    smE = lambda x: np.convolve(np.asarray(x, float), np.ones(w_e) / w_e, "valid")
    UMID = 0.6                                     # between the collapsed (U < 0.5) and the ordered (U > 0.75) branches
    T0 = 500                                       # transient allowed before a run is assigned to a branch
    halfE = len(next(iter(Ed["runs"].values()))["ordered"][0]["U"]) // 2
    extended = {"sizes": Ed["sizes"], "seeds": Ed["seeds"], "steps": Ed["steps"], "threshold": UMID, "transient": T0, "cases": {}}
    for key, byst in Ed["runs"].items():
        sv, L, rho = key.split("|")
        rec = {}
        for start in ("ordered", "collapsed"):
            runs = byst[start]
            late = np.array([np.mean(r["U"][halfE:]) for r in runs])
            exits, exposure = [], 0.0
            for r in runs:
                x = smE(r["U"])
                i0 = T0 // dt_e
                if i0 >= len(x):
                    continue
                side = x[i0] > UMID
                cross = np.where((x[i0:] > UMID) != side)[0]
                if len(cross):
                    exits.append(int((i0 + cross[0]) * dt_e))
                    exposure += (cross[0]) * dt_e
                else:
                    exposure += (len(x) - i0) * dt_e
            rec[start] = {"U_late": late.tolist(), "U_late_mean": float(late.mean()),
                          "branch_at_transient": [bool(smE(r["U"])[T0 // dt_e] > UMID) for r in runs],
                          "n_runs": len(runs), "n_exits": len(exits), "exit_steps": exits, "exposure_steps": exposure,
                          "exit_rate_per_step": (len(exits) / exposure) if exposure > 0 else None}
        o = np.array(rec["ordered"]["U_late"])
        c = np.array(rec["collapsed"]["U_late"])
        db = [o[RNG.integers(0, len(o), len(o))].mean() - c[RNG.integers(0, len(c), len(c))].mean() for _ in range(B)]
        rec["difference"] = float(o.mean() - c.mean())
        rec["difference_ci95"] = [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]
        extended["cases"].setdefault(sv, {}).setdefault(rho, {})[L] = rec
    OUT["extended"] = extended

with open(os.path.join(RES, "mc_results.json"), "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote results/mc_results.json")

# ================================================================ figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 7})
COL = {16: "#9ecae1", 24: "#6baed6", 32: "#3182bd", 48: "#08519c", 64: "#08306b", 96: "#000000"}


def tag(ax, t):
    ax.text(-0.15, 1.06, t, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# ---- Fig_lattice: lattice test of the bistability (Section 4.3)
fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.0))
fig.subplots_adjust(hspace=0.42, wspace=0.30)
ax = axs[0, 0]
ax.axvspan(RHO_M, RHO_P, color="#fdf1c9", zorder=0)
for d, c in ((1, "#c0392b"), (16, "#e08e0b"), (256, "#1a7a4c")):
    runs = H["runs"]["default"]["64"][str(d)]
    dn = np.mean([np.array(r["down"])[asc] for r in runs], axis=0)
    up = np.mean([np.array(r["up"])[asc] for r in runs], axis=0)
    ax.plot(rg[asc], dn, "-", color=c, lw=1.4, label="%d step%s per $\\rho$" % (d, "" if d == 1 else "s"))
    ax.plot(rg[asc], up, "--", color=c, lw=1.4)
st = STAB_MF
ax.plot(RHO_MF[st & (I_MF < 0.4)], 1 - I_MF[st & (I_MF < 0.4)], color="#888", lw=1, alpha=0.8)
ax.plot(RHO_MF[st & (I_MF > 0.4)], 1 - I_MF[st & (I_MF > 0.4)], color="#888", lw=1, alpha=0.8, label="mean-field stable branches")
ax.set_xlim(1, 16)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"$U_\chi$")
ax.set_title("KMC sweeps, L = 64 (solid: down, dashed: up)", loc="left")
ax.legend(frameon=False, loc="lower right")
tag(ax, "A")

ax = axs[0, 1]
for L in sorted(map(int, hyst["default"])):
    h = hyst["default"][str(L)]
    x = np.array(h["dwell"])
    ax.errorbar(x, h["area_mean"], yerr=1.96 * np.array(h["area_se"]), fmt="o-", ms=3.5, lw=0.8, color=COL[L], capsize=2, label="L = %d" % L)
if "cold0" in hyst:
    for L in sorted(map(int, hyst["cold0"])):
        h = hyst["cold0"][str(L)]
        ax.errorbar(h["dwell"], h["area_mean"], yerr=1.96 * np.array(h["area_se"]), fmt="s--", ms=3.5, color="#c0392b",
                    alpha=0.5 + 0.25 * (L == 48), capsize=2, lw=0.9, label="L = %d, T = 0.30, $h_0$ = 0" % L)
ax.axhline(0, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xlabel(r"dwell per $\rho$ value (steps)")
ax.set_ylabel(r"signed loop area $\int (U_\downarrow - U_\uparrow)\,d\rho$")
ax.set_title("Loop area against sweep rate", loc="left")
ax.legend(frameon=False, fontsize=6.5)
tag(ax, "B")

ax = axs[1, 0]
for v, mk, lab in (("default", "o", "T = 0.45, $h_0$ = 0.30"), ("cold0", "s", "T = 0.30, $h_0$ = 0")):
    if v not in branches:
        continue
    for k, (start, fill) in enumerate((("U_from_ordered", True), ("U_from_collapsed", False))):
        rr = sorted(branches[v], key=float)
        Lmax = max(int(x) for x in branches[v][rr[0]])
        y = [branches[v][r][str(Lmax)][start] for r in rr]
        ax.plot([float(r) for r in rr], y, mk + ("-" if fill else "--"), mfc=None if fill else "none",
                color="#08519c" if v == "default" else "#c0392b", ms=8 if fill else 4.5, lw=1, zorder=2 if fill else 3,
                label="%s, from %s (L = %d)" % (lab, "ordered" if fill else "collapsed", Lmax))
ax.axvspan(RHO_M, RHO_P, color="#fdf1c9", zorder=0)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"stationary $U_\chi$")
ax.set_title("Two starts inside the mean-field window", loc="left")
ax.legend(frameon=False, fontsize=6.3, loc="lower right")
tag(ax, "C")

ax = axs[1, 1]
for rho in sorted(ac, key=float):
    for L in sorted(ac[rho], key=int):
        a = ac[rho][L]
        ax.errorbar(a["J_kill"], a["J_repair"], fmt="o", color=COL[int(L)], ms=4)
lim = [min(min(a["J_kill"] for a in d.values()) for d in ac.values()) * 0.95,
       max(max(a["J_kill"] for a in d.values()) for d in ac.values()) * 1.05]
ax.plot(lim, lim, "k-", lw=0.7)
ax.set_xlabel("kill current per site per step")
ax.set_ylabel("repair current per site per step")
ax.set_title("Stationary current balance", loc="left")
tag(ax, "D")
fig.savefig(os.path.join(FIG, "Fig_lattice.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Fig_fss: FSS (Section 5)
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
fig.subplots_adjust(wspace=0.32)
ax = axs[0]
for L in Ls:
    d = fss["per_L"][str(L)]
    ax.errorbar(fs, d["Y"], yerr=1.96 * np.array(d["Y_se"]), fmt="o-", ms=3, lw=1, color=COL[int(L)], capsize=2, label="L = %d" % L)
ax.axhline(NK, color="k", ls="--", lw=1)
ax.text(fs[-1], NK + 0.015, r"$2T/\pi$", ha="right", fontsize=7.5)
ax.set_xlabel("diluted fraction f")
ax.set_ylabel(r"helicity modulus $\Upsilon$")
ax.legend(frameon=False, ncol=2)
ax.set_title(r"$\Upsilon(f)$, T = 0.45, $h_0$ = 0", loc="left")
tag(ax, "A")
ax = axs[1]
lo = [fss["per_L"][str(L)]["f_KT"] - fss["per_L"][str(L)]["f_KT_ci95"][0] for L in Ls]
hi = [fss["per_L"][str(L)]["f_KT_ci95"][1] - fss["per_L"][str(L)]["f_KT"] for L in Ls]
ax.errorbar(X, fk, yerr=[lo, hi], fmt="o", color="#08519c", capsize=3)
xx = np.linspace(0, X.max() * 1.05, 50)
ax.plot(xx, coef[1] + coef[0] * xx, "-", color="#08519c", lw=1)
e = fss["extrap_lnL2"]
ax.errorbar([0], [e["f_inf"]], yerr=[[e["f_inf"] - e["f_inf_ci95"][0]], [e["f_inf_ci95"][1] - e["f_inf"]]], fmt="s",
            color="#c0392b", capsize=3, label=r"extrapolated, $1/\ln^2 L \to 0$")
w = fss["weber_minnhagen"]
ax.errorbar([0.002], [w["f_KT"]], yerr=[[w["f_KT"] - w["f_KT_ci95"][0]], [w["f_KT_ci95"][1] - w["f_KT"]]], fmt="D",
            color="#1a7a4c", capsize=3, label="Weber–Minnhagen fit")
ax.axhline(fss["one_minus_pc"], color="k", ls=":", lw=1)
ax.text(X.max(), fss["one_minus_pc"] - 0.012, r"$1-p_c = 0.407$", ha="right", va="top", fontsize=7.5)
for L, x, y in zip(Ls, X, fk):
    ax.annotate("%d" % L, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6.5)
ax.set_xlabel(r"$1/\ln^2 L$")
ax.set_ylabel(r"$f_{KT}(L)$")
ax.set_ylim(0.1, 0.45)
ax.legend(frameon=False, loc="center right")
ax.set_title("Crossing of $2T/\\pi$ and its extrapolation", loc="left")
tag(ax, "B")
ax = axs[2]
for L in Ls:
    ax.plot(fs, fss["per_L"][str(L)]["binder"], "o-", ms=3, lw=1, color=COL[int(L)], label="L = %d" % L)
ax.axvline(fss["one_minus_pc"], color="k", ls=":", lw=1)
ax.set_xlabel("diluted fraction f")
ax.set_ylabel(r"Binder cumulant $U_4$")
ax.set_title("Binder cumulant", loc="left")
tag(ax, "C")
fig.savefig(os.path.join(FIG, "Fig_fss.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Fig_diagnostics: Monte Carlo diagnostics (appendix)
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
fig.subplots_adjust(wspace=0.33)
ax = axs[0]
for rho, c in zip(sorted(ac, key=float), ("#c0392b", "#e08e0b", "#1a7a4c")):
    a = ac[rho][str(max(int(x) for x in ac[rho]))]
    ax.plot(np.arange(len(a["acf_U"])), a["acf_U"], color=c, lw=1.2, label=r"$\rho$ = %s, $\tau_{int}$ = %.0f steps" % (rho, a["tau_int_U"]))
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(0, 300)
ax.set_xlabel("lag (steps)")
ax.set_ylabel(r"autocorrelation of $U_\chi$")
ax.set_title("Stationary autocorrelation, L = %d" % max(int(x) for x in ac[sorted(ac)[0]]), loc="left")
ax.legend(frameon=False)
tag(ax, "A")
ax = axs[1]
for rho, c in zip(sorted(ac, key=float), ("#c0392b", "#e08e0b", "#1a7a4c")):
    Lz = sorted(ac[rho], key=int)
    ax.plot([int(L) for L in Lz], [ac[rho][L]["tau_int_U"] for L in Lz], "o-", color=c, label=r"$\rho$ = %s" % rho)
ax.set_xscale("log", base=2)
ax.set_xticks([16, 32, 64])
ax.set_xticklabels(["16", "32", "64"])
ax.set_xlabel("L")
ax.set_ylabel(r"$\tau_{int}(U_\chi)$ (steps)")
ax.set_title("Integrated autocorrelation time", loc="left")
ax.legend(frameon=False)
tag(ax, "B")
ax = axs[2]
for v, c in (("default", "#08519c"), ("cold0", "#c0392b")):
    if v not in branches:
        continue
    for rho, mk in zip(sorted(branches[v], key=float), "osD^"):
        Lz = sorted(branches[v][rho], key=int)
        y = [branches[v][rho][L]["t_meet_steps"] for L in Lz]
        x = [int(L) for L, yy in zip(Lz, y) if yy is not None]
        y = [yy for yy in y if yy is not None]
        ax.plot(x, y, mk + "-", color=c, ms=4, lw=0.9, alpha=0.85,
                label=(r"%s, $\rho$ = %s" % ("T = 0.45" if v == "default" else "T = 0.30, $h_0$ = 0", rho)))
ax.set_xscale("log", base=2)
ax.set_xticks([16, 24, 32, 48, 64])
ax.set_xticklabels(["16", "24", "32", "48", "64"])
ax.minorticks_off()
ax.set_ylim(0, 175)
ax.set_xlabel(r"L  (N = $L^2$ = 256 to 4096 sites)")
ax.set_ylabel("steps until the two starts agree within 0.02")
ax.set_title("Time for the two starts to meet", loc="left")
ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper center")
tag(ax, "C")
fig.savefig(os.path.join(FIG, "Fig_diagnostics.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- Fig_mechanism: why the lattice is not bistable, and the sensing range that restores it (Section 4.4)
fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.0))
fig.subplots_adjust(hspace=0.42, wspace=0.30)
axs = axs.ravel()
ax = axs[0]
if closure:
    xx = np.linspace(0, 1, 50)
    ax.plot(xx, xx, "k-", lw=0.7, label=r"closure $\bar\delta = I$")
    for v, c, lab in (("default", "#08519c", "T = 0.45, $h_0$ = 0.30"), ("cold0", "#c0392b", "T = 0.30, $h_0$ = 0")):
        if v not in closure:
            continue
        for start, fill in (("ordered", True), ("collapsed", False)):
            r = [x for x in closure[v] if x["start"] == start]
            ax.plot([x["I"] for x in r], [x["D_intact"] for x in r], "o", color=c, ms=7.5 if fill else 4,
                    mfc=c if fill else "white", zorder=2 if fill else 3, label="%s, from %s" % (lab, start))
    ax.axhline(KK, color="#888", ls=":", lw=0.9)
    ax.text(0.98, KK + 0.01, "K", ha="right", fontsize=7, color="#666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"global information variable $I = 1 - U_\chi$")
    ax.set_ylabel(r"mean local disorder of intact sites $\bar\delta$")
    ax.set_title("Testing the mean-field closure, L = %d" % closure["L"], loc="left")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
else:
    ax.axis("off")
tag(ax, "A")

SCOL = {"nn": "#08306b", "r1": "#3182bd", "r2": "#6baed6", "r4": "#e08e0b", "global": "#c0392b"}
SLAB = {"nn": "4 nearest neighbours (model)", "r1": "block of 8", "r2": "block of 24", "r4": "block of 80", "global": "global (mean-field closure)"}
ORDER = [x for x in ("nn", "r1", "r2", "r4", "global") if sensing and x in sensing["sweep"]]
ax = axs[1]
if sensing:
    ax.axvspan(RHO_M, RHO_P, color="#fdf1c9", zorder=0)
    for sv in ORDER:
        d = sensing["sweep"][sv].get("256")
        if d:
            ax.plot(rg[asc], d["down"], "-", color=SCOL[sv], lw=1.3, label=SLAB[sv])
            ax.plot(rg[asc], d["up"], "--", color=SCOL[sv], lw=1.3)
    ax.set_xlim(1, 16)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$U_\chi$")
    ax.set_title("Slow sweeps (256 steps per $\\rho$), L = %d" % sensing["L"], loc="left")
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
else:
    ax.axis("off")
tag(ax, "B")
ax = axs[2]
if sensing:
    ax.axvspan(RHO_M, RHO_P, color="#fdf1c9", zorder=0)
    for sv in ORDER:
        if sv not in sensing["two"]:
            continue
        rr = sorted(sensing["two"][sv], key=float)
        y = np.array([sensing["two"][sv][r]["difference"] for r in rr])
        lo = y - np.array([sensing["two"][sv][r]["difference_ci95"][0] for r in rr])
        hi = np.array([sensing["two"][sv][r]["difference_ci95"][1] for r in rr]) - y
        ax.errorbar([float(r) for r in rr], y, yerr=[lo, hi], fmt="o-", ms=3.5, lw=1, capsize=2, color=SCOL[sv], label=SLAB[sv])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$U_\chi$(ordered start) $-$ $U_\chi$(collapsed start)")
    ax.set_title("Two starts at fixed $\\rho$", loc="left")
    ax.legend(frameon=False, fontsize=6.3)
else:
    ax.axis("off")
tag(ax, "C")
ax = axs[3]
if sensing:
    for d, mk in (("64", "s"), ("256", "o")):
        xs, ys, lo, hi = [], [], [], []
        for sv in ORDER:
            e = sensing["sweep"][sv].get(d)
            if e:
                xs.append(sensing["neighbours"][sv]); ys.append(e["area"])
                lo.append(e["area"] - e["area_ci95"][0]); hi.append(e["area_ci95"][1] - e["area"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk + "-", ms=4.5, lw=1, capsize=2, color="#08519c" if d == "256" else "#9ecae1",
                    label="%s steps per $\\rho$" % d)
    for sv in ORDER:
        e = sensing["sweep"][sv].get("256")
        if e:
            ax.annotate({"nn": "4 (model)", "r1": "8", "r2": "24", "r4": "80", "global": "global"}[sv],
                        (sensing["neighbours"][sv], e["area"]), textcoords="offset points", xytext=(-6, 8), fontsize=6.5, ha="right")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("sites sensed by each kill rate")
    ax.set_ylabel(r"slow-sweep loop area")
    ax.set_title("Loop area against sensing range", loc="left")
    ax.legend(frameon=False)
else:
    ax.axis("off")
tag(ax, "D")
fig.savefig(os.path.join(FIG, "Fig_mechanism.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
# ---- Fig_extended: extended-range bistability in size and time (Appendix, Figure A2)
if extended:
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.30)
    axs = axs.ravel()
    LCOL = {32: "#9ecae1", 48: "#4292c6", 64: "#08519c", 96: "#000000"}
    for ax, sv, title in ((axs[0], "r4", "80-site neighbourhood"), (axs[1], "global", "global sensing")):
        cs = extended["cases"][sv]
        for k, L in enumerate(sorted({int(L) for r in cs for L in cs[r]})):
            rr = sorted(cs, key=float)
            xo = [float(r) + (k - 1.5) * 0.06 for r in rr]
            ax.plot(xo, [np.mean(cs[r][str(L)]["ordered"]["U_late"]) for r in rr], "o", color=LCOL[L], ms=6, label="L = %d" % L)
            ax.plot(xo, [np.mean(cs[r][str(L)]["collapsed"]["U_late"]) for r in rr], "o", color=LCOL[L], ms=6, mfc="white")
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$U_\chi$, second half of the run")
        ax.set_title("%s, L = 32–96" % title, loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    tag(axs[0], "A")
    tag(axs[1], "B")
    Ed = load("extended_raw.json")
    tt = np.arange(len(Ed["runs"]["r4|32|5.0"]["ordered"][0]["U"])) * Ed["save_every"]
    for ax, key, st, title in ((axs[2], "5.0", "ordered", r"80 sites, $\rho$ = 5.0, ordered start"),
                               (axs[3], "6.5", "collapsed", r"80 sites, $\rho$ = 6.5, collapsed start")):
        for L in (32, 96):
            for j, r in enumerate(Ed["runs"]["r4|%d|%s" % (L, key)][st]):
                ax.plot(tt[9:], np.convolve(r["U"], np.ones(10) / 10, "valid"), color=LCOL[L], lw=0.8, alpha=0.85,
                        label="L = %d" % L if j == 0 else None)
        ax.axhline(extended["threshold"], color="#888", ls=":", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("steps")
        ax.set_ylabel(r"$U_\chi$ (100-step average)")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.5, loc="center right")
    tag(axs[2], "C")
    tag(axs[3], "D")
    fig.savefig(os.path.join(FIG, "Fig_extended.png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

print("figures written")
