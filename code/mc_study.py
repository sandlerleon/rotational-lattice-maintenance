# -*- coding: utf-8 -*-
"""Monte Carlo study for the revised manuscript: six analyses that the earlier version asserted, or
that its approximation left untested.

  hysteresis  loop area of the rho sweep against dwell time per rho value and lattice size, with the
              extrapolation to an infinitely slow sweep (static hysteresis) and its bootstrap interval
  branches    at fixed rho inside the mean-field bistable window, runs started from the ordered and from
              the collapsed state: stationary order from each start, time for the two to meet, and the
              bimodality of the stationary distribution
  autocorr    integrated autocorrelation times in the stationary state, the equilibration discard they
              justify, and the stationary current balance with interval estimates
  fss         field-free quenched dilution (Section 5) at six sizes and eight disorder realizations,
              helicity-modulus crossings with bootstrap intervals and a Weber-Minnhagen extrapolation
  sensing     the same lattice with the disorder sensed over a wider block (r = 2, 4) or globally (the mean-field
              closure imposed on the lattice): slow-sweep loops and two-start tests, to locate what removes
              the bistability
  closure     a direct test of the one approximation of the mean-field reduction, D ~ I: in the stationary
              state, the mean local disorder of intact sites and the effective kill-rate multiplier the
              lattice actually applies, against the global information variable I = 1 - U and against
              mu(I) evaluated as the mean-field closure assumes

Two parameter sets are run where the question depends on them:
  default   T = 0.45, h0 = 0.30  (the MyUncle defaults used for the manuscript's earlier lattice figures)
  cold0     T = 0.30, h0 = 0     (weaker thermal noise and no ordering field, the regime most favourable
                                  to lattice bistability)

    python mc_study.py hysteresis|branches|autocorr|fss|sensing|closure|all  [--quick]

Every task is seeded from SeedSequence([MASTER, task code]), so any single run can be reproduced alone.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lattice_engine import Lattice, Params, Quenched  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(RES, exist_ok=True)
MASTER = 20261003
QUICK = "--quick" in sys.argv
NPROC = max(1, (os.cpu_count() or 2) - 1)

VARIANTS = {"default": dict(T=0.45, h0=0.30), "cold0": dict(T=0.30, h0=0.0)}
BASE = dict(g0=0.12, lam0=0.02, beta=20.0, hill_h=6.0, hill_K=0.5, relax=2)


def seed_of(*code):
    return int(np.random.SeedSequence([MASTER] + [int(c) for c in code]).generate_state(1)[0])


def params(variant, L):
    return Params(L=L, **BASE, **VARIANTS[variant])


# ======================================================================= A. hysteresis
RHO_GRID = np.round(np.arange(16.0, 0.99, -0.5), 3)          # 16 -> 1, 31 points
DWELLS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def task_hysteresis(args):
    variant, L, dwell, s = args
    p = params(variant, L).with_rho(float(RHO_GRID[0]))
    lat = Lattice(p, seed_of(1, list(VARIANTS).index(variant), L, dwell, s))
    for _ in range(200):                                       # settle on the ordered branch at rho = 16
        lat.step()
    down, up = [], []
    for r in RHO_GRID:
        lat.p = p.with_rho(float(r))
        for _ in range(dwell):
            lat.step()
        down.append(lat.order())
    for r in RHO_GRID[::-1]:
        lat.p = p.with_rho(float(r))
        for _ in range(dwell):
            lat.step()
        up.append(lat.order())
    return variant, L, dwell, s, down, up[::-1]


def run_hysteresis():
    sizes = {"default": [16, 24, 32, 48, 64], "cold0": [24, 48]}
    seeds = 3 if QUICK else 8
    dwells = DWELLS[:5] if QUICK else DWELLS
    tasks = [(v, L, d, s) for v in sizes for L in sizes[v] for d in dwells for s in range(seeds)]
    tasks.sort(key=lambda t: -t[1] * t[1] * t[2])               # longest first for load balance
    out = {}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for i, (v, L, d, s, down, up) in enumerate(pool.imap_unordered(task_hysteresis, tasks)):
            out.setdefault(v, {}).setdefault(str(L), {}).setdefault(str(d), []).append({"seed": s, "down": down, "up": up})
            if (i + 1) % 50 == 0:
                print("  hysteresis %d/%d  (%.0f s)" % (i + 1, len(tasks), time.time() - t0), flush=True)
    json.dump({"rho_grid": RHO_GRID.tolist(), "dwells": dwells, "seeds": seeds, "sizes": sizes, "variants": VARIANTS,
               "base": BASE, "runs": out}, open(os.path.join(RES, "hysteresis_raw.json"), "w"))
    print("hysteresis done in %.0f s" % (time.time() - t0))


# ======================================================================= B. branches
BR_RHOS = [4.5, 6.0, 8.0, 10.0]
BR_STEPS = 3000
SAVE_EVERY = 5


def task_branch(args):
    variant, L, rho, s, start = args
    p = params(variant, L).with_rho(rho)
    lat = Lattice(p, seed_of(2, list(VARIANTS).index(variant), L, int(rho * 10), s, start))
    if start == 1:
        lat.set_collapsed(0.95, random_angles=True)
    traj = []
    for t in range(BR_STEPS):
        lat.step()
        if t % SAVE_EVERY == 0:
            traj.append(lat.order())
    return variant, L, rho, s, start, traj


def run_branches():
    sizes = [16, 24, 32] if QUICK else [16, 24, 32, 48, 64]
    seeds = 3 if QUICK else 8
    tasks = [(v, L, r, s, st) for v in VARIANTS for L in sizes for r in BR_RHOS for s in range(seeds) for st in (0, 1)]
    tasks.sort(key=lambda t: -t[1])
    out = {}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for i, (v, L, r, s, st, traj) in enumerate(pool.imap_unordered(task_branch, tasks)):
            key = "%s|%d|%.1f" % (v, L, r)
            out.setdefault(key, {"ordered": [], "collapsed": []})["ordered" if st == 0 else "collapsed"].append(traj)
            if (i + 1) % 100 == 0:
                print("  branches %d/%d  (%.0f s)" % (i + 1, len(tasks), time.time() - t0), flush=True)
    json.dump({"rhos": BR_RHOS, "steps": BR_STEPS, "save_every": SAVE_EVERY, "sizes": sizes, "seeds": seeds,
               "variants": VARIANTS, "base": BASE, "runs": out}, open(os.path.join(RES, "branches_raw.json"), "w"))
    print("branches done in %.0f s" % (time.time() - t0))


# ======================================================================= C. autocorrelation and current balance
AC_STEPS = 12000


def task_autocorr(args):
    variant, L, rho, s = args
    p = params(variant, L).with_rho(rho)
    lat = Lattice(p, seed_of(3, list(VARIANTS).index(variant), L, int(rho * 10), s))
    U, F, K, R = [], [], [], []
    for _ in range(AC_STEPS):
        k, r = lat.step()
        U.append(lat.order())
        F.append(lat.intact_fraction())
        K.append(k / L / L)
        R.append(r / L / L)
    return variant, L, rho, s, U, F, K, R


def run_autocorr():
    sizes = [16, 32] if QUICK else [16, 32, 64]
    seeds = 2 if QUICK else 4
    steps = AC_STEPS
    tasks = [(v, L, r, s) for v in ("default",) for L in sizes for r in (4.5, 6.0, 8.0) for s in range(seeds)]
    tasks.sort(key=lambda t: -t[1])
    out = {}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for v, L, r, s, U, F, K, R in pool.imap_unordered(task_autocorr, tasks):
            out.setdefault("%s|%d|%.1f" % (v, L, r), []).append({"U": U, "intact": F, "kill": K, "repair": R})
    json.dump({"steps": steps, "sizes": sizes, "seeds": seeds, "runs": out}, open(os.path.join(RES, "autocorr_raw.json"), "w"))
    print("autocorr done in %.0f s" % (time.time() - t0))


# ======================================================================= D. finite-size scaling, quenched dilution
FSS_T = 0.45
FSS_F = [0.10, 0.14, 0.17, 0.20, 0.23, 0.26, 0.30, 0.35]
FSS_EQ, FSS_MEAS, FSS_SAMPLE = 2000, 4000, 4


def task_fss(args):
    L, f, s = args
    q = Quenched(L, FSS_T, f, seed_of(4, L, int(round(f * 1000)), s))
    for _ in range(FSS_EQ):
        q.sweep()
    Y, M = [], []
    for t in range(FSS_MEAS):
        q.sweep()
        if t % FSS_SAMPLE == 0:
            Y.append(q.helicity())
            M.append(q.magnetization())
    return L, f, s, Y, M


def run_fss():
    sizes = [16, 24, 32] if QUICK else [16, 24, 32, 48, 64, 96]
    seeds = 3 if QUICK else 8
    tasks = [(L, f, s) for L in sizes for f in FSS_F for s in range(seeds)]
    tasks.sort(key=lambda t: -t[0])
    out = {}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for i, (L, f, s, Y, M) in enumerate(pool.imap_unordered(task_fss, tasks)):
            out.setdefault(str(L), {}).setdefault("%.2f" % f, []).append({"seed": s, "Y": Y, "M": M})
            if (i + 1) % 50 == 0:
                print("  fss %d/%d  (%.0f s)" % (i + 1, len(tasks), time.time() - t0), flush=True)
    json.dump({"T": FSS_T, "f": FSS_F, "eq": FSS_EQ, "meas": FSS_MEAS, "sample_every": FSS_SAMPLE, "sizes": sizes,
               "seeds": seeds, "runs": out}, open(os.path.join(RES, "fss_raw.json"), "w"))
    print("fss done in %.0f s" % (time.time() - t0))


# ======================================================================= E. closure test
CL_RHOS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0]
CL_EQ, CL_MEAS = 800, 1200


def task_closure(args):
    variant, L, rho, s, start = args
    p = params(variant, L).with_rho(rho)
    lat = Lattice(p, seed_of(5, list(VARIANTS).index(variant), L, int(rho * 10), s, start))
    if start == 1:
        lat.set_collapsed(0.95, random_angles=True)
    for _ in range(CL_EQ):
        lat.step()
    I, Dm, Mu, Fi = [], [], [], []
    for t in range(CL_MEAS):
        lat.step()
        if t % 4 == 0:
            D = lat.local_disorder()
            m = lat.intact
            Dm.append(float(D[m].mean()) if m.any() else float("nan"))
            Mu.append(float((1.0 + p.beta * D[m] ** p.hill_h / (p.hill_K ** p.hill_h + D[m] ** p.hill_h)).mean()) if m.any() else float("nan"))
            I.append(1.0 - lat.order())
            Fi.append(lat.intact_fraction())
    return variant, L, rho, s, start, float(np.mean(I)), float(np.nanmean(Dm)), float(np.nanmean(Mu)), float(np.mean(Fi))


def run_closure():
    L = 32 if QUICK else 64
    seeds = 2 if QUICK else 4
    tasks = [(v, L, r, s, st) for v in VARIANTS for r in CL_RHOS for s in range(seeds) for st in (0, 1)]
    out = []
    t0 = time.time()
    with Pool(NPROC) as pool:
        for v, LL, r, s, st, I, Dm, Mu, Fi in pool.imap_unordered(task_closure, tasks):
            out.append({"variant": v, "L": LL, "rho": r, "seed": s, "start": "ordered" if st == 0 else "collapsed",
                        "I": I, "D_intact_mean": Dm, "mu_lattice": Mu, "intact_fraction": Fi})
    json.dump({"L": L, "seeds": seeds, "eq": CL_EQ, "meas": CL_MEAS, "rhos": CL_RHOS, "variants": VARIANTS, "base": BASE,
               "runs": out}, open(os.path.join(RES, "closure_raw.json"), "w"))
    print("closure done in %.0f s" % (time.time() - t0))


# ======================================================================= F. sensing range
SENS = ["r1", "r2", "r4", "global"]
SENS_RHOS = [3.0, 4.5, 6.0, 8.0, 10.0, 12.0]
SENS_DWELLS = [64, 256]
SENS_STEPS = 2000


def task_sensing(args):
    kind, sens, L, x, s = args
    p = params("default", L).replace(sensing=sens)
    code = (6, SENS.index(sens), L, int(x * 10), s)
    if kind == "sweep":
        p0 = p.with_rho(float(RHO_GRID[0]))
        lat = Lattice(p0, seed_of(*code, 0))
        for _ in range(200):
            lat.step()
        down, up = [], []
        for r in RHO_GRID:
            lat.p = p0.with_rho(float(r))
            for _ in range(int(x)):
                lat.step()
            down.append(lat.order())
        for r in RHO_GRID[::-1]:
            lat.p = p0.with_rho(float(r))
            for _ in range(int(x)):
                lat.step()
            up.append(lat.order())
        return kind, sens, L, x, s, {"down": down, "up": up[::-1]}
    res = {}
    for start in (0, 1):
        lat = Lattice(p.with_rho(x), seed_of(*code, start + 1))
        if start == 1:
            lat.set_collapsed(0.95, random_angles=True)
        tr = []
        for t in range(SENS_STEPS):
            lat.step()
            if t % 5 == 0:
                tr.append(lat.order())
        res["ordered" if start == 0 else "collapsed"] = tr
    return kind, sens, L, x, s, res


def run_sensing():
    L = 32 if QUICK else 48
    seeds = 2 if QUICK else 4
    sens = SENS                                     # r1 is the block of eight (nearest and diagonal neighbours)
    tasks = [("sweep", sv, L, float(d), s) for sv in sens for d in SENS_DWELLS for s in range(seeds)]
    tasks += [("two", sv, L, r, s) for sv in sens for r in SENS_RHOS for s in range(seeds)]
    tasks.sort(key=lambda t: -(t[3] * 62 if t[0] == "sweep" else 2 * SENS_STEPS))
    out = {"sweep": {}, "two": {}}
    t0 = time.time()
    with Pool(NPROC) as pool:
        for i, (kind, sv, LL, x, s, res) in enumerate(pool.imap_unordered(task_sensing, tasks)):
            out[kind].setdefault(sv, {}).setdefault(repr(x), []).append(dict(seed=s, **res))
            if (i + 1) % 25 == 0:
                print("  sensing %d/%d  (%.0f s)" % (i + 1, len(tasks), time.time() - t0), flush=True)
    json.dump({"L": L, "seeds": seeds, "sensing": sens, "rho_grid": RHO_GRID.tolist(), "dwells": SENS_DWELLS,
               "rhos": SENS_RHOS, "steps": SENS_STEPS, "save_every": 5, "base": BASE, "runs": out},
              open(os.path.join(RES, "sensing_raw.json"), "w"))
    print("sensing done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    what = [a for a in sys.argv[1:] if not a.startswith("--")] or ["all"]
    jobs = {"hysteresis": run_hysteresis, "branches": run_branches, "autocorr": run_autocorr, "fss": run_fss,
            "closure": run_closure, "sensing": run_sensing}
    for w in (list(jobs) if what == ["all"] else what):
        print("== %s (%d workers%s)" % (w, NPROC, ", quick" if QUICK else ""), flush=True)
        jobs[w]()
