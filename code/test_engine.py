# -*- coding: utf-8 -*-
"""Checks on the engine before any result is computed with it.

  1. Same seed, same parameters: lattice_engine reproduces the MyUncle engine (the code behind the
     manuscript's earlier figures) bit for bit, so results here are continuous with the published ones.
  2. beta = 0: the intact fraction relaxes to the two-state chain. The update is synchronous in discrete
     time (kill, then repair, each step), so the exact fixed point is R/(lam + R - lam R), which tends to the
     continuous-time value R/(lam + R) = rho/(1 + rho) as lam and R go to zero.
  3. Repair at R = 0 and kill at lam = 0 do nothing they should not.
  4. The helicity modulus of a perfectly ordered, undiluted lattice equals 1 (J = 1, no currents).
  5. The sensing options: block sensing of radius r averages exactly the (2r+1)^2 - 1 surrounding rotors, and
     global sensing hands every site the global disorder 1 - U.

    python test_engine.py          (MyUncle is looked for at ../../_repos/MyUncle or on PYTHONPATH)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lattice_engine import Lattice, Params, Quenched  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# 1. bit-for-bit agreement with MyUncle
for cand in (os.path.join(HERE, "..", "..", "_repos", "MyUncle"), os.path.join(HERE, "..", "..", "myuncle-repo")):
    if os.path.exists(os.path.join(cand, "MyUncle_core.py")):
        sys.path.insert(0, os.path.abspath(cand))
        break
try:
    from MyUncle_core import MC, Config
    have_myuncle = True
except ImportError:
    have_myuncle = False
if have_myuncle:
    for h0, rho in ((0.30, 6.0), (0.0, 4.5)):
        cfg = Config(L=24, seed=7, T=0.45, h0=h0, g0=0.12, lam0=0.02, beta=20.0, hill_h=6.0, hill_K=0.5)
        cfg = cfg.copy(R=rho * cfg.lam_eff)
        ref = MC(cfg)
        p = Params(L=24, T=0.45, h0=h0, g0=0.12, lam0=0.02, beta=20.0, hill_h=6.0, hill_K=0.5).with_rho(rho)
        new = Lattice(p, seed=7)
        for _ in range(60):
            a = ref.step()
            b = new.step()
        same = np.array_equal(ref.th, new.th) and np.array_equal(ref.g, new.g) and a == b
        check("reproduces MyUncle exactly after 60 steps (h0=%.2f, rho=%.1f)" % (h0, rho), same)
else:
    print("  [SKIP] MyUncle_core not found; bit-for-bit comparison not run")

# 2. beta = 0: two-state chain
p = Params(L=48, beta=0.0, T=0.45).with_rho(3.0)
lat = Lattice(p, seed=11)
fr = []
for t in range(1500):
    lat.step()
    if t >= 500:
        fr.append(lat.intact_fraction())
lam = p.lam_eff
target = p.R / (lam + p.R - lam * p.R)
se = np.std(fr) / np.sqrt(len(fr) / 50.0)        # conservative: ~50-step correlation
check("beta = 0: intact fraction = R/(lam + R - lam R) of the discrete two-state chain",
      abs(np.mean(fr) - target) < max(4 * se, 0.003), "%.4f vs %.4f (continuous-time %.4f)" % (np.mean(fr), target, p.R / (p.R + lam)))

# 3. trivial rates
p = Params(L=16).with_rho(0.0)
lat = Lattice(p, seed=3)
lat.set_collapsed(0.5)
n_before = int((~lat.intact).sum())
for _ in range(20):
    k, r = lat.step()
    assert r == 0
check("R = 0: no site is ever repaired", int((~lat.intact).sum()) >= n_before)
p = Params(L=16, lam0=0.0, R=0.5)
lat = Lattice(p, seed=3)
ks = [lat.step()[0] for _ in range(20)]
check("lam = 0: no site is ever killed", sum(ks) == 0)

# 4. helicity of the ordered clean lattice
q = Quenched(L=16, T=0.45, f=0.0, seed=1)
q.th[:] = 0.0
check("helicity modulus of the ordered clean lattice = 1", abs(q.helicity() - 1.0) < 1e-12, "%.6f" % q.helicity())

# 5. sensing options
rng = np.random.default_rng(4)
lat = Lattice(Params(L=12, sensing="r2"), seed=1)
lat.th = rng.uniform(-np.pi, np.pi, (12, 12))
z = np.exp(1j * lat.th)
i0, j0 = 5, 7
blk = [z[(i0 + a) % 12, (j0 + b) % 12] for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]
check("block sensing r = 2 averages the 24 surrounding rotors", abs(lat.local_disorder()[i0, j0] - (1 - abs(np.mean(blk)))) < 1e-12)
lat.p = lat.p.replace(sensing="global")
check("global sensing gives every site 1 - U", np.allclose(lat.local_disorder(), 1 - lat.order()))
lat.p = lat.p.replace(sensing="nn")
nn = [z[(i0 + 1) % 12, j0], z[i0 - 1, j0], z[i0, (j0 + 1) % 12], z[i0, j0 - 1]]
check("nearest-neighbour sensing averages the 4 neighbours", abs(lat.local_disorder()[i0, j0] - (1 - abs(np.mean(nn)))) < 1e-12)

print("\n%s" % ("ALL CHECKS PASSED" if not FAILS else "%d FAILED: %s" % (len(FAILS), ", ".join(FAILS))))
sys.exit(1 if FAILS else 0)
