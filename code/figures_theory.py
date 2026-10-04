# -*- coding: utf-8 -*-
"""Figures 1, 2, 3 and 7 of the manuscript, regenerated from this repository so that every figure can be reproduced
from the archive (earlier versions carried them as images made with the MyUncle framework).

  Fig_bridge     Figure 1   order-entropy bridge of the von Mises law (Theorem 1), analytic
  Fig_baseline   Figure 2   the beta = 0 baseline: lattice against the exact discrete two-state chain (seeded runs here)
  Fig_meanfield  Figure 3   S-shaped steady-state manifold, bifurcation diagram and two trajectories of the reduction
  Fig_entropy    Figure 7   measured stationary currents and the entropy production of the auxiliary cycle of Section 6
                            (needs results/currents_raw.json from `python mc_study.py currents`)

    python figures_theory.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import i0e, i1e

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lattice_engine import Lattice, Params  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.fontsize": 7})


def tag(ax, t):
    ax.text(-0.15, 1.06, t, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1: order-entropy bridge
k = np.geomspace(1e-4, 2e3, 6000)
A = i1e(k) / i0e(k)                                   # U_chi = A(kappa)
S = np.log(2 * np.pi * i0e(k)) + k - k * A            # ln(2 pi I0) - kappa A, with I0 = i0e e^kappa
SMAX = np.log(2 * np.pi)
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
fig.subplots_adjust(wspace=0.3)
ax = axs[0]
ax.plot(A, S, color="#c0392b", lw=1.8)
ax.axhline(SMAX, color="k", ls="--", lw=0.8)
ax.text(0.02, SMAX - 0.12, r"$S_{max} = \ln 2\pi$", fontsize=7.5)
ax.set_xlim(0, 1)
ax.set_xlabel(r"directional order $U_\chi = A(\kappa)$")
ax.set_ylabel(r"entropy $S$ (nats)")
ax.set_title(r"$S$ strictly decreasing in $U_\chi$; $dS/dU_\chi = -\kappa$", loc="left")
tag(ax, "A")
ax = axs[1]
Hx = SMAX - S
ax.plot(A, Hx, color="#1a7a4c", lw=1.8, label=r"exact $H_\chi = S_{max} - S$")
uu = np.linspace(0.55, 0.999, 200)
ax.plot(uu, -0.5 * np.log(1 - uu) + 0.5 * (np.log(np.pi) - 1), "--", color="#e08e0b", lw=1.3,
        label=r"tail $-\frac{1}{2}\ln(1-U_\chi) + \frac{1}{2}(\ln\pi - 1)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 3)
ax.set_xlabel(r"directional order $U_\chi$")
ax.set_ylabel(r"information $H_\chi$ (nats)")
ax.set_title("Invertible bridge and its logarithmic tail", loc="left")
ax.legend(frameon=False, loc="upper left")
tag(ax, "B")
save(fig, "Fig_bridge.png")

# ------------------------------------------------------------------ Figure 2: beta = 0 baseline
BASE0 = dict(L=48, T=0.45, h0=0.30, beta=0.0)
lam = Params(**BASE0).lam_eff


def chain(p0, R, n):
    """Exact mean of the synchronous two-state chain: kill with prob lam, then repair (incl. just-killed) with prob R."""
    out = [p0]
    for _ in range(n):
        p = out[-1]
        out.append(p * (1 - lam) + (1 - p * (1 - lam)) * R)
    return np.array(out)


fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
fig.subplots_adjust(wspace=0.32)
ax = axs[0]
nstep = 400
for rho, c in ((0.5, "#c0392b"), (3.0, "#08519c")):
    p = Params(**BASE0).with_rho(rho)
    for s in range(3):
        lat = Lattice(p, seed=1000 + s + int(rho * 10))
        tr = [lat.intact_fraction()]
        for _ in range(nstep):
            lat.step()
            tr.append(lat.intact_fraction())
        ax.plot(np.arange(nstep + 1), tr, color=c, lw=0.9, alpha=0.55, label=r"lattice, $\rho$ = %.1f" % rho if s == 0 else None)
    ax.plot(np.arange(nstep + 1), chain(1.0, p.R, nstep), "k--", lw=0.9, label="exact two-state chain" if rho == 3.0 else None)
ax.set_ylim(0, 1.02)
ax.set_xlabel("steps")
ax.set_ylabel("intact fraction p")
ax.set_title(r"$\beta$ = 0: lattice against the exact chain, L = 48", loc="left")
ax.legend(frameon=False)
tag(ax, "A")
ax = axs[1]
rhos = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
Ps, Us = [], []
for rho in rhos:
    p = Params(**BASE0).with_rho(rho)
    lat = Lattice(p, seed=2000 + int(rho * 100))
    for _ in range(300):
        lat.step()
    pp, uu_ = [], []
    for _ in range(600):
        lat.step()
        pp.append(lat.intact_fraction())
        uu_.append(lat.order())
    Ps.append(np.mean(pp))
    Us.append(np.mean(uu_))
rr = np.geomspace(0.1, 20, 200)
ax.plot(rr, rr / (1 + rr), "k-", lw=1.2, label=r"$\rho/(1+\rho)$ (continuous time)")
ax.plot(rr, rr * lam / (lam + rr * lam - lam * rr * lam), ":", color="#555", lw=1.2, label=r"$R/(\lambda_0 + R - \lambda_0 R)$ (discrete)")
ax.plot(rhos, Ps, "o", color="#08519c", ms=5, label="lattice intact fraction p")
ax.plot(rhos, Us, "s", color="#e08e0b", ms=5, mfc="white", label=r"lattice order $U_\chi$")
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\rho = R/\lambda_0$")
ax.set_ylabel("stationary value")
ax.set_title(r"NESS: $U_\chi$ is slaved to p", loc="left")
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
tag(ax, "B")
ax = axs[2]
for rho, c in ((0.5, "#c0392b"), (3.0, "#08519c")):
    p = Params(**BASE0).with_rho(rho)
    ex = chain(1.0, p.R, nstep)
    pstar = p.R / (lam + p.R - lam * p.R)
    tau = lam * np.arange(nstep + 1)
    ax.plot(tau, np.abs(ex - pstar), color=c, lw=1.4, label=r"$\rho$ = %.1f" % rho)
    ax.plot(tau, (1 - pstar) * np.exp(-(1 + rho) * tau), "--", color=c, lw=0.9)
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlabel(r"$\tau = \lambda_0 t$")
ax.set_ylabel(r"$|p - p^*|$")
ax.set_title(r"Relaxation at rate $1 + \rho$ (dashed)", loc="left")
ax.legend(frameon=False)
tag(ax, "C")
save(fig, "Fig_baseline.png")

# ------------------------------------------------------------------ Figure 3: the mean-field reduction
h, K, beta = 6.0, 0.5, 20.0
I = np.linspace(1e-4, 1 - 1e-4, 20000)
mu = 1 + beta * I ** h / (K ** h + I ** h)
F = mu * (1 - I) / I
dmu = beta * h * K ** h * I ** (h - 1) / (K ** h + I ** h) ** 2
stab = (dmu * (1 - I) - mu - F) < 0
i_lo = np.argmin(np.where((I > 0.1) & (I < 0.4), F, np.inf))
i_hi = np.argmax(np.where((I > 0.4) & (I < 0.8), F, -np.inf))
rm, rp = F[i_lo], F[i_hi]
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
fig.subplots_adjust(wspace=0.32)
ax = axs[0]
ax.plot(F, I, color="#6a3d9a", lw=1.8)
ax.axvline(7.5, color="#c0392b", ls="--", lw=0.9)
ax.text(7.7, 0.92, "three fixed points", color="#c0392b", fontsize=7)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("information I")
ax.set_title(r"S-shaped manifold $\rho = F(I)$", loc="left")
tag(ax, "A")
ax = axs[1]
ax.axvspan(rm, rp, color="#fdf1c9", zorder=0)
ax.plot(F[stab], I[stab], ".", color="#1a7a4c", ms=1.2)
ax.plot(F[~stab], I[~stab], ".", color="#c0392b", ms=1.2)
ax.text(0.5 * (rm + rp), 0.47, "window\n%.2f < ρ < %.2f" % (rm, rp), ha="center", fontsize=7)
ax.set_xlim(0, 15)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"steady $I^*$")
ax.set_title("Saddle-node bifurcations: stable (green), unstable (red)", loc="left")
tag(ax, "B")
ax = axs[2]
rho0 = 7.5
rhs = lambda t, y: (1 + beta * y ** h / (K ** h + y ** h)) * (1 - y) - rho0 * y
for I0, c in ((0.15, "#1a7a4c"), (0.60, "#c0392b")):
    sol = solve_ivp(rhs, (0, 4), [I0], dense_output=True, rtol=1e-9)
    tt = np.linspace(0, 4, 400)
    ax.plot(tt, 1 - sol.sol(tt)[0], color=c, lw=1.6, label=r"$U_\chi(0)$ = %.2f" % (1 - I0))
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$U_\chi = 1 - I$")
ax.set_title(r"Two NESS at $\rho$ = 7.5", loc="left")
ax.legend(frameon=False)
tag(ax, "C")
save(fig, "Fig_meanfield.png")

# ------------------------------------------------------------------ Figure 7: currents and the auxiliary cycle
path = os.path.join(RES, "currents_raw.json")
if os.path.exists(path):
    C = json.load(open(path))
    lam_e = C["lam_eff"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    fig.subplots_adjust(wspace=0.32)
    summary = {}
    for v, col, lab in (("default", "#08519c", "T = 0.45, $h_0$ = 0.30"), ("cold0", "#c0392b", "T = 0.30, $h_0$ = 0")):
        rr = sorted(C["rhos"])
        J = np.array([np.mean([x["J_kill"] for x in C["runs"] if x["variant"] == v and x["rho"] == r]) for r in rr])
        Jr = np.array([np.mean([x["J_repair"] for x in C["runs"] if x["variant"] == v and x["rho"] == r]) for r in rr])
        pb = np.array([np.mean([x["intact_before"] for x in C["runs"] if x["variant"] == v and x["rho"] == r]) for r in rr])
        kbar = J / pb
        Rr = np.array(rr) * lam_e
        summary[v] = {"rho": rr, "J_kill": J.tolist(), "J_repair": Jr.tolist(), "kbar_over_lam": (kbar / lam_e).tolist()}
        axs[1].plot(rr, J, "o-", color=col, ms=4, lw=1.1, label="lattice, " + lab)
        for eps, ls in ((1e-4, "-"), (1e-3, "--"), (1e-2, ":")):
            sig = J * np.log(kbar * Rr / eps ** 2)
            axs[0].plot(rr, sig, ls, color=col, lw=1.2, label=(lab + r", $\epsilon$ = 10$^{%d}$" % round(np.log10(eps))))
    rr = np.geomspace(0.5, 16, 100)
    R0 = rr * lam_e
    axs[1].plot(rr, lam_e * R0 / (lam_e + R0 - lam_e * R0), "k--", lw=1, label=r"uncoupled chain ($\beta$ = 0)")
    axs[1].axhline(lam_e, color="#888", lw=0.7, ls=":")
    axs[1].text(16, lam_e - 0.004, r"$\lambda_0$", ha="right", fontsize=7, color="#666")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\rho$")
    axs[0].set_ylabel(r"$\sigma_{aux}$ (nats per site per step)")
    axs[0].set_title(r"Auxiliary cycle: $\sigma_{aux} = J\,\ln(\bar{k}R/\epsilon^2)$", loc="left")
    axs[0].legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axs[1].set_ylabel("stationary current J per site per step")
    axs[1].set_title("Measured maintenance current", loc="left")
    axs[1].legend(frameon=False, fontsize=6.3)
    tag(axs[0], "A")
    tag(axs[1], "B")
    save(fig, "Fig_entropy.png")
    json.dump(summary, open(os.path.join(RES, "currents_summary.json"), "w"), indent=1)
print("theory figures written")
