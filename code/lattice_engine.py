# -*- coding: utf-8 -*-
"""Kinetic Monte Carlo engine for the maintained rotational lattice.

The update rules are those of the MyUncle framework (github.com/sandlerleon/MyUncle,
doi:10.5281/zenodo.21223569), from which the manuscript's earlier figures were produced; this file
re-states them so that the study is self-contained, and test_engine.py checks that it reproduces the
MyUncle trajectories exactly for the same seed.

One step = `relax` checkerboard Metropolis sweeps of the rotors at fixed integrity, then one
synchronous site update:
  kill   an intact site is diluted (g -> g0) with probability lam_i = lam_eff [1 + beta D_i^h/(K^h + D_i^h)],
         D_i = 1 - |(1/4) sum_{j in nn(i)} exp(i theta_j)|   (local disorder)
  repair a diluted site is restored (g -> 1, theta -> 0) with probability R
rho = R / lam_eff is the single control parameter of the mean-field reduction.

Energy: H = -J0 sum_<ij> g_i g_j cos(theta_i - theta_j) - h0 sum_i g_i cos(theta_i), k_B = 1.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Params:
    L: int = 48
    T: float = 0.45
    h0: float = 0.30
    g0: float = 0.12
    lam0: float = 0.02
    beta: float = 20.0
    hill_h: float = 6.0
    hill_K: float = 0.5
    k: float = 1.0
    G: float = 1.0
    s: float = 0.75
    u_s: float = 0.0
    R: float = 0.10
    relax: int = 2
    theta_init_sigma: float = 0.10
    sensing: str = "nn"     # what a site's kill rate responds to: "nn" its four nearest neighbours (the model);
                            # "rK" the (2K+1)^2 - 1 sites of the surrounding square block; "global" the global
                            # disorder 1 - U, which imposes the mean-field closure on the lattice itself

    @property
    def lam_eff(self):
        return self.lam0 * self.k * self.G * (1.0 - self.s * self.u_s)

    @property
    def rho(self):
        return self.R / self.lam_eff

    def with_rho(self, rho):
        d = asdict(self)
        d["R"] = rho * self.lam_eff
        return Params(**d)

    def replace(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Params(**d)


class Lattice:
    def __init__(self, p: Params, seed: int):
        self.p = p
        self.rng = np.random.default_rng(seed)
        L = p.L
        self.parity = np.add.outer(np.arange(L), np.arange(L)) % 2
        self.th = self.rng.normal(0.0, p.theta_init_sigma, (L, L))
        self.g = np.full((L, L), 1.0)
        self.intact = np.ones((L, L), bool)

    # ------------------------------------------------------------ initial states
    def set_collapsed(self, killed_fraction=0.95, random_angles=True):
        L = self.p.L
        v = self.rng.random((L, L)) < killed_fraction
        self.g[v] = self.p.g0
        self.intact[v] = False
        if random_angles:
            self.th = self.rng.uniform(-np.pi, np.pi, (L, L))
        return self

    # ------------------------------------------------------------ dynamics
    def metropolis(self):
        p, th, g = self.p, self.th, self.g
        L = p.L
        for col in (0, 1):
            m = self.parity == col
            prop = th + self.rng.uniform(-np.pi, np.pi, (L, L))
            e0 = np.zeros((L, L))
            e1 = np.zeros((L, L))
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                J = g * np.roll(g, sh, ax)
                tj = np.roll(th, sh, ax)
                e0 -= J * np.cos(th - tj)
                e1 -= J * np.cos(prop - tj)
            e0 -= p.h0 * g * np.cos(th)
            e1 -= p.h0 * g * np.cos(prop)
            acc = (self.rng.random((L, L)) < np.exp(-(e1 - e0) / p.T)) & m
            th = np.where(acc, prop, th)
        self.th = th

    def local_disorder(self):
        z = np.exp(1j * self.th)
        sens = self.p.sensing
        if sens == "nn":
            nb = (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 4.0
            return 1.0 - np.abs(nb)
        if sens == "global":
            return np.full(z.shape, 1.0 - abs(z.mean()))
        r = int(sens[1:])
        row = sum(np.roll(z, d, 0) for d in range(-r, r + 1))
        block = sum(np.roll(row, d, 1) for d in range(-r, r + 1))
        return 1.0 - np.abs((block - z) / ((2 * r + 1) ** 2 - 1))

    def site_update(self):
        p = self.p
        D = self.local_disorder()
        lam = p.lam_eff * (1.0 + p.beta * D ** p.hill_h / (p.hill_K ** p.hill_h + D ** p.hill_h))
        kill = self.intact & (self.rng.random(D.shape) < lam)
        self.intact &= ~kill
        self.g[kill] = p.g0
        rep = (~self.intact) & (self.rng.random(D.shape) < p.R)
        self.intact |= rep
        self.g[rep] = 1.0
        self.th[rep] = 0.0
        return int(kill.sum()), int(rep.sum())

    def step(self):
        for _ in range(self.p.relax):
            self.metropolis()
        return self.site_update()

    # ------------------------------------------------------------ observables
    def order(self):
        return float(abs(np.mean(np.exp(1j * self.th))))

    def intact_fraction(self):
        return float(self.intact.mean())


# ---------------------------------------------------------------- quenched dilution (Section 5)
class Quenched:
    """Field-free XY model on a quenched site-diluted lattice (killed sites carry g = 0)."""

    def __init__(self, L, T, f, seed):
        self.L, self.T = L, T
        self.rng = np.random.default_rng(seed)
        self.parity = np.add.outer(np.arange(L), np.arange(L)) % 2
        self.g = np.ones((L, L))
        self.g[self.rng.random((L, L)) < f] = 0.0
        self.th = self.rng.normal(0.0, 0.1, (L, L))

    def sweep(self):
        L, th, g = self.L, self.th, self.g
        for col in (0, 1):
            m = self.parity == col
            prop = th + self.rng.uniform(-np.pi, np.pi, (L, L))
            e0 = np.zeros((L, L))
            e1 = np.zeros((L, L))
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                J = g * np.roll(g, sh, ax)
                tj = np.roll(th, sh, ax)
                e0 -= J * np.cos(th - tj)
                e1 -= J * np.cos(prop - tj)
            acc = (self.rng.random((L, L)) < np.exp(-(e1 - e0) / self.T)) & m
            th = np.where(acc, prop, th)
        self.th = th

    def magnetization(self):
        n = max(self.g.sum(), 1.0)
        return float(abs(np.sum(self.g * np.exp(1j * self.th))) / n)

    def helicity(self):
        """Helicity modulus, averaged over the two lattice directions."""
        L, th, g, T = self.L, self.th, self.g, self.T
        ups = []
        for ax in (0, 1):
            J = g * np.roll(g, -1, ax)
            d = th - np.roll(th, -1, ax)
            ups.append(np.sum(J * np.cos(d)) / (L * L) - np.sum(J * np.sin(d)) ** 2 / (L * L * T))
        return 0.5 * (ups[0] + ups[1])
