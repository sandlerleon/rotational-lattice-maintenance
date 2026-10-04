# Rotational-lattice maintenance: does mean-field bistability survive on the lattice?

Leon Sandler, Independent Researcher — sandler.leon@gmail.com
ORCID [0009-0007-4584-808X](https://orcid.org/0009-0007-4584-808X)

This repository holds the kinetic Monte Carlo study behind versions 9 and 10 of *"Self-Maintained Order and
Hysteretic Collapse in a Non-Equilibrium Rotational Lattice"* (submitted to **Physica A**; preprint
[10.5281/zenodo.21210708](https://doi.org/10.5281/zenodo.21210708)). Every number and every figure of the
manuscript, including the analytic Figures 1 and 3, is produced by the scripts here, from the raw run outputs
that are also here.

## The question

The model is a 2D XY lattice whose sites are killed (couplings diluted) at a rate that rises with local
orientational disorder and repaired at rate R. Its annealed mean-field reduction,

```
dI/dτ = μ(I)(1 − I) − ρI,     μ(I) = 1 + β I^h / (K^h + I^h),     ρ = R/λ
```

is bistable for h > 1 and β above a threshold β_c(h, K). With (h, K, β) = (6, 0.5, 20) the window is
3.92 < ρ < 11.30. The reduction rests on one approximation: each site's *local* disorder δ_i is replaced by
the *global* I = 1 − U. Earlier versions showed a hysteresis loop on the lattice at one sweep rate and
called it fluctuation-rounded bistability. A loop at a finite sweep rate does not show bistability: any
system lags behind a moving control parameter.

## What the Monte Carlo shows

| Test | Nearest-neighbour lattice (the model) |
|---|---|
| Loop area vs sweep rate, L = 16–64, 9 rates, 8 runs | peaks at 2.5 for fast sweeps; **zero within error** for ≥ 64 steps per ρ value at every size, in both parameter sets (mean-field static area: 4.30) |
| Ordered vs collapsed start at ρ = 4.5, 6, 8, 10 | stationary order agrees to within **0.004** in all 40 cases |
| Time for the two starts to meet | 15–115 steps, **independent of N** (256 to 4096 sites) |
| Integrated autocorrelation time of U | 6–11 steps, independent of L |
| Current balance J_kill = J_rep | to better than 1 part in 10³ |

The lattice has a single steady state at each ρ, and its hysteresis is purely dynamic.

**Why.** In the stationary lattice the effective kill multiplier is ⟨μ(δ_i)⟩, not μ(I). Where order is
low, intact sites see much less disorder than the global I (at ρ = 2, δ̄ = 0.45 against I = 0.79). Four
random rotors only give δ ≈ 0.55, barely above K, so the switch never fully engages. Where order is high,
the few sites in disordered pockets are killed at nearly full rate (at ρ = 6, ⟨μ(δ)⟩ = 2.47 against
μ(I) = 1.37). Averaging a steep switch over a broad distribution flattens it, and the S-shape is lost.

**The control.** The same lattice, with the kill rate sensing a block of n sites:

| sites sensed | 4 (model) | 8 | 24 | 80 | global (= the mean-field closure) |
|---|---|---|---|---|---|
| slow-sweep loop area (256 steps/ρ) | 0.01 | 0.03 | 0.21 | 1.00 | 1.93 |
| ordered vs collapsed start | agree | agree | agree | split at ρ = 6 | split at ρ = 4.5 and 6 |

Bistability is real, but it needs the degrading flux to sense order over an extended region. The range
of the feedback is a control parameter.

**In size and time (v1.1.0).** Two starts were run at L = 32, 48, 64 and 96, 4 runs each, for 15 000 steps:
- **Global sensing** (ρ = 4.0, 4.5, 6.0, 7.0): none of the 128 runs left its branch, and the separation (0.43–0.54) is independent of L. This is the behaviour of a genuine mean-field bistability.
- **80-site sensing, ρ = 5.5 and 6.0:** the branches persist at every size.
- **80-site sensing, window edges:** at ρ = 5.0 the ordered state decays in 12 of 16 runs, and sooner at larger L (7 400–11 500 steps at L = 32; 1 200–5 100 at L = 96). That is nucleation. At ρ = 6.5 the collapsed state recovers in 13 of 16 runs.

The finite-range case is therefore a discontinuous transition with long-lived metastability. Whether an interval of true coexistence survives N → ∞ is open (`results/extended_raw.json`, Figure A2).

**Maintenance current and entropy production (Section 6).** The simulated dynamics is irreversible, so it
has no finite Schnakenberg entropy production of its own. What is measured is the stationary current J: it
vanishes as ρ → 0, peaks at about 0.052 per site and step near ρ = 4, where order is most fragile, and falls to
λ₀ = 0.02 in the well-maintained regime. The entropy production in Figure 7A belongs to an auxiliary two-state
cycle with added reverse rates ε. It is illustrative and depends on ε.

**Field-free transition (Section 5).** Quenched dilution, T = 0.45, L = 16–96, 8 disorder realizations:
the helicity modulus crosses 2T/π at f_KT(L) = 0.220 → 0.210. The 1/ln²L extrapolation gives
**0.205 (0.193–0.216)**, and a Weber–Minnhagen fit gives **0.205 (0.198–0.213)**, χ² = 0.75 / 4 dof. Both
are far below site percolation (0.407), so the mechanism is BKT. Dropping the smaller lattices moves the
estimate by less than 0.006 (all variants lie between 0.200 and 0.209), so 0.205 is a numerical estimate at
L ≤ 96, not a precise critical value.

**The two-start exceptions.** In 5 of the 40 two-start comparisons the bootstrap interval of the difference
excludes zero. All 5 differ by at most 0.0021, with mixed signs, and none survives a Holm correction
(smallest adjusted p = 0.70). They are listed in `results/mc_results.json` under `branches_tests`.

## Layout

```
code/lattice_engine.py   the engine: Metropolis rotor sweeps + synchronous kill/repair; sensing = nn | rK | global
code/test_engine.py      9 checks, incl. bit-for-bit agreement with MyUncle (run before any result)
code/mc_study.py         eight studies: hysteresis, branches, autocorr, fss, sensing, closure, extended, currents
code/figures_theory.py   Figures 1, 2, 3 and 7 (analytic, the beta = 0 baseline, and the currents of Section 6)
code/analyze_study.py    raw runs -> results/mc_results.json and figures/Fig_*.png
code/mf_threshold.py     beta_c(h, K) of the mean-field reduction -> results/mf_threshold.json
results/*_raw.json       raw outputs of every run
manuscript/              manuscript v9, highlights, graphical abstract
tools/                   Zenodo deposit scripts (token from ZENODO_TOKEN, never stored)
```

Figure map: `Fig_bridge` = Figure 1, `Fig_baseline` = Figure 2, `Fig_meanfield` = Figure 3,
`Fig_lattice` = Figure 4, `Fig_mechanism` = Figure 5, `Fig_fss` = Figure 6, `Fig_entropy` = Figure 7,
`Fig_diagnostics` = Figure A1, `Fig_extended` = Figure A2.

## Reproducing

```bash
pip install -r requirements.txt
cd code
python test_engine.py                       # MyUncle comparison runs if MyUncle_core is importable
python mc_study.py all                      # about 4 h on 7 worker processes; --quick for a smoke test
python analyze_study.py
python figures_theory.py
python mf_threshold.py
```

Each run is seeded from `SeedSequence([20261003, study, variant, L, control, replicate, start])`, so any
single run can be repeated alone. Intervals are 95% percentile bootstrap intervals over independent runs
or disorder realizations (B = 2000). Autocorrelation times use Sokal's automatic window (W ≥ 6τ).

`analyze_study.py` also records a power-law fit of the loop-area decay (`fit_alpha`). It fits poorly
(large χ², stored alongside), and the manuscript does not use it; the zero-rate area is estimated
model-free from the plateau.

## Citation

Software: v1.1.0 (manuscript v10) [10.5281/zenodo.23131432](https://doi.org/10.5281/zenodo.23131432);
v1.0.0 (manuscript v9) [10.5281/zenodo.23130923](https://doi.org/10.5281/zenodo.23130923); concept DOI, always
the latest, [10.5281/zenodo.23130922](https://doi.org/10.5281/zenodo.23130922).
The update rules are those of [MyUncle](https://github.com/sandlerleon/MyUncle)
([10.5281/zenodo.21223569](https://doi.org/10.5281/zenodo.21223569)).

MIT licence.
