#!/usr/bin/env python3
"""
Error Mitigation Shoot-out  (Quantum Computing 25AM3504)
=========================================================
Compares four strategies on four benchmark circuits under a configurable
noise model (simulated, so it runs on a laptop):

    1. none      - raw noisy results
    2. readout   - measurement-error mitigation (confusion-matrix inversion)
    3. zne       - zero-noise extrapolation (global unitary folding)
    4. combined  - readout mitigation inside ZNE

Metric: absolute error of the "success probability" of each circuit
(probability of the outcome(s) an ideal quantum computer would give),
averaged over repeated trials.

Run:   python mitigation_shootout.py --quick        (about a minute)
       python mitigation_shootout.py                (full run)
       python mitigation_shootout.py --sweep        (adds noise-sweep figure)
       python mitigation_shootout.py --mitiq        (adds Mitiq cross-check)
"""
import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Native gate set. sxdg is included so that the inverse of every gate is also
# native: this makes unitary folding scale the gate count *exactly*.
BASIS = ["rz", "sx", "sxdg", "x", "cx"]
METHODS = ["none", "readout", "zne", "combined"]


# --------------------------------------------------------------------------
# 1. Benchmark circuits  (each returns name, circuit, indices of success states)
# --------------------------------------------------------------------------
def ghz(n=4):
    """GHZ state: ideal outcomes are all-0s or all-1s (50/50)."""
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return f"GHZ-{n}", qc, [0, 2**n - 1]


def bernstein_vazirani(secret="101"):
    """BV: one query reveals a hidden bit string. Ancilla ends in |1>."""
    n = len(secret)
    qc = QuantumCircuit(n + 1)
    qc.x(n)
    qc.h(range(n + 1))
    for i in range(n):
        if secret[::-1][i] == "1":  # qubit i <-> rightmost-first
            qc.cx(i, n)
    qc.h(range(n + 1))
    return f"BV-{n + 1}", qc, [(1 << n) | int(secret, 2)]


def _qft(n):
    qc = QuantumCircuit(n)
    for j in reversed(range(n)):
        qc.h(j)
        for q in range(j):
            qc.cp(math.pi / 2 ** (j - q), q, j)
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)
    return qc


def qft_roundtrip(n=3, x=5):
    """Prepare |x>, apply QFT then inverse QFT: ideal result is |x> again."""
    qc = QuantumCircuit(n)
    for q in range(n):
        if (x >> q) & 1:
            qc.x(q)
    f = _qft(n)
    qc.compose(f, inplace=True)
    qc.compose(f.inverse(), inplace=True)
    return f"QFT-{n}", qc, [x]


def _ccz(qc, a, b, c):
    qc.h(c)
    qc.ccx(a, b, c)
    qc.h(c)


def grover3(marked="101", iterations=2):
    """3-qubit Grover search for one marked item (ideal success ~ 0.945)."""
    n = 3
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(iterations):
        # oracle: flip phase of |marked>
        zeros = [i for i in range(n) if marked[::-1][i] == "0"]
        for i in zeros:
            qc.x(i)
        _ccz(qc, 0, 1, 2)
        for i in zeros:
            qc.x(i)
        # diffuser
        qc.h(range(n))
        qc.x(range(n))
        _ccz(qc, 0, 1, 2)
        qc.x(range(n))
        qc.h(range(n))
    return "Grover-3", qc, [int(marked, 2)]


def benchmark_suite():
    return [ghz(4), bernstein_vazirani("101"), qft_roundtrip(3, 5), grover3("101", 2)]


# --------------------------------------------------------------------------
# 2. Noise model + simulator wrapper
# --------------------------------------------------------------------------
@dataclass
class NoiseParams:
    p1: float = 0.001   # 1-qubit depolarizing error per sx/sxdg/x gate
    p2: float = 0.02    # 2-qubit depolarizing error per cx gate
    p01: float = 0.02   # P(read 1 | state 0)
    p10: float = 0.05   # P(read 0 | state 1)


def build_noise_model(p: NoiseParams) -> NoiseModel:
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p.p1, 1), ["sx", "sxdg", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p.p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - p.p01, p.p01], [p.p10, 1 - p.p10]])
    )
    return nm


class NoisySim:
    """Runs circuits on a noisy Aer simulator and returns probability vectors."""

    def __init__(self, noise: NoiseParams, shots: int, seed: int):
        self.sim = AerSimulator(noise_model=build_noise_model(noise))
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def probs(self, circ: QuantumCircuit) -> np.ndarray:
        c = circ.copy()
        c.measure_all()
        # optimization_level=0 -> nothing is cancelled/merged (folding stays intact)
        c = transpile(c, self.sim, optimization_level=0)
        seed = int(self.rng.integers(2**31 - 1))
        counts = self.sim.run(c, shots=self.shots, seed_simulator=seed).result().get_counts()
        arr = np.zeros(2**circ.num_qubits)
        for key, v in counts.items():
            arr[int(key.replace(" ", ""), 2)] += v
        return arr / self.shots


# --------------------------------------------------------------------------
# 3. Mitigation building blocks
# --------------------------------------------------------------------------
def calibrate_readout(sim: NoisySim, n: int) -> np.ndarray:
    """Confusion matrix A[i, j] = P(measure i | prepared j) for n qubits."""
    A = np.zeros((2**n, 2**n))
    for j in range(2**n):
        qc = QuantumCircuit(n)
        for q in range(n):
            if (j >> q) & 1:
                qc.x(q)
        A[:, j] = sim.probs(qc)
    return A


def mitigate_readout(A: np.ndarray, p_meas: np.ndarray) -> np.ndarray:
    """Solve A x = p_meas with x >= 0, then renormalise (physical distribution)."""
    x, _ = nnls(A, p_meas)
    s = x.sum()
    return x / s if s > 0 else p_meas


def fold_global(base: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Global unitary folding: U -> U (U^dagger U)^k with scale = 2k + 1."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale factors must be odd integers (1, 3, 5, ...)")
    k = (scale - 1) // 2
    folded = base.copy()
    inv = base.inverse()
    for _ in range(k):
        folded.compose(inv, inplace=True)
        folded.compose(base, inplace=True)
    return folded


def extrapolate(scales, values, kind="richardson") -> float:
    """Extrapolate measured values at noise scales back to zero noise."""
    deg = len(scales) - 1 if kind == "richardson" else 1
    return float(np.polyfit(scales, values, deg)[-1])


def zne_with_mitiq(base, sim: NoisySim, success, scales, kind):
    """Optional cross-check using the Mitiq library (the project's base paper)."""
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import LinearFactory, RichardsonFactory
    from mitiq.zne.scaling import fold_global as mitiq_fold

    def executor(circ):
        c = transpile(circ, basis_gates=BASIS, optimization_level=0)
        return float(sim.probs(c)[success].sum())

    factory = (RichardsonFactory(list(scales)) if kind == "richardson"
               else LinearFactory(list(scales)))
    return float(execute_with_zne(base, executor, factory=factory, scale_noise=mitiq_fold))


# --------------------------------------------------------------------------
# 4. Experiment
# --------------------------------------------------------------------------
def ideal_value(logical: QuantumCircuit, success) -> float:
    return float(Statevector(logical).probabilities()[success].sum())


def run_experiment(noise, shots, trials, scales, kind, seed, use_mitiq=False, verbose=True):
    suite = benchmark_suite()
    prepared = []
    for name, logical, success in suite:
        base = transpile(logical, basis_gates=BASIS, optimization_level=0)
        prepared.append((name, logical, base, success, ideal_value(logical, success)))

    rows = []
    for t in range(trials):
        sim = NoisySim(noise, shots, seed + 1000 * t)
        calib = {}
        for name, logical, base, success, ideal in prepared:
            n = base.num_qubits
            if n not in calib:
                calib[n] = calibrate_readout(sim, n)
            A = calib[n]

            raw, ro = [], []
            for s in scales:
                p = sim.probs(fold_global(base, s))
                raw.append(p[success].sum())
                ro.append(mitigate_readout(A, p)[success].sum())

            i1 = list(scales).index(1) if 1 in scales else 0
            est = {
                "none": raw[i1],
                "readout": ro[i1],
                "zne": extrapolate(scales, raw, kind),
                "combined": extrapolate(scales, ro, kind),
            }
            if use_mitiq:
                try:
                    est["zne_mitiq"] = zne_with_mitiq(base, sim, success, scales, kind)
                except Exception as e:  # keep the main experiment alive
                    if t == 0:
                        print(f"  [warn] Mitiq cross-check skipped: {e}")
            for m, v in est.items():
                rows.append(dict(trial=t, circuit=name, method=m, value=float(v),
                                 ideal=ideal, abs_error=abs(float(v) - ideal)))
        if verbose:
            print(f"  trial {t + 1}/{trials} done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Reporting
# --------------------------------------------------------------------------
def print_circuit_table(scales):
    print("\nBenchmark circuits (after transpiling to native gates):")
    print(f"{'circuit':10s}{'qubits':>7s}{'depth':>7s}{'cx':>5s}{'ideal':>8s}" +
          "".join(f"{'cx@x' + str(s):>8s}" for s in scales))
    for name, logical, success in benchmark_suite():
        base = transpile(logical, basis_gates=BASIS, optimization_level=0)
        cx = base.count_ops().get("cx", 0)
        print(f"{name:10s}{base.num_qubits:7d}{base.depth():7d}{cx:5d}"
              f"{ideal_value(logical, success):8.3f}" + "".join(f"{cx * s:8d}" for s in scales))


def summarise(df):
    g = df.groupby(["circuit", "method"])["abs_error"].agg(["mean", "std"]).reset_index()
    raw = g[g.method == "none"].set_index("circuit")["mean"]
    g["improvement_%"] = g.apply(
        lambda r: 100 * (raw[r.circuit] - r["mean"]) / raw[r.circuit] if raw[r.circuit] > 0 else np.nan,
        axis=1)
    return g


def plot_bars(summary, path, title):
    methods = [m for m in METHODS + ["zne_mitiq"] if m in summary.method.unique()]
    circuits = list(dict.fromkeys(summary.circuit))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, m in enumerate(methods):
        sub = summary[summary.method == m].set_index("circuit").loc[circuits]
        ax.bar(np.arange(len(circuits)) + k * w, sub["mean"], w, yerr=sub["std"],
               capsize=2, label=m)
    ax.set_xticks(np.arange(len(circuits)) + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(circuits)
    ax.set_ylabel("Absolute error of success probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_sweep(sweep_df, path):
    g = sweep_df.groupby(["p2", "method"])["abs_error"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        s = g[g.method == m]
        ax.plot(s.p2, s["abs_error"], marker="o", label=m)
    ax.set_xlabel("Two-qubit gate error rate (cx depolarizing p)")
    ax.set_ylabel("Mean absolute error (all circuits)")
    ax.set_title("When does each method help? (noise sweep)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--extrap", choices=["richardson", "linear"], default="richardson")
    ap.add_argument("--p1", type=float, default=0.001)
    ap.add_argument("--p2", type=float, default=0.02)
    ap.add_argument("--p01", type=float, default=0.02)
    ap.add_argument("--p10", type=float, default=0.05)
    ap.add_argument("--sweep", action="store_true", help="also sweep the cx error rate")
    ap.add_argument("--sweep-trials", type=int, default=5)
    ap.add_argument("--mitiq", action="store_true", help="add a Mitiq ZNE cross-check")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="fast smoke test: 2048 shots, 3 trials")
    a = ap.parse_args()
    if a.quick:
        a.shots, a.trials, a.sweep_trials = 2048, 3, 2

    os.makedirs(a.outdir, exist_ok=True)
    noise = NoiseParams(a.p1, a.p2, a.p01, a.p10)
    print(f"Noise: {noise}\nshots={a.shots} trials={a.trials} scales={a.scales} extrap={a.extrap}")
    print_circuit_table(a.scales)

    t0 = time.time()
    print("\nRunning main experiment ...")
    df = run_experiment(noise, a.shots, a.trials, a.scales, a.extrap, a.seed, a.mitiq)
    df.to_csv(os.path.join(a.outdir, "results_raw.csv"), index=False)
    summary = summarise(df)
    summary.to_csv(os.path.join(a.outdir, "results_summary.csv"), index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nMean absolute error (lower is better):")
    print(summary.pivot(index="circuit", columns="method", values="mean"))
    print("\nImprovement vs. no mitigation (%):")
    print(summary.pivot(index="circuit", columns="method", values="improvement_%"))
    plot_bars(summary, os.path.join(a.outdir, "fig_error_by_method.png"),
              f"Error mitigation shoot-out (cx error {a.p2:g}, {a.shots} shots, {a.trials} trials)")

    if a.sweep:
        print("\nRunning noise sweep ...")
        frames = []
        for p2 in [0.005, 0.01, 0.02, 0.03, 0.05]:
            print(f" cx error = {p2}")
            d = run_experiment(NoiseParams(a.p1, p2, a.p01, a.p10), a.shots, a.sweep_trials,
                               a.scales, a.extrap, a.seed, verbose=False)
            d["p2"] = p2
            frames.append(d)
        sw = pd.concat(frames)
        sw.to_csv(os.path.join(a.outdir, "results_sweep.csv"), index=False)
        plot_sweep(sw, os.path.join(a.outdir, "fig_noise_sweep.png"))

    print(f"\nDone in {time.time() - t0:.0f}s. Files saved in ./{a.outdir}/")


if __name__ == "__main__":
    main()
