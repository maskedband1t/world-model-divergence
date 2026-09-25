#!/usr/bin/env python3
"""Every number in the README comes from here.

Written and committed BEFORE any run existed, so the analysis cannot be tuned to the
data it will see. Pure numpy; no scipy, no sklearn.

  python harness/analysis.py --self-test                      # verify on synthetic data
  python harness/analysis.py --game Breakout --results results

Definitions are those of PREDICTIONS.md and are not restated loosely here:
  noise floor        95th percentile of control distance at that step index
  diverged at step k first k where the world-model distance exceeds the floor at k
  faithful           not yet diverged
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FLOOR_PCT = 95.0
FIXED_H = 50


# ----------------------------------------------------------------------------- io

def load_jsonl(path: Path) -> Tuple[dict, List[dict]]:
    header, rows = {}, []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if "_header" in r:
                header = r["_header"]
            else:
                rows.append(r)
    return header, rows


def to_curves(rows: List[dict], run: str, value: str) -> Dict[tuple, np.ndarray]:
    """{(seed, sample): array indexed by step-1} for one run type and one metric."""
    buf: Dict[tuple, Dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r.get("run") != run or r.get(value) is None:
            continue
        buf[(r["seed"], r.get("sample", 0))][r["step"]] = float(r[value])
    out = {}
    for k, d in buf.items():
        steps = sorted(d)
        if steps and steps == list(range(1, len(steps) + 1)):
            out[k] = np.array([d[s] for s in steps], dtype=float)
    return out


# ----------------------------------------------------------- floor and divergence

def noise_floor(control: Dict[tuple, np.ndarray], pct: float = FLOOR_PCT) -> np.ndarray:
    """Per-step percentile across control rollouts. Shape (H,)."""
    if not control:
        raise SystemExit("no control curves: run `analysis` only after the control run")
    H = min(len(v) for v in control.values())
    M = np.stack([v[:H] for v in control.values()])
    return np.percentile(M, pct, axis=0)


def divergence_step(curve: np.ndarray, floor: np.ndarray) -> Optional[int]:
    """1-indexed first crossing, or None if the curve never leaves the band."""
    H = min(len(curve), len(floor))
    above = np.nonzero(curve[:H] > floor[:H])[0]
    return int(above[0]) + 1 if above.size else None


def faithful_mask(curve: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Boolean per step: still faithful, i.e. before the first crossing."""
    H = min(len(curve), len(floor))
    d = divergence_step(curve[:H], floor[:H])
    m = np.ones(H, dtype=bool)
    if d is not None:
        m[d - 1:] = False
    return m


# --------------------------------------------------------------------- statistics

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties at .5. Rank-based."""
    pos, neg = labels.astype(bool), ~labels.astype(bool)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tie groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1]."""
    probs = np.clip(probs, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def fit_bin_calibrator(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Map score -> P(faithful) by empirical rate in quantile bins.

    Deliberately the simplest defensible mapping. If a calibrated probability needs a
    cleverer fit than this to appear, that is itself worth reporting.
    """
    qs = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    rates = np.full(n_bins, labels.mean(), dtype=float)
    for i in range(n_bins):
        m = (scores > qs[i]) & (scores <= qs[i + 1])
        if m.any():
            rates[i] = labels[m].mean()

    def apply(x: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(qs, x, side="left") - 1, 0, n_bins - 1)
        return rates[idx]

    return apply


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """Percentile CI over the seed axis. The noise floor of the claim itself."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# -------------------------------------------------------------- the predictions

def evaluate(control: Dict[tuple, np.ndarray],
             diverge: Dict[tuple, np.ndarray],
             spread: Dict[tuple, np.ndarray]) -> dict:
    floor = noise_floor(control)
    out: dict = {"floor_pct": FLOOR_PCT, "floor_len": int(len(floor))}

    # --- P1: is the shipped horizon already past the break?
    dsteps = [divergence_step(c, floor) for c in diverge.values()]
    finite = [d for d in dsteps if d is not None]
    out["P1"] = {
        "n_rollouts": len(dsteps),
        "n_never_diverged": int(sum(d is None for d in dsteps)),
        "median_divergence_step": float(np.median(finite)) if finite else None,
        "ci95_mean": bootstrap_ci(np.array(finite, dtype=float)) if finite else None,
        "fixed_H": FIXED_H,
        "verdict_median_before_H": (float(np.median(finite)) < FIXED_H) if finite else None,
    }

    # --- P2: does the break vary more across seeds than it should?
    by_seed: Dict[int, List[int]] = defaultdict(list)
    for (seed, _), c in diverge.items():
        d = divergence_step(c, floor)
        if d is not None:
            by_seed[seed].append(d)
    within = [float(np.std(v)) for v in by_seed.values() if len(v) > 1]
    seed_means = [float(np.mean(v)) for v in by_seed.values() if v]
    out["P2"] = {
        "within_seed_std_mean": float(np.mean(within)) if within else None,
        "across_seed_std": float(np.std(seed_means)) if len(seed_means) > 1 else None,
    }

    # --- P3 / P4: does sample spread rank, and does it calibrate?
    # Cross-fitted on seeds so the calibrator never scores its own training seeds.
    S, L = [], []
    for (seed, sample), c in diverge.items():
        sp = spread.get((seed, 0))
        if sp is None:
            continue
        H = min(len(c), len(sp), len(floor))
        S.append(np.stack([sp[:H], np.full(H, seed, dtype=float)], axis=1))
        L.append(faithful_mask(c[:H], floor[:H]))
    if S:
        X = np.concatenate(S)          # col 0 = spread, col 1 = seed
        y = np.concatenate(L).astype(float)
        scores, seeds_col = X[:, 0], X[:, 1]
        uniq = np.unique(seeds_col)
        half = uniq[: max(1, len(uniq) // 2)]
        tr, te = np.isin(seeds_col, half), ~np.isin(seeds_col, half)
        # low spread should mean faithful, so the ranking score is negated spread
        out["P3"] = {"auroc_neg_spread": float(auroc(-scores, y)), "n": int(len(y))}
        if tr.any() and te.any() and len(np.unique(y[tr])) > 1:
            cal = fit_bin_calibrator(-scores[tr], y[tr])
            p_te = cal(-scores[te])
            out["P4"] = {
                "ece_heldout": float(ece(p_te, y[te])),
                "base_rate_heldout": float(y[te].mean()),
                "ece_of_constant_base_rate": float(
                    ece(np.full(int(te.sum()), y[tr].mean()), y[te])),
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
            }
        else:
            out["P4"] = {"error": "not enough seeds to cross-fit; need >= 2"}
    else:
        out["P3"] = out["P4"] = {"error": "no spread rows"}

    # --- P5: calibrated gate vs fixed H at EQUAL budget
    if S and "error" not in out.get("P4", {}):
        cal = fit_bin_calibrator(-np.concatenate([x[:, 0] for x in S]),
                                 np.concatenate(L).astype(float))
        fixed_f, fixed_b, gate_curves = [], [], []
        for (seed, sample), c in diverge.items():
            sp = spread.get((seed, 0))
            if sp is None:
                continue
            H = min(len(c), len(sp), len(floor))
            fm = faithful_mask(c[:H], floor[:H])
            k = min(FIXED_H, H)
            fixed_f.append(int(fm[:k].sum()))
            fixed_b.append(k)
            gate_curves.append((cal(-sp[:H]), fm))
        # pick the threshold whose MEAN stopping length matches fixed-H's mean budget
        target = float(np.mean(fixed_b))
        best = None
        for thr in np.linspace(0.01, 0.99, 99):
            lens, faith = [], []
            for p, fm in gate_curves:
                below = np.nonzero(p < thr)[0]
                stop = int(below[0]) if below.size else len(p)
                lens.append(stop)
                faith.append(int(fm[:stop].sum()))
            gap = abs(float(np.mean(lens)) - target)
            if best is None or gap < best[0]:
                best = (gap, float(thr), float(np.mean(lens)), float(np.mean(faith)))
        out["P5"] = {
            "fixed_H_budget": target,
            "fixed_H_faithful_frames": float(np.mean(fixed_f)),
            "gate_threshold": best[1], "gate_budget": best[2],
            "gate_faithful_frames": best[3],
            "gate_minus_fixed": best[3] - float(np.mean(fixed_f)),
            "budget_mismatch": best[0],
        }
    return out


# ---------------------------------------------------------------------- self-test

def self_test() -> None:
    """Synthetic data with known answers. Guards the maths, not the experiment."""
    rng = np.random.default_rng(0)
    H = 80

    # AUROC: perfect separation, perfect inversion, and pure noise
    s = np.array([0.1, 0.2, 0.3, 0.4]); y = np.array([0, 0, 1, 1])
    assert abs(auroc(s, y) - 1.0) < 1e-9, auroc(s, y)
    assert abs(auroc(-s, y) - 0.0) < 1e-9
    assert abs(auroc(np.ones(4), y) - 0.5) < 1e-9, "all ties must give .5"

    # ECE: a perfectly calibrated set is ~0, a confidently wrong one is ~1
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    lab = (rng.random(3000) < p).astype(float)
    assert ece(p, lab) < 0.03, ece(p, lab)
    assert ece(np.ones(1000), np.zeros(1000)) > 0.97

    # floor and divergence step
    ctrl = {(i, 0): np.full(H, 1.0) for i in range(10)}
    floor = noise_floor(ctrl)
    assert floor.shape == (H,) and abs(floor[0] - 1.0) < 1e-9
    curve = np.concatenate([np.full(30, 0.5), np.full(H - 30, 5.0)])
    assert divergence_step(curve, floor) == 31, divergence_step(curve, floor)
    assert divergence_step(np.full(H, 0.1), floor) is None
    fm = faithful_mask(curve, floor)
    assert fm[:30].all() and not fm[30:].any()

    # calibrator is monotone in the right direction and cannot exceed [0,1]
    sc = rng.normal(size=4000)
    lb = (rng.random(4000) < 1 / (1 + np.exp(-sc))).astype(float)
    cal = fit_bin_calibrator(sc, lb)
    got = cal(np.array([-3.0, 0.0, 3.0]))
    assert got[0] < got[1] < got[2], got
    assert (got >= 0).all() and (got <= 1).all()

    # end-to-end: a spread that genuinely predicts divergence must beat chance,
    # and a spread that is pure noise must not.
    def build(informative: bool):
        dv, sp = {}, {}
        for seed in range(8):
            brk = int(rng.integers(20, 60))
            c = np.where(np.arange(H) < brk, 0.5, 5.0)
            dv[(seed, 0)] = c
            base = np.linspace(0, 1, H) if informative else rng.random(H)
            sp[(seed, 0)] = base + 0.01 * rng.random(H)
        return dv, sp

    dv, sp = build(True)
    r = evaluate(ctrl, dv, sp)
    assert r["P3"]["auroc_neg_spread"] > 0.8, r["P3"]
    assert r["P1"]["median_divergence_step"] is not None
    assert "gate_minus_fixed" in r["P5"]

    dv, sp = build(False)
    r2 = evaluate(ctrl, dv, sp)
    assert 0.35 < r2["P3"]["auroc_neg_spread"] < 0.65, r2["P3"]

    print("self-test passed:")
    print("  auroc, ece, floor, divergence_step, faithful_mask, calibrator, evaluate")
    print(f"  informative spread -> AUROC {r['P3']['auroc_neg_spread']:.3f}")
    print(f"  noise spread       -> AUROC {r2['P3']['auroc_neg_spread']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--game", default="Breakout")
    ap.add_argument("--results", default="results")
    ap.add_argument("--metric", default="lpips", choices=["lpips", "l2"],
                    help="PREDICTIONS.md: report both; disagreement IS the finding")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    res = Path(a.results)
    _, crows = load_jsonl(res / f"control_{a.game}.jsonl")
    _, drows = load_jsonl(res / f"diverge_{a.game}.jsonl")
    out = evaluate(
        to_curves(crows, "control", a.metric),
        to_curves(drows, "diverge", a.metric),
        to_curves(drows, "spread", "spread"),
    )
    out["game"], out["metric"] = a.game, a.metric
    print(json.dumps(out, indent=2))
    (res / f"summary_{a.game}_{a.metric}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
