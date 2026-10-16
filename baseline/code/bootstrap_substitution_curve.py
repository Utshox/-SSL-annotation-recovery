#!/usr/bin/env python
"""Bootstrap confidence intervals for the label-substitution curve.

The curve reports, at each labeled fraction f, how much of the value of
annotating the remaining volumes is recovered by using those same volumes
unlabeled:

    ratio(f) = [UA-MT(f) - sup(f)] / [sup(100) - sup(f)]

Numerator and denominator are both noisy estimates over the same 281 patients,
so the quotient's sampling distribution is not something a delta on the means
can describe. Everything here resamples PATIENTS with replacement and recomputes
the whole curve inside each replicate, which keeps the pairing across arms and
propagates the shared noise correctly.

The headline question is not whether any single ratio differs from zero, it is
whether the ratio MOVES with the label budget. That is the paired contrast
ratio(50) - ratio(10), reported last, and it is the number that decides whether
"unlabeled data substitutes better when labels are plentiful" is a finding or an
artifact of the denominator collapsing as the supervised curve saturates.

Usage:
    python baseline/code/bootstrap_substitution_curve.py --kind gaussian
    python baseline/code/bootstrap_substitution_curve.py --kind uniform
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

FRACTIONS = [10, 25, 50]
N_BOOT = 20000
SEED = 42


def pooled(arm, kind):
    """One Dice per patient, pooled over the five out-of-fold prediction sets."""
    name = "dice_results_gaussian.json" if kind == "gaussian" else "dice_results.json"
    out = {}
    for f in sorted(glob.glob(f"baseline/models/{arm}/fold_*/{name}")):
        with open(f) as fh:
            per_case = json.load(fh).get("per_case") or {}
        overlap = out.keys() & per_case.keys()
        if overlap:
            raise ValueError(f"{arm}: patient(s) {sorted(overlap)[:3]} appear in two folds")
        out.update(per_case)
    if not out:
        raise FileNotFoundError(f"no {name} under baseline/models/{arm}/fold_*/")
    return out


def cliffs_delta(a, b):
    """Non-parametric effect size: P(a>b) - P(a<b), computed on the paired diffs."""
    d = a - b
    return (np.sum(d > 0) - np.sum(d < 0)) / len(d)


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return {labels[i]: adj[i] for i in range(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gaussian", "uniform"], default="gaussian")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sup = {f: pooled(f"supervised_frac_{f}", args.kind) for f in FRACTIONS + [100]}
    uam = {f: pooled(f"ssl_uamt_{f}_5fold", args.kind) for f in FRACTIONS}

    # Restrict to patients present in every arm, and fix one ordering so that
    # every array below is aligned patient-by-patient.
    patients = set.intersection(*[set(d) for d in list(sup.values()) + list(uam.values())])
    patients = sorted(patients)
    n = len(patients)
    print(f"blending = {args.kind} | patients common to all 7 arms = {n}\n")

    S = {f: np.array([sup[f][p] for p in patients]) for f in sup}
    U = {f: np.array([uam[f][p] for p in patients]) for f in uam}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    def ci(v):
        return np.percentile(v, [2.5, 97.5])

    report = {"kind": args.kind, "n_patients": n, "n_boot": args.n_boot, "points": {}}
    boot_ratio = {}
    pvals, labels = [], []

    print(f"{'f':>4} {'sup':>7} {'UA-MT':>7} {'d_unlab':>18} {'d_label':>18} {'ratio':>20}")
    for f in FRACTIONS:
        d_unlab = U[f] - S[f]
        d_label = S[100] - S[f]
        r = d_unlab.mean() / d_label.mean()

        bu = d_unlab[idx].mean(axis=1)
        bl = d_label[idx].mean(axis=1)
        boot_ratio[f] = bu / bl

        w = stats.wilcoxon(U[f], S[f])
        pvals.append(w.pvalue)
        labels.append(f"d_unlab@{f}")

        cu, cl, cr = ci(bu), ci(bl), ci(boot_ratio[f])
        print(f"{f:>3}% {S[f].mean():7.4f} {U[f].mean():7.4f} "
              f"{d_unlab.mean():+.4f} [{cu[0]:+.4f},{cu[1]:+.4f}] "
              f"{d_label.mean():+.4f} [{cl[0]:+.4f},{cl[1]:+.4f}] "
              f"{r:6.3f} [{cr[0]:.3f},{cr[1]:.3f}]")

        report["points"][f] = {
            "sup": float(S[f].mean()), "uamt": float(U[f].mean()),
            "d_unlabeled": float(d_unlab.mean()), "d_unlabeled_ci": cu.tolist(),
            "d_labeled": float(d_label.mean()), "d_labeled_ci": cl.tolist(),
            "ratio": float(r), "ratio_ci": cr.tolist(),
            "wilcoxon_p": float(w.pvalue),
            "cliffs_delta": float(cliffs_delta(U[f], S[f])),
        }

    adj = holm(np.array(pvals), labels)
    print("\nHolm-adjusted p (unlabeled gain > 0):")
    for k in labels:
        print(f"  {k:>14}: {adj[k]:.3e}   cliff's d = "
              f"{report['points'][int(k.split('@')[1])]['cliffs_delta']:+.3f}")
        report["points"][int(k.split("@")[1])]["holm_p"] = float(adj[k])

    print("\n=== THE GATE: does the ratio actually move with label budget? ===")
    report["contrasts"] = {}
    for a, b in [(50, 10), (50, 25), (25, 10)]:
        diff = boot_ratio[a] - boot_ratio[b]
        lo, hi = ci(diff)
        # Two-sided bootstrap p: how often does the replicate cross zero?
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        verdict = "SEPARATED" if lo > 0 else "NOT separated (CI spans 0)"
        print(f"  ratio({a}%) - ratio({b}%) = {diff.mean():+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]  p={p:.4f}  -> {verdict}")
        report["contrasts"][f"{a}_minus_{b}"] = {
            "diff": float(diff.mean()), "ci": [float(lo), float(hi)],
            "p": float(p), "separated": bool(lo > 0),
        }

    out = args.out or f"baseline/logs/substitution_curve_{args.kind}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
