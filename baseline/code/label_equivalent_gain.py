#!/usr/bin/env python
"""Annotation-equivalent value of the semi-supervised intervention.

Reports, at each labeled fraction, the supervised label count whose expected
Dice matches what the semi-supervised arm achieves:

    S(n_eq) = D_SSL(n)        ->        gain g(n) = n_eq - n

This is an interpretable transformation of the measured supervised operating
points, not the recovery of a hidden continuous law. The supervised curve is
known at four label counts only (22, 56, 112, 225), so the inversion is a
piecewise-linear interpolation between the two adjacent measured points. A
parametric saturating fit would carry nearly as many free parameters as data
points, so it is offered only as a labelled sensitivity check, never averaged
into the primary result.

The whole inversion is bootstrapped, not just the Dice differences: each
replicate resamples PATIENTS with replacement, recomputes every arm's mean,
rebuilds the supervised curve and inverts it inside that replicate. Pairing
across arms is preserved because one patient index vector indexes every arm.

A resampled supervised curve can come out locally non-monotonic, which makes the
inversion ill-posed. A supervised learning curve is expected to be
non-decreasing in label count, so such replicates are repaired by isotonic
(pool-adjacent-violators) regression rather than discarded, and the frequency
and size of those repairs is reported so the reader can judge how often it bit.

Usage:
    python baseline/code/label_equivalent_gain.py
    python baseline/code/label_equivalent_gain.py --kind uniform --n_boot 20000
"""
import argparse
import json

import numpy as np

from bootstrap_substitution_curve import FRACTIONS, N_BOOT, SEED, pooled

# Labeled volumes per fraction, and the full-supervision endpoint.
N_LABELS = {10: 22, 25: 56, 50: 112, 100: 225}
N_TOTAL = 225


def isotonic(y):
    """Pool-adjacent-violators: nearest non-decreasing sequence, equal weights."""
    y = list(map(float, y))
    # Each block carries (sum, count) so merged blocks average correctly.
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return np.array(out)


def invert(counts, dices, target):
    """Smallest label count whose interpolated supervised Dice reaches target.

    Returns (n_eq, censored). censored is True when the target exceeds the
    curve's own endpoint, in which case n_eq is pinned to that endpoint and the
    equivalent gain is a lower bound rather than an estimate.
    """
    if target <= dices[0]:
        return float(counts[0]), False
    if target >= dices[-1]:
        return float(counts[-1]), True
    for i in range(len(counts) - 1):
        d0, d1 = dices[i], dices[i + 1]
        if d0 <= target <= d1:
            if d1 == d0:
                return float(counts[i]), False
            w = (target - d0) / (d1 - d0)
            return float(counts[i] + w * (counts[i + 1] - counts[i])), False
    return float(counts[-1]), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gaussian", "uniform"], default="gaussian")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="baseline/logs/label_equivalent_gain.json")
    args = ap.parse_args()

    sup = {f: pooled(f"supervised_frac_{f}", args.kind) for f in FRACTIONS + [100]}
    uam = {f: pooled(f"ssl_uamt_{f}_5fold", args.kind) for f in FRACTIONS}

    patients = sorted(set.intersection(*[set(d) for d in list(sup.values()) + list(uam.values())]))
    n = len(patients)
    print(f"blending = {args.kind} | patients common to all 7 arms = {n}")
    print(f"bootstrap replicates = {args.n_boot}\n")

    S = {f: np.array([sup[f][p] for p in patients]) for f in sup}
    U = {f: np.array([uam[f][p] for p in patients]) for f in uam}

    counts = np.array([N_LABELS[f] for f in FRACTIONS + [100]], dtype=float)

    def curve_and_gains(pick):
        """One replicate: build the supervised curve, invert it at each budget."""
        dices = np.array([S[f][pick].mean() for f in FRACTIONS + [100]])
        repaired = bool(np.any(np.diff(dices) < 0))
        shift = 0.0
        if repaired:
            fixed = isotonic(dices)
            shift = float(np.max(np.abs(fixed - dices)))
            dices = fixed
        gains, censored = {}, {}
        for f in FRACTIONS:
            n_eq, cens = invert(counts, dices, U[f][pick].mean())
            gains[f] = n_eq - N_LABELS[f]
            censored[f] = cens
        return gains, repaired, shift, censored

    all_idx = np.arange(n)
    point, _, _, point_cens = curve_and_gains(all_idx)

    rng = np.random.default_rng(SEED)
    boot = {f: np.empty(args.n_boot) for f in FRACTIONS}
    n_repaired = 0
    max_shift = 0.0
    n_censored = {f: 0 for f in FRACTIONS}
    for b in range(args.n_boot):
        pick = rng.integers(0, n, size=n)
        g, rep, shift, cens = curve_and_gains(pick)
        n_repaired += rep
        max_shift = max(max_shift, shift)
        for f in FRACTIONS:
            boot[f][b] = g[f]
            n_censored[f] += cens[f]

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    print("Annotation-equivalent value of the UA-MT intervention")
    print("(report order: observed Dice gain, equivalent labels, per-scan, ratio)\n")
    header = f"{'labels':>7} {'unlab':>6} {'dDice':>8} {'+equiv labels':>22} {'per unannotated scan':>26}"
    print(header)
    print("-" * len(header))
    rows = {}
    for f in FRACTIONS:
        nl = N_LABELS[f]
        unlab = N_TOTAL - nl
        ddice = U[f].mean() - S[f].mean()
        lo, hi = ci(boot[f])
        plo, phi = lo / unlab, hi / unlab
        star = "*" if n_censored[f] else " "
        print(f"{nl:>7} {unlab:>6} {ddice:>+8.4f} "
              f"{point[f]:>+9.0f} [{lo:>+6.0f},{hi:>+6.0f}]{star} "
              f"{point[f]/unlab:>13.2f} [{plo:.2f},{phi:.2f}]")
        rows[f] = {
            "n_labeled": nl, "n_unlabeled": unlab,
            "dice_ssl": float(U[f].mean()), "dice_sup": float(S[f].mean()),
            "delta_dice": float(ddice),
            "equiv_labels_gained": float(point[f]),
            "equiv_labels_ci": [lo, hi],
            "per_unannotated_scan": float(point[f] / unlab),
            "per_unannotated_scan_ci": [plo, phi],
            "point_estimate_censored": bool(point_cens[f]),
            "boot_replicates_censored": int(n_censored[f]),
        }

    pct = 100.0 * n_repaired / args.n_boot
    print(f"\nisotonic repairs: {n_repaired}/{args.n_boot} replicates ({pct:.2f}%), "
          f"largest adjustment {max_shift:.4f} Dice")
    if any(n_censored.values()):
        print("* some replicates had the SSL result above the supervised endpoint; "
              "those are pinned to 225 labels, so the interval is a lower bound there.")
    print("\nInterpolation is piecewise-linear between measured supervised points "
          f"at {[int(c) for c in counts]} labels.")

    payload = {
        "blending": args.kind, "n_patients": n, "n_boot": args.n_boot,
        "supervised_curve": {int(N_LABELS[f]): float(S[f].mean()) for f in FRACTIONS + [100]},
        "rows": rows,
        "isotonic_repairs": n_repaired,
        "isotonic_repair_fraction": pct / 100.0,
        "isotonic_max_adjustment": max_shift,
        "interpolation": "piecewise-linear",
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
