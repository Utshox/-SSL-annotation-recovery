#!/usr/bin/env python3
"""Pool out-of-fold Dice scores for the augmentation x SSL experiment."""
import argparse
import json
import os

import numpy as np

ARMS = {
    "A0": "baseline/models/clean_sup_10_es30",
    "C0": "baseline/models/clean_uamt_10",
    "A1": "baseline/models/aug_sup_10",
    "C1": "baseline/models/aug_uamt_10",
}
CONTRASTS = [
    ("aug gain (supervised)", "A1 - A0", {"A1": 1, "A0": -1}),
    ("aug gain (SSL)", "C1 - C0", {"C1": 1, "C0": -1}),
    ("SSL gain without aug", "C0 - A0", {"C0": 1, "A0": -1}),
    ("SSL gain with aug (her main contrast)", "C1 - A1", {"C1": 1, "A1": -1}),
    ("interaction", "(C1 - A1) - (C0 - A0)",
     {"C1": 1, "A1": -1, "C0": -1, "A0": 1}),
    ("total", "C1 - A0", {"C1": 1, "A0": -1}),
]
RESAMPLES = 20000


def load_arm(root, relative):
    pooled, fold_means = {}, {}
    for fold in range(5):
        path = os.path.join(root, relative, f"fold_{fold}",
                            "dice_results_gaussian.json")
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            data = json.load(handle)
        fold_mean = float(data["mean_dice"])
        if not np.isfinite(fold_mean):
            raise ValueError(f"Nonfinite mean_dice: {path}")
        fold_means[fold] = fold_mean
        for case, value in data["per_case"].items():
            if case in pooled:
                raise ValueError(f"Duplicate case {case!r} within arm {relative}")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite Dice for {case!r}: {path}")
            pooled[case] = value
    return pooled, fold_means


def paired_statistics(differences):
    n = differences.size
    rng = np.random.default_rng(0)
    observed = float(differences.mean())
    boot = np.empty(RESAMPLES)
    extreme = 0
    threshold = abs(observed)
    tolerance = 1e-14 * max(1.0, threshold)
    for start in range(0, RESAMPLES, 500):
        size = min(500, RESAMPLES - start)
        indices = rng.integers(0, n, size=(size, n))
        boot[start:start + size] = differences[indices].mean(axis=1)
    for start in range(0, RESAMPLES, 500):
        size = min(500, RESAMPLES - start)
        signs = 2 * rng.integers(0, 2, size=(size, n)) - 1
        permuted = (signs * differences).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= threshold - tolerance))
    low, high = np.percentile(boot, [2.5, 97.5])
    p_value = (extreme + 1) / (RESAMPLES + 1)
    reading = ("positive" if low > 0 else
               "negative" if high < 0 else "null (CI spans 0)")
    return observed, low, high, p_value, reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=["generic", "abayomi"], default="generic",
                        help="abayomi = arms trained with --aug_style abayomi")
    parser.add_argument("--root",
                        default=".")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    if args.style == "abayomi":
        ARMS["A1"] = "baseline/models/augab_sup_10"
        ARMS["C1"] = "baseline/models/augab_uamt_10"
    arms = {name: load_arm(root, path) for name, path in ARMS.items()}
    lines = [
        "# Augmentation × SSL: pooled 5-fold pancreas CT Dice",
        "",
        f"Root: `{root}`",
        "",
        "Expected complete coverage: 281 unique out-of-fold volumes per arm.",
        "Means below pool cases, not fold means. Fold means are from the JSON files.",
        "",
        "| Arm | Folds found | Folds missing | n cases | Mean Dice | Per-fold mean Dice |",
        "|---|---|---|---:|---:|---|",
    ]
    for name, (cases, folds) in arms.items():
        found = ", ".join(map(str, sorted(folds))) or "none"
        missing = ", ".join(str(f) for f in range(5) if f not in folds) or "none"
        mean = f"{np.mean(list(cases.values())):.6f}" if cases else "NA"
        per_fold = "; ".join(f"{f}: {folds[f]:.6f}" for f in sorted(folds)) or "NA"
        lines.append(f"| {name} | {found} | {missing} | {len(cases)} | {mean} | {per_fold} |")
    lines += [
        "",
        "## Paired contrasts",
        "",
        "Each contrast uses only case IDs shared by all involved arms.",
        "95% percentile paired bootstrap CIs: 20,000 resamples.",
        "Two-sided sign-flip tests: 20,000 flips, p = (extreme + 1) / 20,001.",
        "Seed 0 per contrast; improved means a strictly positive difference (ties excluded).",
        "",
        "| Contrast | Definition | n | Mean difference | 95% CI | Improved / n | p-value |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    readings = []
    for label, formula, coefficients in CONTRASTS:
        shared = set.intersection(*(set(arms[a][0]) for a in coefficients))
        ids = sorted(shared)
        n = len(ids)
        if not n:
            lines.append(f"| {label} | {formula} | 0 | NA | NA | 0 / 0 | NA |")
            readings.append(f"- {label}: unavailable (no paired cases).")
            continue
        differences = np.array([
            sum(weight * arms[arm][0][case] for arm, weight in coefficients.items())
            for case in ids
        ], dtype=float)
        mean, low, high, p_value, reading = paired_statistics(differences)
        improved = int(np.count_nonzero(differences > 0))
        lines.append(
            f"| {label} | {formula} | {n} | {mean:.6f} | "
            f"[{low:.6f}, {high:.6f}] | {improved} / {n} | {p_value:.6f} |"
        )
        readings.append(f"- {label}: {reading}.")
    lines += ["", "## Reading", ""] + readings
    content = "\n".join(lines) + "\n"
    destination = os.path.join(root, f"baseline/logs/RESULTS_aug_2x2_{args.style}.md")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
