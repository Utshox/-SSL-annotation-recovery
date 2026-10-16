"""Corrected-protocol label equivalence (2026-10-16). 25% arms = mean of 3 training seeds per volume.
Same inversion as label_equivalent_gain.py (piecewise-linear, isotonic repair, patient bootstrap).
Run from repo root. Writes baseline/logs/label_equivalent_corrected.json"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "baseline/code")
from collect_sept28 import load
from label_equivalent_gain import isotonic, invert
N_LABELS = {10: 20, 25: 50, 50: 101, 100: 202}  # corrected protocol, fold 0 split_info (22 inner-val removed)

def arm(*runs):
    ds = [load(r) for r in runs]
    for d, f in ds: assert f == 5, runs
    keys = sorted(set.intersection(*[set(d) for d, _ in ds]))
    return {k: np.mean([d[k] for d, _ in ds]) for k in keys}

S = {10: arm("clean_sup_10"), 25: arm("clean_sup_25", "clean_sup_25_ts1", "clean_sup_25_ts2"),
     50: arm("clean_sup_50"), 100: arm("clean_sup_100")}
U = {10: arm("clean_uamt_10"), 25: arm("clean_uamt_25", "clean_uamt_25_ts1", "clean_uamt_25_ts2"), 50: arm("clean_uamt_50")}
keys = sorted(set.intersection(*[set(v) for v in list(S.values()) + list(U.values())]))
s = {f: np.array([S[f][k] for k in keys]) for f in S}; u = {f: np.array([U[f][k] for k in keys]) for f in U}
counts = [N_LABELS[f] for f in (10, 25, 50, 100)]
out = {"n": len(keys), "sup_means": {f: float(s[f].mean()) for f in s}, "uamt_means": {f: float(u[f].mean()) for f in u}}
rng = np.random.default_rng(42); B = 10000
for f in (10, 25, 50):
    curve = isotonic([s[g].mean() for g in (10, 25, 50, 100)])
    neq, cens = invert(counts, curve, u[f].mean())
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        c = isotonic([s[g][idx].mean() for g in (10, 25, 50, 100)])
        boots.append(invert(counts, c, u[f][idx].mean())[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[f"eq_{f}"] = {"n_labels": N_LABELS[f], "n_eq": neq, "gain": neq - N_LABELS[f], "gain_ci": [lo - N_LABELS[f], hi - N_LABELS[f]],
                      "multiple": neq / N_LABELS[f], "censored": cens}
for name, a, b in [("uamt25_vs_sup50", u[25], s[50]), ("uamt10_vs_sup25", u[10], s[25]), ("uamt50_vs_sup100", u[50], s[100])]:
    d = a - b; bt = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    out[name] = {"diff": float(d.mean()), "ci": list(map(float, np.percentile(bt, [2.5, 97.5]))), "p": float(wilcoxon(a, b).pvalue)}
json.dump(out, open("baseline/logs/label_equivalent_corrected.json", "w"), indent=1)
print(json.dumps(out, indent=1))
