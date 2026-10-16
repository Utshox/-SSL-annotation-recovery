"""Make the epoch-30 early-stopping control (clean_sup_10_es30) the primary 10% control (user decision 2026-10-16).
UA-MT@10 (clean_uamt_10) is unchanged: all its selected checkpoints fall after epoch 70, so an
early-stopping start at epoch 30 selects identically. Writes baseline/logs/recompute_10pct_es30.json"""
import json, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "baseline/code")
from collect_sept28 import load, contrast, ratio, B
from label_equivalent_gain import isotonic, invert
N_LABELS = {10: 20, 25: 50, 50: 101, 100: 202}
rng = np.random.default_rng(42)
L = lambda r: load(r)[0]
def avg(*runs):
    ds = [L(r) for r in runs]; ks = set.intersection(*[set(d) for d in ds])
    return {k: float(np.mean([d[k] for d in ds])) for k in ks}
sup10 = L("clean_sup_10_es30"); sup10_old = L("clean_sup_10"); u10 = L("clean_uamt_10")
sup25 = avg("clean_sup_25", "clean_sup_25_ts1", "clean_sup_25_ts2"); u25 = avg("clean_uamt_25", "clean_uamt_25_ts1", "clean_uamt_25_ts2")
sup50 = L("clean_sup_50"); u50 = L("clean_uamt_50"); sup100 = L("clean_sup_100")
out = {}
n, d, lo, hi, p, w = contrast(u10, sup10, rng); out["dSSL10"] = dict(n=n, d=d, lo=lo, hi=hi, p=p, wins=w)
n, d, lo, hi, p, w = contrast(sup100, sup10, rng); out["dLab10"] = dict(d=d, lo=lo, hi=hi, p=p, wins=w)
out["sup10_mean"] = float(np.mean(list(sup10.values()))); out["sup10_sd"] = float(np.std(list(sup10.values()), ddof=1))
out["sup10_old_mean"] = float(np.mean(list(sup10_old.values())))
n, d, lo, hi, p, w = contrast(sup10, sup10_old, rng); out["es30_vs_es0_control"] = dict(d=d, lo=lo, hi=hi, p=p)
r, lo, hi = ratio(u10, sup10, sup100, rng); out["r10"] = dict(r=r, lo=lo, hi=hi)
# paired contrast r50 - r10 (joint patient resampling)
k = sorted(set(u10) & set(sup10) & set(u50) & set(sup50) & set(sup100))
a = lambda dct: np.array([dct[i] for i in k])
U10, S10, U50, S50, F = map(a, (u10, sup10, u50, sup50, sup100))
idx = rng.integers(0, len(k), (B, len(k)))
r10b = (U10 - S10)[idx].mean(1) / (F - S10)[idx].mean(1); r50b = (U50 - S50)[idx].mean(1) / (F - S50)[idx].mean(1)
est = (U50 - S50).mean() / (F - S50).mean() - (U10 - S10).mean() / (F - S10).mean()
out["contrast"] = dict(d=float(est), lo=float(np.percentile(r50b - r10b, 2.5)), hi=float(np.percentile(r50b - r10b, 97.5)))
# label equivalence at 10% with the new curve point
k2 = sorted(set(k) & set(sup25) & set(u25)); S25 = np.array([sup25[i] for i in k2])
c = lambda arr: np.array([arr[i] for i in k2])
S10b, U10b, S50b, F2 = c(sup10), c(u10), c(sup50), c(sup100)
counts = [20, 50, 101, 202]
neq, cens = invert(counts, isotonic([S10b.mean(), S25.mean(), S50b.mean(), F2.mean()]), U10b.mean())
bs = []
for _ in range(10000):
    ii = rng.integers(0, len(k2), len(k2))
    bs.append(invert(counts, isotonic([S10b[ii].mean(), S25[ii].mean(), S50b[ii].mean(), F2[ii].mean()]), U10b[ii].mean())[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
out["eq10"] = dict(n_eq=neq, lo=lo, hi=hi, mult=neq / 20, per_scan=(neq - 20) / 182, per_lo=(lo - 20) / 182, per_hi=(hi - 20) / 182,
                   tot=42, tot_eq=neq + 22, tot_mult=(neq + 22) / 42)
json.dump(out, open("baseline/logs/recompute_10pct_es30.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
