"""
Pool the corrected-protocol runs launched 2026-10-15 on the second account
(MUST #1-3, #5) against their matched controls. Same paired per-volume
bootstrap as collect_sept28.py, reused from it.

Run from the repo root:  python baseline/code/collect_oct15.py
Writes baseline/logs/RESULTS_oct15.md
"""
import numpy as np

from collect_sept28 import contrast, load, ratio

ROWS = [
    # MUST #1: 25% budget under the corrected protocol
    ("UA-MT@25 vs sup@25", "clean_uamt_25", "clean_sup_25", "clean_sup_100"),
    # MUST #2: MT / CPS at 10% (150 ep) and 50%
    ("MT@10 vs sup@10", "clean_mt_10", "clean_sup_10", "clean_sup_100"),
    ("CPS@10 vs sup@10", "clean_cps_10", "clean_sup_10", "clean_sup_100"),
    ("MT@50 vs sup@50", "clean_mt_50", "clean_sup_50", "clean_sup_100"),
    ("CPS@50 vs sup@50", "clean_cps_50", "clean_sup_50", "clean_sup_100"),
    # MUST #3: arm B = consistency on labeled scans only
    ("arm B@10 vs sup@10 (consistency alone)", "clean_armb_10", "clean_sup_10", "clean_sup_100"),
    ("UA-MT@10 vs arm B@10 (unlabeled scans)", "clean_uamt_10", "clean_armb_10", None),
    ("arm B@50 vs sup@50 (consistency alone)", "clean_armb_50", "clean_sup_50", "clean_sup_100"),
    ("UA-MT@50 vs arm B@50 (unlabeled scans)", "clean_uamt_50", "clean_armb_50", None),
    # reference rows already in RESULTS_sept28
    ("UA-MT@10 vs sup@10 (Sept)", "clean_uamt_10", "clean_sup_10", "clean_sup_100"),
    ("UA-MT@50 vs sup@50 (Sept)", "clean_uamt_50", "clean_sup_50", "clean_sup_100"),
    # label-equivalent: does UA-MT@25 reach sup@50?
    ("UA-MT@25 vs sup@50", "clean_uamt_25", "clean_sup_50", None),
    ("UA-MT@10 vs sup@25", "clean_uamt_10", "clean_sup_25", None),
]
SEEDS = {"sup@25": ["clean_sup_25", "clean_sup_25_ts1", "clean_sup_25_ts2"],
         "UA-MT@25": ["clean_uamt_25", "clean_uamt_25_ts1", "clean_uamt_25_ts2"]}


def mean(arm):
    d, f = load(arm)
    assert f == 5, (arm, f)
    return d


def main():
    rng = np.random.default_rng(0)
    L = ["# Corrected-protocol runs of 2026-10-15 (Gaussian, paired per volume)", "",
         "| contrast | n | Δ Dice | 95% CI | p | wins | ratio [95% CI] |", "|---|---|---|---|---|---|---|"]
    for label, t, c, e in ROWS:
        a, b = mean(t), mean(c)
        n, m, lo, hi, p, w = contrast(a, b, rng)
        rs = ""
        if e:
            r, rlo, rhi = ratio(a, b, mean(e), rng)
            rs = f"{r:.3f} [{rlo:.3f}, {rhi:.3f}]"
        L.append(f"| {label} | {n} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.1e} | {w}/{n} | {rs} |")

    L += ["", "## Seed repeats at 25% (same split, --train_seed 42/1/2)", "",
          "| arm | seed 42 | seed 1 | seed 2 | mean | SD |", "|---|---|---|---|---|---|"]
    per = {}
    for k, arms in SEEDS.items():
        v = [np.mean(list(mean(a).values())) for a in arms]
        per[k] = v
        L.append(f"| {k} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {np.mean(v):.4f} | {np.std(v, ddof=1):.4f} |")
    g = [u - s for u, s in zip(per["UA-MT@25"], per["sup@25"])]
    L.append(f"| Δ (UA-MT - sup) | " + " | ".join(f"{x:+.4f}" for x in g) + f" | {np.mean(g):+.4f} | {np.std(g, ddof=1):.4f} |")
    # seed-averaged paired contrast
    u = [mean(a) for a in SEEDS["UA-MT@25"]]
    s = [mean(a) for a in SEEDS["sup@25"]]
    k = sorted(set.intersection(*[set(x) for x in u + s]))
    ua = {i: np.mean([x[i] for x in u]) for i in k}
    sa = {i: np.mean([x[i] for x in s]) for i in k}
    n, m, lo, hi, p, w = contrast(ua, sa, rng)
    r, rlo, rhi = ratio(ua, sa, mean("clean_sup_100"), rng)
    L.append("")
    L.append(f"Seed-averaged UA-MT@25 vs sup@25: n={n}, Δ {m:+.4f} [{lo:+.4f}, {hi:+.4f}], p={p:.1e}, wins {w}/{n}, ratio {r:.3f} [{rlo:.3f}, {rhi:.3f}]")

    L += ["", "## Arm means", "", "| arm | mean Dice |", "|---|---|"]
    arms = sorted({x for r in ROWS for x in r[1:] if x} | {a for v in SEEDS.values() for a in v})
    L += [f"| {a} | {np.mean(list(mean(a).values())):.4f} |" for a in arms]
    out = "\n".join(L) + "\n"
    open("baseline/logs/RESULTS_oct15.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
