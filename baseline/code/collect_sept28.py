"""
Pool every experiment launched 2026-09-28 against its matched control.

Each contrast is paired per volume over the patients both arms scored; a
contrast is reported only when BOTH arms have all 5 folds (partial pools
would silently change the patient set). Bootstrap resamples patients jointly
for numerator and denominator, as bootstrap_substitution_curve.py does.

Run from the repo root:  python baseline/code/collect_sept28.py
Writes baseline/logs/RESULTS_sept28.md
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

M = "baseline/models"
B = 20000


def load(arm):
    out, folds = {}, 0
    for f in range(5):
        p = f"{M}/{arm}/fold_{f}/dice_results_gaussian.json"
        if os.path.exists(p):
            out.update(json.load(open(p))["per_case"])
            folds += 1
    return out, folds


def contrast(a, b, rng):
    """Mean of a-b over shared patients, 95% bootstrap CI, Wilcoxon p, wins."""
    k = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in k])
    idx = rng.integers(0, len(d), (B, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return len(k), d.mean(), lo, hi, wilcoxon(d).pvalue, int((d > 0).sum())


def ratio(ssl, sup, full, rng):
    k = sorted(set(ssl) & set(sup) & set(full))
    u = np.array([ssl[i] - sup[i] for i in k])
    l = np.array([full[i] - sup[i] for i in k])
    idx = rng.integers(0, len(k), (B, len(k)))
    r = u[idx].mean(1) / l[idx].mean(1)
    return u.mean() / l.mean(), *np.percentile(r, [2.5, 97.5])


# (label, treated arm, control arm, endpoint arm for the ratio or None)
ROWS = [
    ("UA-MT@10, 150 ep  vs sup@10, 150 ep", "ssl_uamt_10_5fold_e150", "supervised_frac_10_e150", "supervised_frac_100"),
    ("UA-MT@10, 60 ep (legacy) vs sup@10", "ssl_uamt_10_5fold", "supervised_frac_10", "supervised_frac_100"),
    ("sup@10 150 ep vs 60 ep (convergence check)", "supervised_frac_10_e150", "supervised_frac_10", None),
    ("MT@10 vs sup@10", "ssl_mt_10_5fold", "supervised_frac_10", "supervised_frac_100"),
    ("CPS@10 vs sup@10", "ssl_cps_10_5fold", "supervised_frac_10", "supervised_frac_100"),
    ("arm B@10 vs sup@10 (consistency only)", "ssl_armb_10_5fold", "supervised_frac_10", None),
    ("UA-MT@10 (60 ep) vs arm B@10 (distinct unlabeled patients)", "ssl_uamt_10_5fold", "ssl_armb_10_5fold", None),
    ("MT@25 vs sup@25", "ssl_mt_25_5fold", "supervised_frac_25", "supervised_frac_100"),
    ("CPS@25 vs sup@25", "ssl_cps_25_5fold", "supervised_frac_25", "supervised_frac_100"),
    ("arm B@25 vs sup@25", "ssl_armb_25_5fold", "supervised_frac_25", None),
    ("UA-MT@25 vs arm B@25 (distinct unlabeled patients)", "ssl_uamt_25_5fold", "ssl_armb_25_5fold", None),
    ("UA-MT@25 vs sup@25 (total SSL gain, same scoring)", "ssl_uamt_25_5fold", "supervised_frac_25", None),
    ("UA-MT@25, 150 ep vs sup@25, 150 ep", "ssl_uamt_25_5fold_e150", "supervised_frac_25_e150", "supervised_frac_100"),
    ("sup@25 150 ep vs 60 ep (convergence check)", "supervised_frac_25_e150", "supervised_frac_25", None),
    ("CLEAN UA-MT@50 vs CLEAN sup@50", "clean_uamt_50", "clean_sup_50", "clean_sup_100"),
    ("CLEAN UA-MT@10 vs CLEAN sup@10", "clean_uamt_10", "clean_sup_10", "clean_sup_100"),
    ("CLEAN sup@100 vs CLEAN sup@50", "clean_sup_100", "clean_sup_50", None),
]


def main():
    rng = np.random.default_rng(0)
    lines = ["# Results of the 2026-09-28 launches (Gaussian, paired per volume)", "",
             "| contrast | n | Δ Dice | 95% CI | p | wins | ratio [95% CI] | status |",
             "|---|---|---|---|---|---|---|---|"]
    for label, t, c, e in ROWS:
        a, fa = load(t)
        b, fb = load(c)
        if fa < 5 or fb < 5:
            lines.append(f"| {label} | | | | | | | waiting ({t} {fa}/5, {c} {fb}/5) |")
            continue
        n, m, lo, hi, p, w = contrast(a, b, rng)
        rs = ""
        if e:
            f, fe = load(e)
            if fe == 5:
                r, rlo, rhi = ratio(a, b, f, rng)
                rs = f"{r:.3f} [{rlo:.3f}, {rhi:.3f}]"
        lines.append(f"| {label} | {n} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.1e} | {w}/{n} | {rs} | done |")
    means = ["", "## Arm means (all folds present only)", "", "| arm | folds | mean Dice |", "|---|---|---|"]
    for arm in sorted({x for r in ROWS for x in r[1:] if x}):
        d, f = load(arm)
        means.append(f"| {arm} | {f}/5 | {np.mean(list(d.values())):.4f} |" if f == 5 else f"| {arm} | {f}/5 | |")
    out = "\n".join(lines + means) + "\n"
    open("baseline/logs/RESULTS_sept28.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
