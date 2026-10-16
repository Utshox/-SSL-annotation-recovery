"""Collect corrected-protocol HD95/LCC (MSD 5-fold) and TCIA external Dice (16 Oct 2026)."""
import json, glob, collections, numpy as np
from scipy.stats import wilcoxon
L = 'baseline/logs/'
rng = np.random.default_rng(0)

def ci(d):
    b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return np.percentile(b, [2.5, 97.5])

def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line); out.setdefault(r['case'], []).append(r)
    return out

def contrast(a, b, key, agg):
    common = sorted(set(a) & set(b))
    x = np.array([agg([r[key] for r in a[c]]) for c in common])
    y = np.array([agg([r[key] for r in b[c]]) for c in common])
    d = x - y; lo, hi = ci(d); p = wilcoxon(x, y).pvalue
    return len(common), x.mean(), y.mean(), d.mean(), lo, hi, p

lines = ['# Corrected-protocol evaluation, 16 Oct 2026', '']
# MSD HD95 / LCC
H = {a: load(L + f'hd95_5fold/partial_{a}_fold*_gaussian_lcc.jsonl') for a in
     ['csup10', 'cuamt10', 'csup25', 'cuamt25', 'csup50', 'cuamt50', 'csup100']}
lines += ['## MSD 5-fold, per-arm means (raw / after LCC)', '',
          '| arm | n | Dice | HD95 mean | HD95 median | ASD | Dice LCC | HD95 LCC mean | HD95 LCC median |', '|' + '---|' * 9]
for a, h in H.items():
    g = lambda k: np.array([h[c][0][k] for c in h])
    lines.append(f"| {a} | {len(h)} | {g('dice').mean():.3f} | {g('hd95').mean():.1f} | {np.median(g('hd95')):.1f} | {g('asd').mean():.2f} | "
                 f"{g('dice_lcc').mean():.3f} | {g('hd95_lcc').mean():.1f} | {np.median(g('hd95_lcc')):.1f} |")
lines += ['', '## MSD paired contrasts (A - B; HD95 negative = A better)', '',
          '| A vs B | metric | n | A | B | diff | 95% CI | Wilcoxon p |', '|' + '---|' * 8]
pairs = [('cuamt10', 'csup10'), ('cuamt25', 'csup25'), ('cuamt50', 'csup50'),
         ('cuamt25', 'csup50'), ('cuamt10', 'csup25'), ('cuamt50', 'csup100')]
for a, b in pairs:
    for k in ['dice', 'hd95', 'asd', 'dice_lcc', 'hd95_lcc']:
        n, x, y, d, lo, hi, p = contrast(H[a], H[b], k, lambda v: v[0])
        lines.append(f'| {a} vs {b} | {k} | {n} | {x:.3f} | {y:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |')
# TCIA: each case scored by all 5 fold models -> average per case
T = {a: load(L + f'tcia_crossdomain/partial_{a}_fold*_gaussian.jsonl') for a in
     ['csup25', 'cuamt25', 'csup50', 'cuamt50', 'csup100', 'sup50', 'uamt', 'sup100']}
lines += ['', '## TCIA external (n cases, Dice averaged over the 5 fold models)', '',
          '| arm | n cases | rows | Dice mean | Dice median |', '|---|---|---|---|---|']
for a, t in T.items():
    if not t: lines.append(f'| {a} | missing | | | |'); continue
    m = np.array([np.mean([r['dice'] for r in t[c]]) for c in t])
    lines.append(f'| {a} | {len(t)} | {sum(map(len, t.values()))} | {m.mean():.3f} | {np.median(m):.3f} |')
lines += ['', '| A vs B | n | A | B | diff | 95% CI | Wilcoxon p |', '|' + '---|' * 7]
for a, b in [('cuamt25', 'csup25'), ('cuamt50', 'csup50'), ('cuamt25', 'csup50'), ('cuamt50', 'csup100'), ('csup100', 'csup50')]:
    n, x, y, d, lo, hi, p = contrast(T[a], T[b], 'dice', np.mean)
    lines.append(f'| {a} vs {b} | {n} | {x:.3f} | {y:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3g} |')
open(L + 'RESULTS_oct16_eval.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
