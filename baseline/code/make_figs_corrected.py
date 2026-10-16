"""Corrected-protocol figures (2026-10-16). Run from repo root; writes overleaf_export/images/*_corrected.png"""
import glob, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
def M(t): return t.replace("-", "\u2212")
import matplotlib.pyplot as plt
sys.path.insert(0, "baseline/code")
from collect_sept28 import load
rng = np.random.default_rng(0); B = 4000
OUT = "overleaf_export/images/"
BLUE, ORANGE, GREEN, GREY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#777777", "#CC79A7"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def ci_mean(x): b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
def arm(*runs):
    ds = [load(r)[0] for r in runs]; ks = sorted(set.intersection(*[set(d) for d in ds]))
    return dict(zip(ks, [np.mean([d[k] for d in ds]) for k in ks]))
def vec(d, keys): return np.array([d[k] for k in keys])
S = {10: arm("clean_sup_10_es30"), 25: arm("clean_sup_25", "clean_sup_25_ts1", "clean_sup_25_ts2"), 50: arm("clean_sup_50"), 100: arm("clean_sup_100")}
U = {10: arm("clean_uamt_10"), 25: arm("clean_uamt_25", "clean_uamt_25_ts1", "clean_uamt_25_ts2"), 50: arm("clean_uamt_50")}
K = sorted(S[50])
# ---------------- Fig 1: annotation-recovery curve
ORIG = {"sup": {10: 0.4907, 25: 0.6250, 50: 0.6937, 100: 0.7398}, "uamt": {10: 0.5884, 25: 0.6948, 50: 0.7295},
        "ratio": {10: (0.392, 0.347, 0.435), 25: (0.608, 0.536, 0.676), 50: (0.78, 0.67, 0.89)}}
CR = {10: (0.406, 0.366, 0.445), 25: (0.514, 0.470, 0.561), 50: (0.788, 0.690, 0.901)}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xs = [10, 25, 50]
for key, col, lab, mk in [("sup", BLUE, "Supervised control", "o"), ("uamt", ORANGE, "UA-MT", "s")]:
    src = S if key == "sup" else U
    m = [vec(src[f], K).mean() for f in xs]; c = np.array([ci_mean(vec(src[f], K)) for f in xs])
    ax[0].errorbar(xs, m, yerr=[np.array(m) - c[:, 0], c[:, 1] - np.array(m)], color=col, marker=mk, ms=7, capsize=3, lw=1.8, label=f"{lab} (corrected)")
    ax[0].plot(xs, [ORIG[key][f] for f in xs], color=col, marker=mk, ms=5, alpha=0.35, ls="--", label=f"{lab} (original)")
full = vec(S[100], K).mean()
ax[0].axhline(full, color=BLUE, ls=":", lw=1); ax[0].text(51, full + 0.004, f"whole pool {full:.3f}", color=BLUE, fontsize=8)
for f in xs:
    d = vec(U[f], K).mean() - vec(S[f], K).mean()
    ax[0].annotate(f"+{d:.3f}", (f, (vec(U[f], K).mean() + vec(S[f], K).mean()) / 2), xytext=(6, 0), textcoords="offset points", color=ORANGE, fontsize=8, va="center")
ax[0].set_xticks(xs); ax[0].set_xlabel("Labeled fraction of the training pool (%)"); ax[0].set_ylabel("Dice (281 volumes)")
ax[0].set_title("(a) Absolute gain falls with the budget", loc="left"); ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
r = np.array([CR[f] for f in xs]); o = np.array([ORIG["ratio"][f] for f in xs])
ax[1].errorbar(xs, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], color=GREEN, marker="o", ms=7, capsize=3, lw=1.8, label="corrected")
ax[1].errorbar(xs, o[:, 0], yerr=[o[:, 0] - o[:, 1], o[:, 2] - o[:, 0]], color=GREEN, marker="o", ms=5, capsize=3, alpha=0.35, ls="--", label="original")
for f, v in zip(xs, r[:, 0]): ax[1].annotate(f"{v:.2f}", (f, v), xytext=(-30, 4), textcoords="offset points", color=GREEN, fontsize=9, fontweight="bold")
ax[1].axhline(1, color="k", ls="--", lw=1); ax[1].text(10, 1.03, "an unlabeled volume worth an annotated one", fontsize=8)
ax[1].set_ylim(0, 1.15); ax[1].set_xticks(xs); ax[1].set_xlabel("Labeled fraction of the training pool (%)")
ax[1].set_ylabel(r"Recovery ratio $\Delta_{SSL}/\Delta_{labeled}$"); ax[1].set_title("(b) The share of annotation value recovered rises", loc="left")
ax[1].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "substitution_curve_corrected.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Fig 2: methods forest (Dice, corrected, 10 and 50)
rows = []
for f in (10, 50):
    sup = vec(arm(f"clean_sup_{f}"), K)
    for m, lab, col in [("mt", "Mean Teacher", PURPLE), ("cps", "CPS", GREEN), ("uamt", "UA-MT", ORANGE)]:
        d = vec(arm(f"clean_{m}_{f}"), K) - sup; rows.append((f"{lab}, {f}%", d.mean(), *ci_mean(d), col))
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, (lab, m, lo, hi, col) in enumerate(rows[::-1]):
    ax.plot([lo, hi], [i, i], color=col, lw=3); ax.plot(m, i, "o", color=col, ms=8); ax.text(max(hi, 0) + 0.004, i, M(f"{m:+.3f}"), va="center", fontsize=8.5)
ax.axhline(2.5, color=GREY, lw=0.6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel(r"$\Delta$ Dice vs. the matched supervised control (95% CI over patients)")
ax.set_title("Below control at 10%, about half of UA-MT's gain at 50%", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT + "method_forest_corrected.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Fig 3: boundary, raw vs filtered HD95 change
H = {}
for a in ["csup10", "cuamt10", "csup25", "cuamt25", "csup50", "cuamt50", "cmt50", "ccps50", "csup100"]:
    d = {}
    for f in glob.glob(f"baseline/logs/{'hd95_5fold_es30' if a == 'csup10' else 'hd95_5fold'}/partial_{a}_fold*_gaussian_lcc.jsonl"):  # 10% primary control = es30
        for l in open(f): rr = json.loads(l); d[rr["case"]] = rr
    H[a] = d
HK = sorted(H["csup50"])
def hv(a, k): return np.array([H[a][c][k] for c in HK])
pairs = [("cuamt10", "csup10", "UA-MT, 10%", ORANGE), ("cuamt25", "csup25", "UA-MT, 25%*", ORANGE), ("cuamt50", "csup50", "UA-MT, 50%", ORANGE),
         ("cmt50", "csup50", "Mean Teacher, 50%", PURPLE), ("ccps50", "csup50", "CPS, 50%", GREEN)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
for i, (a, b, lab, col) in enumerate(pairs[::-1]):
    d = hv(a, "dice") - hv(b, "dice"); lo, hi = ci_mean(d)
    ax[0].plot([lo, hi], [i, i], color=col, lw=3); ax[0].plot(d.mean(), i, "o", color=col, ms=7)
    for k, off, mk, fc in [("hd95", 0.15, "o", col), ("hd95_lcc", -0.15, "D", "white")]:
        d = hv(a, k) - hv(b, k); lo, hi = ci_mean(d)
        ax[1].plot([lo, hi], [i + off, i + off], color=col, lw=2.5, alpha=0.9 if k == "hd95" else 0.6)
        ax[1].plot(d.mean(), i + off, mk, color=col, mfc=fc, ms=7); ax[1].text(hi + 0.6, i + off, M(f"{d.mean():+.1f}"), va="center", fontsize=8)
for a_ in ax: a_.axvline(0, color="k", ls="--", lw=1); a_.axhline(1.5, color=GREY, lw=0.6)
ax[0].set_yticks(range(len(pairs))); ax[0].set_yticklabels([p[2] for p in pairs[::-1]]); ax[1].set_yticks([])
ax[0].set_xlabel(r"$\Delta$ Dice vs. matched control"); ax[0].set_title("(a) Overlap", loc="left")
ax[1].set_xlabel(r"$\Delta$ HD95 vs. matched control (mm; negative = better)")
ax[1].set_title("(b) Boundary: filled = raw output, open = largest component kept", loc="left")
fig.text(0.01, -0.02, "Bars are 95% percentile bootstrap over the 281 patients. *25%: one training seed (seed 42).", fontsize=8, color=GREY)
fig.tight_layout(); fig.savefig(OUT + "boundary_methods_corrected.png", dpi=300, bbox_inches="tight"); plt.close(fig)
# ---------------- Fig 4: TCIA bars
def tcia(a):
    d = {}
    for f in glob.glob(f"baseline/logs/tcia_crossdomain/partial_{a}_fold*_gaussian.jsonl"):
        for l in open(f): rr = json.loads(l); d.setdefault(rr["case"], []).append(rr["dice"])
    return np.mean([np.mean(v) for v in d.values()])
arms = [("csup25", "Supervised\n25%", BLUE), ("cuamt25", "UA-MT\n25%", ORANGE), ("csup50", "Supervised\n50%", BLUE), ("cuamt50", "UA-MT\n50%", ORANGE), ("csup100", "Supervised\nwhole pool", BLUE)]
fig, ax = plt.subplots(figsize=(9, 3.9)); w = 0.36
for i, (a, lab, col) in enumerate(arms):
    ind = hv(a, "dice").mean(); tc = tcia(a)
    ax.bar(i - w / 2, ind, w, color=col, edgecolor="k", lw=0.5); ax.bar(i + w / 2, tc, w, color=col, alpha=0.45, hatch="//", edgecolor="k", lw=0.5)
    ax.text(i - w / 2, ind + 0.01, f"{ind:.3f}", ha="center", fontsize=8, fontweight="bold"); ax.text(i + w / 2, tc + 0.01, f"{tc:.3f}", ha="center", fontsize=8)
    ax.text(i, 0.05, M(f"{tc - ind:+.3f}"), ha="center", color="#B00000", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms]); ax.set_ylabel("Dice"); ax.set_ylim(0, 0.85)
from matplotlib.patches import Patch
ax.legend([Patch(fc="grey", ec="k"), Patch(fc="grey", alpha=0.45, hatch="//", ec="k")], ["In domain (MSD, n=281)", "External (TCIA, n=80)"], frameon=False, fontsize=8, loc="upper left")
ax.set_title("Every arm loses 0.03 to 0.04 Dice; UA-MT's gain over its control transfers", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT + "crossdomain_corrected.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("done")
