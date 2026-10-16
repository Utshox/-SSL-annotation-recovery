"""Corrected-protocol LaTeX tables (2026-10-16), numbers read from result files.
Run from repo root -> manuscript_versions/argon_drafts/tables_corrected.tex"""
import glob, json, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, "baseline/code")
from collect_sept28 import load
rng = np.random.default_rng(0)
def ci(d, B=10000): b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
def f3(x): return f"{x:.3f}"
def s3(x): return f"${x:+.3f}$"
def s1(x): return f"${x:+.1f}$"
def pfmt(p): 
    if p < 1e-3: e = int(np.floor(np.log10(p))); return f"$<10^{{{e+1}}}$" if p < 1e-20 else f"${p/10**e:.0f}{{\\times}}10^{{{e}}}$"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
H = {}
for a in ["csup10","cuamt10","csup25","cuamt25","csup50","cuamt50","csup100","cmt50","ccps50"]:
    d = {}
    for f in glob.glob(f"baseline/logs/hd95_5fold/partial_{a}_fold*_gaussian_lcc.jsonl"):
        for l in open(f): r = json.loads(l); d[r["case"]] = r
    assert len(d) == 281, (a, len(d)); H[a] = d
K = sorted(H["csup50"])
def v(a, k): return np.array([H[a][c][k] for c in K])
T = []
# ---------- boundary table
T.append(r"""\begin{table}[!t]
\centering
\caption{Overlap and boundary metrics on the out-of-fold predictions, corrected protocol, all 281 volumes, paired per volume. HD95 and ASD in millimetres from each volume's own voxel spacing; lower is better. ``Filtered'' keeps the largest 3D connected component (26-connectivity) of every arm's prediction, control included. The 25\% rows use one training seed (seed 42, whose control was the weakest of three). Intervals are 95\% percentile bootstrap over patients; $p$ is the two-sided Wilcoxon signed-rank test on the paired per-volume differences.}
\label{tab:hd95}
\scriptsize
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Arm} & \textbf{Dice} & \textbf{HD95 mean} & \textbf{HD95 median} & \textbf{ASD} & \textbf{HD95 filtered} \\
\midrule""")
names = {"csup10":"Supervised, 10\\%","cuamt10":"UA-MT, 10\\%","csup25":"Supervised, 25\\%","cuamt25":"UA-MT, 25\\%","csup50":"Supervised, 50\\% (control)",
         "cmt50":"Mean Teacher, 50\\%","ccps50":"CPS, 50\\%","cuamt50":"UA-MT, 50\\%","csup100":"Supervised, whole pool"}
for a in ["csup10","cuamt10","csup25","cuamt25","csup50","cmt50","ccps50","cuamt50","csup100"]:
    T.append(f"{names[a]} & {f3(v(a,'dice').mean())} & {v(a,'hd95').mean():.1f} & {np.median(v(a,'hd95')):.1f} & {v(a,'asd').mean():.2f} & {v(a,'hd95_lcc').mean():.1f} \\\\")
    if a in ("cuamt10","cuamt25","csup100"): T.append(r"\midrule")
T.append(r"\multicolumn{6}{l}{\textit{Paired mean change from the matched control [95\% bootstrap CI]; Wilcoxon $p$ in parentheses}} \\")
T.append(r" & $\Delta$Dice & \multicolumn{2}{c}{$\Delta$HD95 (raw)} & $\Delta$ASD & $\Delta$HD95 (filtered) \\")
for a, b, lab in [("cuamt10","csup10","UA-MT, 10\\%"),("cuamt25","csup25","UA-MT, 25\\%"),("cuamt50","csup50","UA-MT, 50\\%"),("cmt50","csup50","Mean Teacher, 50\\%"),("ccps50","csup50","CPS, 50\\%")]:
    cells = []
    for k, nd in [("dice",3),("hd95",1),("asd",2),("hd95_lcc",1)]:
        d = v(a,k) - v(b,k); lo, hi = ci(d); p = wilcoxon(v(a,k), v(b,k)).pvalue
        f = lambda x: f"${x:+.{nd}f}$"
        cells.append(f"{f(d.mean())} [{f(lo)},\\,{f(hi)}] ({pfmt(p)})")
    T.append(f"{lab} & {cells[0]} & \\multicolumn{{2}}{{c}}{{{cells[1]}}} & {cells[2]} & {cells[3]} \\")
T.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
# ---------- multimethod table (Dice, from model JSONs)
def dice(*runs):
    ds = [load(r)[0] for r in runs]; ks = sorted(set.intersection(*[set(d) for d in ds]))
    return ks, np.array([np.mean([d[k] for d in ds]) for k in ks])
_, sup100 = dice("clean_sup_100")
T.append(r"""\begin{table}[!t]
\centering
\caption{Three semi-supervised methods against the same matched control, corrected protocol, five-fold cross-validation over all 281 volumes, one training seed, one recipe shared across methods (not tuned per method). At 10\\% all three methods are compared with the control trained under their own stopping rule (early stopping from the first epoch, \\cSupTenEsZero{}); the primary 10\\% control of Table~\\ref{tab:substitution} starts early stopping at epoch 30 (\\cSupTen{}), against which UA-MT gains \\cDTen{}. Intervals are 95\% percentile bootstrap over patients. Improved counts volumes where the arm beats its control. The ratio is $\Delta_{\text{SSL}}$ divided by the gain from annotating the same volumes.}
\label{tab:multimethod}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
\textbf{Labels} & \textbf{Method} & \textbf{Dice} & $\Delta_{\text{SSL}}$ & \textbf{95\% CI} & \textbf{Improved} & \textbf{Ratio} \\
\midrule""")
for f, n in [(10, 20), (50, 101)]:
    _, sup = dice(f"clean_sup_{f}")
    T.append(f"{f}\\% ({n}) & Supervised (control) & {f3(sup.mean())} & --- & --- & --- & --- \\\\")
    for m, lab in [("mt","Mean Teacher \\cite{tarvainen2017mean}"),("cps","Cross-pseudo supervision \\cite{cps}"),("uamt","UA-MT \\cite{uamt}")]:
        _, x = dice(f"clean_{m}_{f}"); d = x - sup; lo, hi = ci(d)
        T.append(f" & {lab} & {f3(x.mean())} & {s3(d.mean())} & [{s3(lo)},\\,{s3(hi)}] & {int((d>0).sum())}/281 & ${d.mean()/(sup100-sup).mean():.2f}$ \\\\")
    T.append(r"\midrule")
T.append(f"100\\% (202) & Supervised & {f3(sup100.mean())} & --- & --- & --- & 1.00 \\\\")
T.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
# ---------- TCIA table
def tcia(a):
    d = {}
    for f in glob.glob(f"baseline/logs/tcia_crossdomain/partial_{a}_fold*_gaussian.jsonl"):
        for l in open(f): r = json.loads(l); d.setdefault(r["case"], []).append(r["dice"])
    assert len(d) == 80 and all(len(x) == 5 for x in d.values()), a
    return {k: np.mean(x) for k, x in d.items()}
Tc = {a: tcia(a) for a in ["csup25","cuamt25","csup50","cuamt50","csup100"]}
ck = sorted(Tc["csup50"])
T.append(r"""\begin{table}[!t]
\centering
\caption{Zero-shot Dice on the external TCIA cohort ($N=80$), corrected protocol, averaged over the five fold checkpoints scored in domain (25\%: training seed 42). The change column is each arm's TCIA Dice minus its own in-domain Dice, the quantity a robustness claim is about. $\Delta$ vs.\ control is paired over the 80 TCIA volumes (95\% CI; Wilcoxon $p$).}
\label{tab:tcia_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{In-domain} & \textbf{TCIA} & \textbf{Change} & $\Delta$ \textbf{vs.\ control (TCIA)} \\
\midrule""")
for a, b, lab in [("csup25",None,"Supervised, 25\\% (control)"),("cuamt25","csup25","UA-MT, 25\\%"),("csup50",None,"Supervised, 50\\% (control)"),("cuamt50","csup50","UA-MT, 50\\%"),("csup100",None,"Supervised, whole pool")]:
    ind = v(a,"dice").mean(); tc = np.mean([Tc[a][k] for k in ck])
    if b:
        x = np.array([Tc[a][k] for k in ck]); y = np.array([Tc[b][k] for k in ck]); d = x - y; lo, hi = ci(d)
        last = f"{s3(d.mean())} [{s3(lo)},\\,{s3(hi)}] ({pfmt(wilcoxon(x,y).pvalue)})"
    else: last = "---"
    T.append(f"{lab} & {f3(ind)} & {f3(tc)} & {s3(tc-ind)} & {last} \\\\")
x = np.array([Tc["cuamt25"][k] for k in ck]); y = np.array([Tc["csup50"][k] for k in ck]); d = x - y; lo, hi = ci(d)
T.append(r"\midrule" + f"\nUA-MT 25\\% vs.\\ supervised 50\\% & & & & {s3(d.mean())} [{s3(lo)},\\,{s3(hi)}] ({pfmt(wilcoxon(x,y).pvalue)}) \\\\")
T.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
open("manuscript_versions/argon_drafts/tables_corrected.tex", "w").write("\n".join(T))
print("\n".join(T))
