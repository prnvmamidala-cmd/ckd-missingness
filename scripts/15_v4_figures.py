#!/usr/bin/env python3
"""Figures 1-5 (v4). Figures 2-3 use v4_summary.json (hierarchical bootstrap, conservative 95% CIs).
Figures 4-5 are exploratory v3 analyses (point estimates averaged over 20 splits; SII = v3 'MICE').
Built at final print width (<= 6.5 in); text >= 12 pt; PNG/TIFF 600 dpi + PDF."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import RESULTS, FIGURES
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 12, "axes.labelsize": 14, "axes.labelweight": "bold",
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "white"})
S = json.load(open(RESULTS / "v4_summary.json")); K = S["keys"]
for _k, _v in json.load(open(RESULTS / "v4_summary_s0ce.json"))["keys"].items(): K.setdefault(_k, _v)
V3 = json.load(open(RESULTS / "v3_summary.json"))
OI = {"low": "#D55E00", "mid": "#009E73", "high": "#0072B2"}
GL = {"low": "Low income", "mid": "Middle income", "high": "High income"}
METH = ["mean", "median", "knn", "det", "sii", "native"]
ML = {"mean": "Mean", "median": "Median", "knn": "KNN", "det": "CE", "sii": "SII", "native": "XGBoost\nnative"}
MC = {"mean": "#E69F00", "median": "#56B4E9", "knn": "#009E73", "det": "#882255", "sii": "#CC79A7", "native": "#000000",
      "complete": "#0072B2", "treat_all": "#999999"}
W = 6.5
def letter(ax, s, x=-0.14, y=1.03): ax.text(x, y, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom")
def save(fig, name):
    for ext, kw in [("png", dict(dpi=600)), ("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"})), ("pdf", {})]:
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
# ---- Figure 1: design ----
fig, ax = plt.subplots(figsize=(W, 7.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 12.2); ax.axis("off")
def box(x, y, w, h, txt, fc="#F2F2F2"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12, linespacing=1.25)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333"))
dm = S["design_point"]
box(0.2, 10.8, 9.6, 1.3, "NHANES 2013–2018 adults with complete records (N = 12,546)\nCKD = eGFR < 60 or UACR ≥ 30 mg/g (single measurement)")
arrow(5, 10.8, 5, 10.35)
box(0.2, 9.0, 9.6, 1.3, f"20 repeated splits of whole survey strata\ntrain ≈ {dm['train_n_mean']:,.0f} | calibration ≈ {dm['cal_n_mean']:,.0f} | test ≈ {dm['test_n_mean']:,.0f}")
arrow(2.6, 9.0, 2.6, 8.55); arrow(7.4, 9.0, 7.4, 8.55)
box(0.2, 6.95, 4.8, 1.55, "Train: XGBoost (main),\nlogistic regression,\nrandom forest; imputers")
box(5.2, 6.95, 4.6, 1.55, "Calibration: threshold\n(80% sensitivity; Youden\nand 20% FPR sensitivity)")
arrow(5, 6.95, 5, 6.5)
box(0.2, 4.15, 9.6, 2.3, "Test set: 10 simulated deletion masks per split\nS0 (preregistered): all 15 features, 30 / 10 / 5% by income\nS1: as S0, age and sex protected  |  S2: block deletion\nS3: NHANES nonmeasurement patterns  |  S11: uniform 15%", fc="#FCE5D6")
arrow(5, 4.15, 5, 3.7)
box(0.2, 2.25, 9.6, 1.4, "Repair: mean, median, KNN, CE, SII, LCI, XGBoost native\nOutcome: change in FNR vs complete data, by income", fc="#DCEAF7")
arrow(5, 2.25, 5, 1.8)
box(0.2, 0.05, 9.6, 1.7, "95% CI: bootstrap of whole PSUs within strata (B = 120);\neach replicate redoes splits (4-fold stratum cross-fit) and\nmasks; percentile interval widened by √2 (2 PSUs per stratum)", fc="#E8F3E8")
save(fig, "Figure1")
# ---- Figure 2 ----
fig, axes = plt.subplots(4, 1, figsize=(W, 10.6), sharex=True)
for ax, sc, L in zip(axes, ["S7_cell_all15", "S1_cell", "S2_block", "S3_empirical"], "ABCD"):
    x = np.arange(len(METH)); bw = 0.26
    for j, t in enumerate(["low", "mid", "high"]):
        e = [K[f"{sc}|xgb|{m}|sens80|w"][f"dFNR_{t}"] for m in METH]
        est = np.array([v["est"] for v in e]) * 100; err = np.array([[v["est"] - v["lo"], v["hi"] - v["est"]] for v in e]).T * 100
        ax.bar(x + (j - 1) * bw, est, bw, color=OI[t], label=GL[t], yerr=err, capsize=2.5, error_kw=dict(lw=1.0))
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in METH])
    ax.set_ylabel("Change in\nFNR (pp)"); letter(ax, L, -0.17)
lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
for a in axes: a.set_ylim(lo, hi)
axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.03), handlelength=1.2, columnspacing=1.0)
axes[-1].set_xlabel("Repair method"); fig.tight_layout(); save(fig, "Figure2")
# ---- Figure 3: sensitivity forest ----
fig, axes = plt.subplots(1, 2, figsize=(W, 5.8), sharey=True, sharex=True)
rows = [("sens80", "w", "Main analysis"), ("sens80", "ipw", "Selection-\nweighted"), ("youden", "w", "Youden\nthreshold"), ("fpr20", "w", "20% FPR\nthreshold")]
meths = ["mean", "knn", "det", "sii"]
for ax, sc, L in zip(axes, ["S7_cell_all15", "S2_block"], "AB"):
    ypos = 0; yt, yl = [], []
    for th, ws, lab in rows:
        for j, m in enumerate(meths):
            k = f"{sc}|xgb|{m}|{th}|{ws}"
            if k not in K: continue
            v = K[k]["dFNR_low"]
            ax.errorbar(v["est"] * 100, ypos - j * 0.2, xerr=[[100 * (v["est"] - v["lo"])], [100 * (v["hi"] - v["est"])]], fmt="o",
                        color=MC[m], ms=5, capsize=2, lw=1.2, label=ML[m] if (ypos == 0 and L == "A") else None)
        yt.append(ypos - 0.3); yl.append(lab); ypos -= 1
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("Low-income change\nin FNR (pp)"); letter(ax, L, -0.05 if L == "B" else -0.55)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.6, 0.98), handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure3")
# ---- Figure 4: mechanism (v3, exploratory) ----
M = V3["mech"]; fe = list(M["S1_cell|mean"]["mean_dSHAP_flipped"].keys())
NAMES = {"bmi": "BMI", "waist": "Waist", "sbp": "Systolic BP", "dbp": "Diastolic BP", "htn_told": "Hypertension hx", "ever_smoke": "Ever smoked",
         "hba1c": "HbA1c", "LBXSGL": "Glucose", "LBXSUA": "Uric acid", "LBXSATSI": "ALT", "LBXSTR": "Triglycerides", "LBXSAL": "Albumin",
         "LBXTC": "Cholesterol", "age": "Age", "female": "Female"}
feats = [f for f in fe if f not in ("age", "female")]
fig, axes = plt.subplots(1, 2, figsize=(W, 5.6), sharey=True); y = np.arange(len(feats))
for j, (m, lab, col) in enumerate([("mean", "Mean", MC["mean"]), ("knn", "KNN", MC["knn"]), ("mice", "SII", MC["sii"])]):
    axes[0].barh(y + (j - 1) * 0.27, [M[f"S1_cell|{m}"]["mean_dSHAP_flipped"][f] for f in feats], 0.27, color=col, label=lab)
    axes[1].barh(y + (j - 1) * 0.27, [100 * V3["single"][f"{f}|{m}"]["est"] for f in feats], 0.27, color=col)
axes[0].axvline(0, color="black", lw=1); axes[0].set_yticks(y); axes[0].set_yticklabels([NAMES[f] for f in feats]); axes[0].invert_yaxis()
axes[0].set_xlabel("Change in SHAP\n(log-odds)"); letter(axes[0], "A", -0.6)
axes[1].axvline(0, color="black", lw=1); axes[1].set_xlabel("Low-income change\nin FNR (pp)"); letter(axes[1], "B", -0.08)
axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.06), handlelength=1.2, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure4")
# ---- Figure 5: decision curves (v3, exploratory) ----
DC = json.load(open(RESULTS / "v3_dca.json")); D = DC["curves"]; pt = np.array(DC["pt"])
fig, axes = plt.subplots(2, 1, figsize=(W, 7.6), sharex=True)
for ax, t, L in zip(axes, ["low", "high"], "AB"):
    ax.plot(pt, D["treat_all"][t]["mean"], color=MC["treat_all"], lw=1.5, ls=":", label="Flag everyone")
    ax.axhline(0, color="black", lw=1, label="Flag no one")
    ax.plot(pt, D["complete"][t]["mean"], color=MC["complete"], lw=2.5, label="Complete data")
    for m, lab, col in [("mean", "Mean", MC["mean"]), ("median", "Median", MC["median"]), ("knn", "KNN", MC["knn"]), ("mice", "SII", MC["sii"]), ("xgb_native", "XGBoost native", MC["native"])]:
        ax.plot(pt, D[f"S2_block|{m}"][t]["mean"], color=col, lw=1.6, ls="--" if m == "xgb_native" else "-", label=lab)
    ax.set_ylim(-0.02, max(D["complete"][t]["mean"]) * 1.2); ax.set_ylabel("Net benefit"); letter(ax, L)
axes[1].set_xlabel("Threshold probability"); axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.03), handlelength=1.6, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure5")
print("v4 figures saved")
