#!/usr/bin/env python3
"""Figures 1-5 (v3). Built at final print width (<= 6.5 in); text >= 12 pt; bold axis labels 14 pt; panel letters only;
no frames. PNG/TIFF 600 dpi + PDF."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from config import RESULTS, FIGURES
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 12, "axes.labelsize": 14, "axes.labelweight": "bold",
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.facecolor": "white"})
S = json.load(open(RESULTS / "v3_summary.json")); K = S["keys"]
OI = {"low": "#D55E00", "mid": "#009E73", "high": "#0072B2"}
GL = {"low": "Low income", "mid": "Middle income", "high": "High income"}
METH = ["mean", "median", "knn", "mice", "xgb_native"]
ML = {"mean": "Mean", "median": "Median", "knn": "KNN", "mice": "MICE\n(m = 10)", "xgb_native": "XGBoost\nnative",
      "xgb_aware_native": "Aware,\nnative", "xgb_aware_ind": "Aware,\nindicators"}
MC = {"mean": "#E69F00", "median": "#56B4E9", "knn": "#009E73", "mice": "#CC79A7", "xgb_native": "#000000",
      "complete": "#0072B2", "treat_all": "#999999"}
W = 6.5
def letter(ax, s, x=-0.14, y=1.03): ax.text(x, y, s, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom")
def save(fig, name):
    for ext, kw in [("png", dict(dpi=600)), ("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"})), ("pdf", {})]:
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
def bars(ax, sc, methods, model="xgb", th="sens80"):
    x = np.arange(len(methods)); bw = 0.26
    for j, t in enumerate(["low", "mid", "high"]):
        e = [K[f"{sc}|{model}|{m}|{th}"][f"dFNR_{t}"] for m in methods]
        est = np.array([v["est"] for v in e]) * 100; err = np.array([[v["est"] - v["lo"], v["hi"] - v["est"]] for v in e]).T * 100
        ax.bar(x + (j - 1) * bw, est, bw, color=OI[t], label=GL[t], yerr=err, capsize=3, error_kw=dict(lw=1.1))
    ax.axhline(0, color="black", lw=1); ax.set_xticks(x); ax.set_xticklabels([ML[m] for m in methods])

# ---- Figure 1: design ----
fig, ax = plt.subplots(figsize=(W, 6.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
def box(x, y, w, h, txt, fc="#F2F2F2", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333"))
dm = S["design_mean"]
box(0.2, 9.2, 9.6, 1.25, "NHANES 2013–2018 adults with complete records (N = 12,546)\nCKD = eGFR < 60 or UACR ≥ 30 mg/g (single measurement)")
arrow(5, 9.2, 5, 8.75)
box(0.2, 7.45, 9.6, 1.25, f"20 repeated splits of whole survey strata\ntrain ≈ {dm['train_n']:,.0f} | calibration ≈ {dm['cal_n']:,.0f} | test ≈ {dm['test_n']:,.0f} ({dm['test_psu']:.0f} PSUs)")
arrow(2.6, 7.45, 2.6, 7.0); arrow(7.4, 7.45, 7.4, 7.0)
box(0.2, 5.15, 4.8, 1.8, "Train: XGBoost, logistic\nregression, random forest\nand imputers", fc="#F2F2F2")
box(5.2, 5.15, 4.6, 1.8, "Calibration: thresholds\n(80% sensitivity, Youden,\n20% FPR)", fc="#F2F2F2")
arrow(5, 5.15, 5, 4.7)
box(0.2, 2.15, 9.6, 2.5, "Test set: 20 simulated deletion masks per scenario\n(age and sex never deleted)\nCell: 30 / 10 / 5% of values by income\nBlock: lab, exam, questionnaire blocks, 30 / 10 / 5%\nEmpirical: patterns observed in NHANES by income", fc="#FCE5D6")
arrow(5, 2.15, 5, 1.7)
box(0.2, 0.05, 9.6, 1.6, "Repair: mean, median, KNN, MICE (m = 10), XGBoost native,\nor missingness-aware training\nOutcome: change in FNR vs complete data, by income group", fc="#DCEAF7")
save(fig, "Figure1")

# ---- Figure 2: primary scenarios ----
fig, axes = plt.subplots(3, 1, figsize=(W, 9.0), sharex=True)
for ax, sc, L in zip(axes, ["S1_cell", "S2_block", "S3_empirical"], "ABC"):
    bars(ax, sc, METH); ax.set_ylabel("Change in FNR (pp)"); letter(ax, L)
lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
for a in axes: a.set_ylim(lo, hi)
axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.03), handlelength=1.2, columnspacing=1.0)
axes[-1].set_xlabel("Repair method"); fig.tight_layout(); save(fig, "Figure2")

# ---- Figure 3: robustness across model families and thresholds (low-income dFNR and gap, block scenario) ----
fig, axes = plt.subplots(1, 2, figsize=(W, 6.2), sharey=True, sharex=True)
rows = [("xgb", "sens80", "XGBoost, 80% sens."), ("lr", "sens80", "Logistic, 80% sens."), ("rf", "sens80", "Random forest, 80% sens."),
        ("xgb", "youden", "XGBoost, Youden"), ("xgb", "fpr20", "XGBoost, 20% FPR")]
meths = ["mean", "median", "knn", "mice"]
for ax, sc, L in zip(axes, ["S1_cell", "S2_block"], "AB"):
    ypos = 0; yt, yl = [], []
    for model, th, lab in rows:
        for j, m in enumerate(meths):
            v = K[f"{sc}|{model}|{m}|{th}"]["dFNR_low"]
            ax.errorbar(v["est"] * 100, ypos - j * 0.18, xerr=[[100 * (v["est"] - v["lo"])], [100 * (v["hi"] - v["est"])]], fmt="o",
                        color=MC[m], ms=5, capsize=2, lw=1.2, label=ML[m].replace("\n", " ") if (ypos == 0 and L == "A") else None)
        yt.append(ypos - 0.27); yl.append(lab); ypos -= 1
    ax.axvline(0, color="black", lw=1); ax.set_xlabel("Low-income change\nin FNR (pp)"); letter(ax, L, -0.05 if L == "B" else -0.75)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.6, 0.98), handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure3")

# ---- Figure 4: mechanism ----
M = S["mech"]; fe = list(M["S1_cell|mean"]["mean_dSHAP_flipped"].keys())
NAMES = {"bmi": "BMI", "waist": "Waist", "sbp": "Systolic BP", "dbp": "Diastolic BP", "htn_told": "Hypertension hx", "ever_smoke": "Ever smoked",
         "hba1c": "HbA1c", "LBXSGL": "Glucose", "LBXSUA": "Uric acid", "LBXSATSI": "ALT", "LBXSTR": "Triglycerides", "LBXSAL": "Albumin",
         "LBXTC": "Cholesterol", "age": "Age", "female": "Female"}
feats = [f for f in fe if f not in ("age", "female")]
fig, axes = plt.subplots(1, 2, figsize=(W, 5.6), sharey=True)
y = np.arange(len(feats))
for j, m in enumerate(["mean", "knn", "mice"]):
    v = [M[f"S1_cell|{m}"]["mean_dSHAP_flipped"][f] for f in feats]
    axes[0].barh(y + (j - 1) * 0.27, v, 0.27, color=MC[m], label=ML[m].replace("\n", " "))
axes[0].axvline(0, color="black", lw=1); axes[0].set_yticks(y); axes[0].set_yticklabels([NAMES[f] for f in feats]); axes[0].invert_yaxis()
axes[0].set_xlabel("Change in SHAP\n(log-odds)"); letter(axes[0], "A", -0.6)
SG = S["single"]
for j, m in enumerate(["mean", "knn", "mice"]):
    v = [100 * SG[f"{f}|{m}"]["est"] for f in feats]
    axes[1].barh(y + (j - 1) * 0.27, v, 0.27, color=MC[m])
axes[1].axvline(0, color="black", lw=1); axes[1].set_xlabel("Low-income change\nin FNR (pp)"); letter(axes[1], "B", -0.08)
axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.06), handlelength=1.2, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure4")

# ---- Figure 5: decision curves (block scenario; recalibrated probabilities from 07_decision_curve.py) ----
DC = json.load(open(RESULTS / "v3_dca.json")); D = DC["curves"]; pt = np.array(DC["pt"])
fig, axes = plt.subplots(2, 1, figsize=(W, 7.6), sharex=True)
for ax, t, L in zip(axes, ["low", "high"], "AB"):
    ax.plot(pt, D["treat_all"][t]["mean"], color=MC["treat_all"], lw=1.5, ls=":", label="Flag everyone")
    ax.axhline(0, color="black", lw=1, label="Flag no one")
    ax.plot(pt, D["complete"][t]["mean"], color=MC["complete"], lw=2.5, label="Complete data")
    for m in METH:
        ax.plot(pt, D[f"S2_block|{m}"][t]["mean"], color=MC[m], lw=1.6, ls="--" if m == "xgb_native" else "-", label=ML[m].replace("\n", " "))
    ax.set_ylim(-0.02, max(D["complete"][t]["mean"]) * 1.2); ax.set_ylabel("Net benefit"); letter(ax, L)
axes[1].set_xlabel("Threshold probability"); axes[0].legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.03), handlelength=1.6, columnspacing=0.8)
fig.tight_layout(); save(fig, "Figure5")
print("figures saved")
