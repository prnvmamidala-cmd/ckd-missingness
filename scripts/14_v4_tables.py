#!/usr/bin/env python3
"""v4 main tables 2-3 and Appendix tables (CSV). Values in percentage points (pp).
Intervals: 'conservative' 95% CI from the hierarchical PSU bootstrap (13_v4_aggregate.py) unless marked.
Rows without a bootstrap interval show the point estimate with the range over the 20 split estimates in square brackets.
v3-derived exploratory analyses (S4-S10, aware training, ablation, intersectional, mechanism, single-feature, decision curves)
are point estimates over 20 splits with split ranges; their v3 pooled CIs were withdrawn (they double-counted participants)."""
import json, sys
import numpy as np, pandas as pd
from config import RESULTS
from decision_timeline import ROWS as TIMELINE
S = json.load(open(RESULTS / "v4_summary.json")); K = S["keys"]
for _k, _v in json.load(open(RESULTS / "v4_summary_s0ce.json"))["keys"].items(): K.setdefault(_k, _v)  # CE with IPW / other thresholds (S0)
H = json.load(open(RESULTS / "v4_summary_h2.json"))["keys"]
V3 = json.load(open(RESULTS / "v3_summary.json")); K3 = V3["keys"]
A = json.load(open(RESULTS / "v3_ablation_summary.json"))
E = json.load(open(RESULTS / "v3_external.json")); DC = json.load(open(RESULTS / "v3_dca.json"))
COV = json.load(open(RESULTS / "v4_coverage_sim.json"))
AP = RESULTS / "appendix"; AP.mkdir(exist_ok=True)
def num(x, d=1):
    s = f"{x:.{d}f}"; return "0." + "0" * d if s == "-0." + "0" * d else s
def ci(v, d=1, sc=100, kind=""):
    lo, hi = v.get(kind + "lo" if kind else "lo"), v.get(kind + "hi" if kind else "hi")
    if lo is None or hi is None or not np.isfinite(lo):
        return f"{num(sc * v['est'], d)} [{num(sc * v['split_min'], d)} to {num(sc * v['split_max'], d)}]"
    return f"{num(sc * v['est'], d)} ({num(sc * lo, d)}, {num(sc * hi, d)})"
def ci_kind(v, kind): return f"{num(100 * v['est'])} ({num(100 * v[kind + '_lo'])}, {num(100 * v[kind + '_hi'])})" if np.isfinite(v.get(kind + "_lo", np.nan)) else "n/a"
def sr(v, sc=100): return f"{num(sc * v['est'])} [{num(sc * v['split_min'])} to {num(sc * v['split_max'])}]"
SC = {"S7_cell_all15": "S0 preregistered design: cell deletion of all 15 features (H1 confirmatory for KNN and CE)",
      "S1_cell": "S1 cell deletion, age and sex protected (exploratory)", "S2_block": "S2 block deletion (exploratory)",
      "S3_empirical": "S3 empirically calibrated NHANES nonmeasurement patterns (exploratory)"}
SH = {"S7_cell_all15": "S0", "S1_cell": "S1", "S2_block": "S2", "S3_empirical": "S3"}
ML = {"mean": "Mean", "median": "Median", "knn": "KNN", "det": "Chained-equation (CE; preregistered 'MICE')",
      "sii": "Stochastic iterative imputation (SII)", "sii_frac": "SII, fractional binaries", "lci": "Logistic chained imputation (LCI)",
      "native": "XGBoost native handling"}
ML3 = {"mean": "Mean", "median": "Median", "knn": "KNN", "mice": "SII", "xgb_native": "XGBoost native",
       "xgb_aware_native": "Aware training, native", "xgb_aware_ind": "Aware training, indicators"}
METH = ["mean", "median", "knn", "det", "sii", "native"]
c = S["complete"]["xgb|sens80|w"]
# ---------------- Table 2 ----------------
rows = [["Complete data, FNR (%)", ci(c["FNR_low"]), ci(c["FNR_mid"]), ci(c["FNR_high"]), ""]]
for sc, lab in SC.items():
    rows.append([lab, "", "", "", ""])
    order = ["knn", "det", "mean", "median", "sii", "native"] if sc == "S7_cell_all15" else METH
    for m in order:
        v = K[f"{sc}|xgb|{m}|sens80|w"]
        tag = " *" if sc == "S7_cell_all15" and m in ("knn", "det") else ""
        rows.append(["    " + ML[m] + tag, ci(v["dFNR_low"]), ci(v["dFNR_mid"]), ci(v["dFNR_high"]), ci(v["gap"])])
pd.DataFrame(rows, columns=["Condition", "Low income", "Middle income", "High income", "Low minus high"]).to_csv(RESULTS / "table2_primary.csv", index=False)
# ---------------- Table 3 ----------------
r3 = []
def add(g, lab, v): r3.append([g, lab, ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"])])
g = "H2 control (prespecified): uniform 15% cell deletion, all 15 features (S11)"
for m in ["knn", "det", "mean", "median", "native"]: add(g, ML[m].replace(" (CE; preregistered 'MICE')", ""), H[f"S11_cell_unif_all15|xgb|{m}|sens80|w"])
g = "Selection weighting (inverse probability of complete case)"
for sc, m in [("S7_cell_all15", "knn"), ("S7_cell_all15", "det"), ("S2_block", "mean"), ("S2_block", "sii"), ("S3_empirical", "mean")]:
    add(g, f"{SH[sc]}, {ML[m].split(' (')[0]}", K[f"{sc}|xgb|{m}|sens80|ipw"])
g = "Binary-variable handling, S0 and S2"
for sc in ["S7_cell_all15", "S2_block"]:
    for m in ["sii", "sii_frac", "lci"]: add(g, f"{SH[sc]}, {ML[m].replace('Stochastic iterative imputation (SII)', 'SII, rounded binaries (main)')}", K[f"{sc}|xgb|{m}|sens80|w"])
g = "Other thresholds and model families, S2"
for th, tl in [("youden", "Youden"), ("fpr20", "20% FPR")]:
    for m in ["mean", "knn", "sii"]: add(g, f"XGBoost {tl}, {ML[m].split(' (')[0]}", K[f"S2_block|xgb|{m}|{th}|w"])
for mod, tl in [("lr", "Logistic regression"), ("rf", "Random forest")]:
    for m in ["mean", "knn", "sii"]: add(g, f"{tl}, {ML[m].split(' (')[0]}", K[f"S2_block|{mod}|{m}|sens80|w"])
g = "External validation, NHANES 2021–2023 (v3; CI conditional on the fitted model)"
for m in ["mean", "knn", "mice"]: add(g, f"S2, {ML3[m]}", E["scenarios"][f"S2_block|xgb|{m}|sens80"])
add(g, "S3, Mean", E["scenarios"]["S3_empirical|xgb|mean|sens80"])
pd.DataFrame(r3, columns=["Analysis", "Repair", "Low income", "High income", "Low minus high"]).to_csv(RESULTS / "table3_sensitivity.csv", index=False)

# ---------------- Appendix ----------------
pd.DataFrame(TIMELINE, columns=["Decision", "When made (2026, PT)", "Results seen before the decision", "Label"]).to_csv(AP / "A00_decision_timeline.csv", index=False)
# A2 selection balance (A1 empirical missingness, A2_included_vs_excluded kept from v3 tables script)
pd.read_csv(RESULTS / "table_selection_balance.csv").to_csv(AP / "A02b_selection_balance.csv", index=False)
SM = json.load(open(RESULTS / "selection_model.json"))
pd.DataFrame([["Eligible interviewed adults with recorded income", f"{SM['n_eligible_with_PIR']:,}"], ["Included complete cases", f"{SM['n_included']:,}"],
              ["Selection model AUROC (interview-weighted)", f"{SM['selection_auc_weighted']:.3f}"],
              ["Predicted inclusion probability among included", f"{SM['p_hat_range_included'][0]:.3f} to {SM['p_hat_range_included'][1]:.3f}"],
              ["Weights truncated at 99th percentile", f"{SM['n_truncated']}"], ["Effective sample size, MEC weights", f"{SM['ess_mec_weights']:,.0f}"],
              ["Effective sample size, selection weights", f"{SM['ess_ipw_weights']:,.0f}"]] +
             [[f"Coefficient: {k}", f"{v:.3f}"] for k, v in SM["coefficients"].items()], columns=["Quantity", "Value"]).to_csv(AP / "A02c_selection_model.csv", index=False)
# A5 CI comparison (v3 pooled vs v4)
r = []
v3n = {"sii": "mice", "native": "xgb_native"}
for sc in SC:
    for m in METH:
        v = K[f"{sc}|xgb|{m}|sens80|w"]
        for st, sl in [("dFNR_low", "ΔFNR low"), ("gap", "Gap")]:
            d = v[st]; v3s = d.get("v3") or {}
            v3txt = f"{num(100 * v3s['est'])} ({num(100 * v3s['lo'])}, {num(100 * v3s['hi'])})" if v3s.get("est") is not None and np.isfinite(v3s.get("lo", np.nan)) else "not in v3"
            r.append([SH[sc], ML[m].split(" (")[0], sl, v3txt, ci(d), ci_kind(d, "naive"), ci_kind(d, "corr"),
                      f"{100 * (d['hi'] - d['lo']):.1f}", f"{100 * (v3s['hi'] - v3s['lo']):.1f}" if v3txt != "not in v3" else ""])
pd.DataFrame(r, columns=["Scenario", "Repair", "Statistic", "v3 pooled CI (withdrawn)", "v4 conservative CI (primary)", "v4 naive percentile", "v4 noise-corrected (Wald)",
                         "v4 width", "v3 width"]).to_csv(AP / "A05_ci_comparison.csv", index=False)
# A6 coverage simulation
r = []
for kname, kl in [("naive", "Naive percentile"), ("cons", "Conservative √2-rescaled percentile (primary)"), ("corr", "Noise-corrected Wald")]:
    r.append([kl] + sum([[f"{COV['coverage'][kname][st]:.3f}", f"{COV['mean_width_pp'][kname][st]:.2f}"] for st in ["dFNR_low", "gap"]], []))
D_ = COV["diagnostics_pp"]
r.append([f"Truth (superpopulation): ΔFNR low {COV['truth_dFNR_low_pp']:.2f} pp, gap {COV['truth_gap_pp']:.2f} pp; R = {COV['R']} samples, B = {COV['B']}; Monte Carlo SE of coverage at 0.95 = {COV['mc_se_of_coverage_at_0.95']:.3f}", "", "", "", ""])
r.append([f"Bias of estimate {D_['mean_estimate_minus_truth'][0]:.2f} / {D_['mean_estimate_minus_truth'][1]:.2f} pp; empirical SD {D_['empirical_sd_of_estimate'][0]:.2f} / {D_['empirical_sd_of_estimate'][1]:.2f}; mean √2 × bootstrap SD {D_['mean_sqrt2_boot_sd'][0]:.2f} / {D_['mean_sqrt2_boot_sd'][1]:.2f}; mean noise-corrected SE {D_['mean_corrected_se'][0]:.2f} / {D_['mean_corrected_se'][1]:.2f} (ΔFNR low / gap)", "", "", "", ""])
pd.DataFrame(r, columns=["Interval", "Coverage, ΔFNR low", "Mean width (pp), ΔFNR low", "Coverage, gap", "Mean width (pp), gap"]).to_csv(AP / "A06_coverage_sim.csv", index=False)
# A7 all v4 XGBoost results
r = []
for k, v in K.items():
    sc, mod, m, th, ws = k.split("|")
    r.append([SH.get(sc, sc), mod, ML.get(m, m).split(" (")[0], th, "selection" if ws == "ipw" else "MEC", ci(v["dFNR_low"]), ci(v["dFNR_mid"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"]), ci(v["dFPR_high"])])
for k, v in H.items():
    sc, mod, m, th, ws = k.split("|")
    r.append(["S11 (H2 control)", mod, ML.get(m, m).split(" (")[0], th, "MEC", ci(v["dFNR_low"]), ci(v["dFNR_mid"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"]), ci(v["dFPR_high"])])
pd.DataFrame(r, columns=["Scenario", "Model", "Repair", "Threshold", "Weights", "ΔFNR low", "ΔFNR mid", "ΔFNR high", "Gap", "ΔFPR low", "ΔFPR high"]).to_csv(AP / "A07_all_v4_results.csv", index=False)
# A7b complete-data baselines
r = []
for k, v in S["complete"].items():
    r.append([k] + [ci(v[f]) for f in ["FNR_low", "FNR_mid", "FNR_high", "FNR_all"]])
pd.DataFrame(r, columns=["Model|threshold|weights", "FNR low", "FNR mid", "FNR high", "FNR all"]).to_csv(AP / "A07b_complete_baselines.csv", index=False)
# A10 other v3 scenarios (point estimates, split ranges)
LAB = {"S4_emp_pooled": "S4 empirical pooled", "S5_cell_unif": "S5 cell uniform 15% (age/sex protected)", "S6_block_unif": "S6 block uniform 15%",
       "S8_lab_only": "S8 lab block only", "S9_exam_only": "S9 exam block only", "S10_quest_only": "S10 questionnaire only"}
r = []
for sc, lab in LAB.items():
    for m in ["mean", "median", "knn", "mice", "xgb_native"]:
        v = K3[f"{sc}|xgb|{m}|sens80"]; r.append([lab, ML3[m], sr(v["dFNR_low"]), sr(v["dFNR_high"]), sr(v["gap"])])
pd.DataFrame(r, columns=["Scenario (v3, exploratory)", "Repair", "ΔFNR low [split range]", "ΔFNR high [split range]", "Gap [split range]"]).to_csv(AP / "A10_other_scenarios_v3.csv", index=False)
# A11 aware training (v3)
r = []
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for m in ["xgb_aware_native", "xgb_aware_ind", "xgb_native", "mean"]:
        v = K3[f"{sc}|xgb|{m}|sens80"]; r.append([sc.split("_")[0], ML3[m], sr(v["dFNR_low"]), sr(v["gap"]), sr(v["dFPR_low"]), f"{V3['auroc'][f'{sc}|xgb|{m}']['low']:.3f}"])
pd.DataFrame(r, columns=["Scenario", "Model/repair", "ΔFNR low [split range]", "Gap [split range]", "ΔFPR low [split range]", "AUROC low"]).to_csv(AP / "A11_aware_training_v3.csv", index=False)
# A12 proxy ablation (v3)
r = []
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for m in ["mean", "median", "knn", "mice"]:
        v = A["keys"][f"{sc}|xgb|{m}|sens80"]; v0 = K3[f"{sc}|xgb|{m}|sens80"]
        r.append([sc.split("_")[0], ML3[m], sr(v["dFNR_low"]), sr(v0["dFNR_low"]), f"{A['auroc'][f'{sc}|xgb|{m}']['low']:.3f}"])
pd.DataFrame(r, columns=["Scenario", "Repair", "ΔFNR low, 13 features [split range]", "ΔFNR low, 15 features [split range]", "AUROC low, 13 features"]).to_csv(AP / "A12_proxy_ablation_v3.csv", index=False)
# A13 external (v3, conditional CIs)
r = []
for th in ["sens80", "youden", "fpr20"]:
    v = E["complete"][f"xgb|{th}"]; r.append(["Complete data (FNR, %)", "", th, ci(v["FNR_low"]), ci(v["FNR_high"]), ""])
for sc in ["S1_cell", "S2_block", "S3_empirical", "S4_emp_pooled", "S5_cell_unif", "S6_block_unif"]:
    for m in ["mean", "median", "knn", "mice", "xgb_native"]:
        k = f"{sc}|xgb|{m}|sens80"
        if k in E["scenarios"]:
            v = E["scenarios"][k]; r.append([sc.split("_")[0], ML3[m], "sens80", ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"])])
pd.DataFrame(r, columns=["Scenario", "Repair", "Threshold", "Low income", "High income", "Gap"]).to_csv(AP / "A13_external_2021_2023.csv", index=False)
# A14 intersectional (v3, point + split range)
r = []
for t in ["low", "mid", "high"]:
    for dim, levels in [("sex", ["female", "male"]), ("age", ["20-44", "45-64", "65+"]), ("race", ["NHWhite", "NHBlack", "MexAm", "OtherHisp", "Other"])]:
        for lv in levels:
            key = f"{t}|{dim}|{lv}"; n = V3["inter_n_mean"][key]; comp = V3["inter"].get(f"complete|{key}")
            r.append([t, dim, lv, f"{n:.0f}" + (" †" if n < 30 else ""), sr(comp) if comp else "", sr(V3["inter"][f"S2_block|mean|{key}"]), sr(V3["inter"][f"S2_block|mice|{key}"]), sr(V3["inter"][f"S1_cell|mean|{key}"])])
pd.DataFrame(r, columns=["Income", "Dimension", "Level", "Mean CKD cases per test set", "Complete FNR (%)", "ΔFNR S2 mean", "ΔFNR S2 SII", "ΔFNR S1 mean"]).to_csv(AP / "A14_intersectional_v3.csv", index=False)
# A16 single-feature (rename)
sf = pd.read_csv(AP / "A10_single_feature.csv"); sf.columns = [c_.replace("MICE", "SII") for c_ in sf.columns]; sf.to_csv(AP / "A16_single_feature_v3.csv", index=False)
mech = pd.read_csv(AP / "A9_mechanism.csv"); mech["Scenario|repair"] = mech["Scenario|repair"].str.replace("mice", "SII"); mech.to_csv(AP / "A15_mechanism_v3.csv", index=False)
dca = pd.read_csv(AP / "A12_decision_curve.csv"); dca["Condition"] = dca["Condition"].str.replace("MICE", "SII"); dca.to_csv(AP / "A17_decision_curve_v3.csv", index=False)
au = pd.read_csv(AP / "A13_auroc.csv"); au["Condition"] = au["Condition"].str.replace("mice", "SII"); au.to_csv(AP / "A18_auroc_v3.csv", index=False)
print("v4 tables written")
