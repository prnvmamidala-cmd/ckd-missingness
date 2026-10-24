#!/usr/bin/env python3
"""Main tables 2-3 and Appendix tables A1-A12 (CSV) from the aggregated v3 results. Table 1 and the complete-case vs excluded
table come from 02_descriptives.py. All values in percentage points (pp) unless stated; 95% CIs pool split and mask variability."""
import json
import numpy as np, pandas as pd
from config import RESULTS
S = json.load(open(RESULTS / "v3_summary.json")); K = S["keys"]
A = json.load(open(RESULTS / "v3_ablation_summary.json"))
E = json.load(open(RESULTS / "v3_external.json")); DC = json.load(open(RESULTS / "v3_dca.json"))
def num(x, d=1):
    s = f"{x:.{d}f}"; return "0." + "0" * d if s in ("-0." + "0" * d,) else s
def ci(v, d=1, sc=100): return f"{num(sc * v['est'], d)} ({num(sc * v['lo'], d)}, {num(sc * v['hi'], d)})"
SC = {"S1_cell": "Cell-level deletion (S1, confirmatory)", "S2_block": "Block deletion (S2, co-primary)",
      "S3_empirical": "Empirically calibrated NHANES patterns (S3, key sensitivity)"}
ML = {"mean": "Mean", "median": "Median", "knn": "KNN", "mice": "MICE (m = 10)", "xgb_native": "XGBoost native handling",
      "xgb_aware_native": "Aware training, native", "xgb_aware_ind": "Aware training, indicators"}
METH = ["mean", "median", "knn", "mice", "xgb_native"]
c = S["complete"]["xgb|sens80"]
# ---- Table 2 ----
rows = [["Complete data, FNR (%)", ci(c["FNR_low"]), ci(c["FNR_mid"]), ci(c["FNR_high"]), ""]]
for sc, lab in SC.items():
    rows.append([lab, "", "", "", ""])
    for m in METH:
        v = K[f"{sc}|xgb|{m}|sens80"]; rows.append(["    " + ML[m], ci(v["dFNR_low"]), ci(v["dFNR_mid"]), ci(v["dFNR_high"]), ci(v["gap"])])
pd.DataFrame(rows, columns=["Condition", "Low income", "Middle income", "High income", "Low minus high"]).to_csv(RESULTS / "table2_primary.csv", index=False)
# ---- Table 3 (sensitivity; low, high, gap) ----
r3 = []
def add(group, lab, v): r3.append([group, lab, ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"])])
g = "Controls: same overall missingness, no income dependence"
add(g, "Empirical patterns pooled (S4), mean", K["S4_emp_pooled|xgb|mean|sens80"])
for m in ["mean", "knn", "mice"]: add(g, f"Uniform 15% block (S6), {ML[m]}", K[f"S6_block_unif|xgb|{m}|sens80"])
g = "Age and sex also deletable (S7, v2-style)"
for m in ["mean", "knn", "mice"]: add(g, ML[m], K[f"S7_cell_all15|xgb|{m}|sens80"])
g = "Single block deleted at 30/10/5% (S8–S10), mean"
for s_, lab in [("S8_lab_only", "Lab block only"), ("S9_exam_only", "Exam block only"), ("S10_quest_only", "Questionnaire block only")]:
    add(g, lab, K[f"{s_}|xgb|mean|sens80"])
g = "Other model families, block deletion (S2)"
for mod, ml in [("lr", "Logistic regression"), ("rf", "Random forest")]:
    for m in ["mean", "knn", "mice"]: add(g, f"{ml}, {ML[m]}", K[f"S2_block|{mod}|{m}|sens80"])
g = "Alternative thresholds, XGBoost, block deletion (S2)"
for th, tl in [("youden", "Youden"), ("fpr20", "20% FPR")]:
    for m in ["mean", "knn", "mice"]: add(g, f"{tl}, {ML[m]}", K[f"S2_block|xgb|{m}|{th}"])
g = "Missingness-aware training, block deletion (S2)"
for m in ["xgb_aware_native", "xgb_aware_ind"]: add(g, ML[m].replace("Aware training, ", "Trained with missing values, "), K[f"S2_block|xgb|{m}|sens80"])
g = "Proxy ablation (uric acid and albumin removed), S2"
for m in ["mean", "knn", "mice"]: add(g, ML[m], A["keys"][f"S2_block|xgb|{m}|sens80"])
g = "External validation, NHANES 2021–2023"
for m in ["mean", "knn", "mice"]: add(g, f"Block (S2), {ML[m]}", E["scenarios"][f"S2_block|xgb|{m}|sens80"])
add(g, "Empirical (S3), mean", E["scenarios"]["S3_empirical|xgb|mean|sens80"])
pd.DataFrame(r3, columns=["Analysis", "Repair", "Low income", "High income", "Low minus high"]).to_csv(RESULTS / "table3_sensitivity.csv", index=False)

# ---------------- Appendix tables ----------------
AP = RESULTS / "appendix"; AP.mkdir(exist_ok=True)
# A1 empirical missingness
em = pd.read_csv(RESULTS / "empirical_missingness_by_income.csv")
NAMES = {"bmi": "BMI", "waist": "Waist", "sbp": "Systolic BP", "dbp": "Diastolic BP", "htn_told": "Hypertension history", "ever_smoke": "Ever smoked",
         "hba1c": "HbA1c", "LBXSGL": "Glucose", "LBXSUA": "Uric acid", "LBXSATSI": "ALT", "LBXSTR": "Triglycerides", "LBXSAL": "Albumin",
         "LBXTC": "Total cholesterol", "age": "Age", "female": "Female"}
a1 = em.set_index("Income group").T
a1.index = [NAMES.get(i, i) for i in a1.index]
a1 = a1.apply(lambda col: [f"{int(v):,}" if a1.index[k] == "N" else f"{float(v):.1f}" for k, v in enumerate(col)])
a1.columns = ["Low income", "Middle income", "High income"]; a1.index.name = "Feature or block (% missing, weighted)"
a1.to_csv(AP / "A1_empirical_missingness.csv")
pd.read_csv(RESULTS / "table_cc_vs_excluded.csv").to_csv(AP / "A2_included_vs_excluded.csv", index=False)
# A3 all scenarios, XGBoost, sens80
LAB = {"S1_cell": "S1 cell 30/10/5%", "S2_block": "S2 block 30/10/5% (ρ = 0.5)", "S3_empirical": "S3 empirical by income",
       "S4_emp_pooled": "S4 empirical pooled", "S5_cell_unif": "S5 cell uniform 15%", "S6_block_unif": "S6 block uniform 15%",
       "S7_cell_all15": "S7 cell, age/sex deletable", "S8_lab_only": "S8 lab block only", "S9_exam_only": "S9 exam block only",
       "S10_quest_only": "S10 questionnaire only"}
r = []
for sc, lab in LAB.items():
    for m in METH:
        v = K[f"{sc}|xgb|{m}|sens80"]
        r.append([lab, ML[m], ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"]), f"{100 * v['flag_low']:.1f}"])
pd.DataFrame(r, columns=["Scenario", "Repair", "ΔFNR low", "ΔFNR high", "Gap (low − high)", "ΔFPR low", "Low-income flagged (%)"]).to_csv(AP / "A3_all_scenarios.csv", index=False)
# A4 model families and thresholds (low, gap, dFPR low) for S1-S3
r = []
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for mod, th in [("xgb", "sens80"), ("lr", "sens80"), ("rf", "sens80"), ("xgb", "youden"), ("xgb", "fpr20"), ("lr", "youden"), ("rf", "youden")]:
        for m in ["mean", "median", "knn", "mice"]:
            v = K[f"{sc}|{mod}|{m}|{th}"]
            r.append([sc.split("_")[0], {"xgb": "XGBoost", "lr": "Logistic", "rf": "Random forest"}[mod], {"sens80": "80% sens.", "youden": "Youden", "fpr20": "20% FPR"}[th],
                      ML[m], ci(v["dFNR_low"]), ci(v["gap"]), ci(v["dFPR_low"])])
pd.DataFrame(r, columns=["Scenario", "Model", "Threshold", "Repair", "ΔFNR low", "Gap", "ΔFPR low"]).to_csv(AP / "A4_models_thresholds.csv", index=False)
# baseline per model/threshold
r = []
for mod in ["xgb", "lr", "rf"]:
    for th in ["sens80", "youden", "fpr20"]:
        v = S["complete"][f"{mod}|{th}"]; thr = np.mean(S["thresholds"][mod][th])
        r.append([mod, th, f"{thr:.3f}", ci(v["FNR_low"]), ci(v["FNR_high"]), ci(v["FNR_all"]), ci(v["FPR_all"])])
pd.DataFrame(r, columns=["Model", "Threshold", "Mean cut-point", "FNR low", "FNR high", "FNR all", "FPR all"]).to_csv(AP / "A4b_complete_baselines.csv", index=False)
# A5 aware training
r = []
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for m in ["xgb_aware_native", "xgb_aware_ind", "xgb_native", "mean"]:
        for th in ["sens80", "youden", "fpr20"]:
            v = K[f"{sc}|xgb|{m}|{th}"]
            r.append([sc.split("_")[0], ML[m], th, ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"]),
                      f"{S['auroc'][f'{sc}|xgb|{m}']['low']:.3f}"])
pd.DataFrame(r, columns=["Scenario", "Model/repair", "Threshold", "ΔFNR low", "ΔFNR high", "Gap", "ΔFPR low", "AUROC low"]).to_csv(AP / "A5_aware_training.csv", index=False)
# A6 proxy ablation
r = [["Complete data", "", f"{A['auroc']['complete|xgb']['low']:.3f} / {A['auroc']['complete|xgb']['all']:.3f}",
      f"{S['auroc']['complete|xgb']['low']:.3f} / {S['auroc']['complete|xgb']['all']:.3f}", "", ""]]
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for m in METH:
        for th in ["sens80", "youden"]:
            v = A["keys"][f"{sc}|xgb|{m}|{th}"]; v0 = K[f"{sc}|xgb|{m}|{th}"]
            r.append([f"{sc.split('_')[0]} {ML[m]}", th, f"{A['auroc'][f'{sc}|xgb|{m}']['low']:.3f}", f"{S['auroc'][f'{sc}|xgb|{m}']['low']:.3f}",
                      ci(v["dFNR_low"]) + " | gap " + ci(v["gap"]), ci(v0["dFNR_low"]) + " | gap " + ci(v0["gap"])])
pd.DataFrame(r, columns=["Condition", "Threshold", "AUROC low (13 features)", "AUROC low (15 features)", "ΔFNR low, 13 features", "ΔFNR low, 15 features (main)"]).to_csv(AP / "A6_proxy_ablation.csv", index=False)
# A7 external
r = []
for th in ["sens80", "youden", "fpr20"]:
    v = E["complete"][f"xgb|{th}"]; r.append(["Complete data (FNR, %)", "", th, ci(v["FNR_low"]), ci(v["FNR_high"]), "", ""])
for sc in ["S1_cell", "S2_block", "S3_empirical", "S4_emp_pooled", "S5_cell_unif", "S6_block_unif"]:
    for m in METH:
        k = f"{sc}|xgb|{m}|sens80"
        if k in E["scenarios"]:
            v = E["scenarios"][k]; r.append([LAB[sc], ML[m], "sens80", ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"])])
for mod in ["lr", "rf"]:
    for m in ["mean", "knn", "mice"]:
        k = f"S2_block|{mod}|{m}|sens80"
        if k in E["scenarios"]:
            v = E["scenarios"][k]; r.append([LAB["S2_block"], f"{mod.upper()} {ML[m]}", "sens80", ci(v["dFNR_low"]), ci(v["dFNR_high"]), ci(v["gap"]), ci(v["dFPR_low"])])
pd.DataFrame(r, columns=["Scenario", "Repair", "Threshold", "Low income", "High income", "Gap", "ΔFPR low"]).to_csv(AP / "A7_external_2021_2023.csv", index=False)
# A8 intersectional (S2 mean, MICE; S1 mean)
r = []
for t in ["low", "mid", "high"]:
    for dim, levels in [("sex", ["female", "male"]), ("age", ["20-44", "45-64", "65+"]), ("race", ["NHWhite", "NHBlack", "MexAm", "OtherHisp", "Other"])]:
        for lv in levels:
            key = f"{t}|{dim}|{lv}"; n = S["inter_n_mean"][key]
            comp = S["inter"].get(f"complete|{key}")
            r.append([t, dim, lv, f"{n:.0f}" + (" †" if n < 30 else ""), ci(comp) if comp else "",
                      ci(S["inter"][f"S2_block|mean|{key}"]), ci(S["inter"][f"S2_block|mice|{key}"]), ci(S["inter"][f"S1_cell|mean|{key}"])])
pd.DataFrame(r, columns=["Income", "Dimension", "Level", "Mean CKD cases per test set", "Complete FNR (%)", "ΔFNR S2 mean", "ΔFNR S2 MICE", "ΔFNR S1 mean"]).to_csv(AP / "A8_intersectional.csv", index=False)
# A9 mechanism
r = []
for key in ["S1_cell|mean", "S1_cell|knn", "S1_cell|mice", "S2_block|mean", "S2_block|knn", "S2_block|mice"]:
    M = S["mech"][key]
    for f in M["mean_dSHAP_flipped"]:
        if f in ("age", "female"): continue
        r.append([key.replace("_cell", "").replace("_block", ""), NAMES[f], M["n_flip_events"], f"{M['mean_dSHAP_flipped'][f]:.3f}", f"{M['mean_dSHAP_stayed'][f]:.3f}",
                  f"{M['mean_risk_signed_shift_deleted_flipped'].get(f, np.nan):.2f}", f"{M['mean_risk_signed_shift_deleted_stayed'].get(f, np.nan):.2f}"])
pd.DataFrame(r, columns=["Scenario|repair", "Feature", "Flip events", "ΔSHAP flipped", "ΔSHAP stayed flagged", "Risk-signed shift (SD), flipped", "Risk-signed shift (SD), stayed"]).to_csv(AP / "A9_mechanism.csv", index=False)
# A10 single feature / block deletion
r = []
sp = lambda v: f"{num(100 * v['est'])} ({num(100 * v['p2_5'])}, {num(100 * v['p97_5'])})"
for f in ["block:lab", "block:exam", "block:questionnaire"] + list(NAMES)[:13]:
    if f"{f}|mean" not in S["single"]: continue
    r.append([NAMES.get(f, f.replace("block:", "").capitalize() + " block (all features)"), sp(S["single"][f"{f}|mean"]), sp(S["single"][f"{f}|knn"]), sp(S["single"][f"{f}|mice"])])
pd.DataFrame(r, columns=["Deleted for every low-income test participant", "Mean (split 2.5–97.5%)", "KNN", "MICE"]).to_csv(AP / "A10_single_feature.csv", index=False)
# A11 split distribution and CI decomposition
r = []
for sc in ["S1_cell", "S2_block", "S3_empirical"]:
    for m in METH:
        v = K[f"{sc}|xgb|{m}|sens80"]; d = v["dFNR_low"]
        r.append([sc.split("_")[0], ML[m], ci(d), f"{100 * d['within_lo']:.1f} to {100 * d['within_hi']:.1f}", f"{100 * d['split_sd']:.1f}",
                  f"{100 * d['split_min']:.1f} to {100 * d['split_max']:.1f}", f"{100 * v['mask_sd_within_split']:.1f}", f"{100 * v['frac_masks_positive']:.0f}"])
pd.DataFrame(r, columns=["Scenario", "Repair", "ΔFNR low, pooled CI", "Within-split CI only", "Split SD", "Split range", "Mask SD within split", "Masks with ΔFNR > 0 (%)"]).to_csv(AP / "A11_split_distribution.csv", index=False)
# A12 decision curve
pt = DC["pt"]; sel = [0.05, 0.10, 0.15, 0.20, 0.30]; ix = [pt.index(x) for x in sel]
r = []
for t in ["low", "high"]:
    for key, lab in [("complete", "Complete data"), ("treat_all", "Flag everyone")] + [(f"S2_block|{m}", f"S2 {ML[m]}") for m in METH] + [(f"S1_cell|{m}", f"S1 {ML[m]}") for m in ["mean", "knn", "mice"]] + [("S3_empirical|mean", "S3 Mean")]:
        row = [t, lab] + [f"{DC['curves'][key][t]['mean'][i]:.3f}" for i in ix]
        if key in DC["diff_vs_complete"]:
            dd = DC["diff_vs_complete"][key][t]; row.append(f"{dd['mean'][pt.index(0.10)]:.3f} ({dd['p2.5'][pt.index(0.10)]:.3f}, {dd['p97.5'][pt.index(0.10)]:.3f})")
        else: row.append("")
        r.append(row)
pd.DataFrame(r, columns=["Income", "Condition"] + [f"NB at {x:.2f}" for x in sel] + ["ΔNB vs complete at 0.10 (split 2.5–97.5%)"]).to_csv(AP / "A12_decision_curve.csv", index=False)
# A13 AUROC
r = []
for key, v in S["auroc"].items(): r.append([key, f"{v['low']:.3f}", f"{v['mid']:.3f}", f"{v['high']:.3f}", f"{v['all']:.3f}"])
pd.DataFrame(r, columns=["Condition", "Low", "Middle", "High", "All"]).to_csv(AP / "A13_auroc.csv", index=False)
print("tables written")
