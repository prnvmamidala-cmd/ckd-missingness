#!/usr/bin/env python3
"""Descriptive analyses (v3): (1) Table 1 cohort by income; (2) complete-case vs excluded adults; (3) observed
(empirical) missingness by income group among ALL interviewed adults, per feature and per block (survey-weighted)."""
import json, numpy as np, pandas as pd
from config import RESULTS, FEATS, BLOCKS, CYCLES
from common import load_all, load_cohort, DELETABLE, wilson
RESULTS.mkdir(exist_ok=True)
df, flow = load_cohort(return_flow=True)
allp = load_all()
elig = allp[(allp.age >= 20) & (allp.pregnant != 1)].copy()
inc = set(df.SEQN.astype(int))
elig["included"] = elig.SEQN.astype(int).isin(inc)
flow["eligible_interviewed_adults"] = len(elig)
# reasons for exclusion (first failing step)
ex = elig[~elig.included]
reason = np.select([ex.examined != 1, ex.ckd.isna(), ex[FEATS].isna().any(axis=1), ex.pir_tier.isna()],
                   ["not examined", "no creatinine or UACR", "missing >=1 of 15 features", "missing PIR"], "other")
flow["excluded_reasons"] = pd.Series(reason).value_counts().to_dict()
wv = lambda s, w: np.average(s[s.notna()], weights=w[s.notna()]) if s.notna().any() else np.nan
rows = []
def col(d):
    w = d.WTINT2YR
    c = {"N": f"{len(d):,}", "Age, years (weighted mean)": f"{wv(d.age, w):.1f}", "Female, %": f"{100*wv(d.female, w):.1f}"}
    for lab, rc in [("Non-Hispanic White", "NHWhite"), ("Non-Hispanic Black", "NHBlack"), ("Mexican American", "MexAm"),
                    ("Other Hispanic", "OtherHisp"), ("Other/multiracial", "Other")]:
        c[f"  {lab}, %"] = f"{100*wv((d.race5 == rc).astype(float), w):.1f}"
    for lab, t in [("Low income (PIR < 1.3), %", "low"), ("Middle income, %", "mid"), ("High income (PIR ≥ 3.5), %", "high")]:
        c[lab] = f"{100*wv((d.pir_tier == t).astype(float), w):.1f}"
    c["PIR missing, %"] = f"{100*wv(d.pir_tier.isna().astype(float), w):.1f}"
    c["Examined at MEC, %"] = f"{100*wv(d.examined.astype(float), w):.1f}"
    c["Told had hypertension, %*"] = f"{100*wv(d.htn_told, w):.1f}"
    c["Ever smoked, %*"] = f"{100*wv(d.ever_smoke, w):.1f}"
    c["BMI, kg/m² (weighted mean)*"] = f"{wv(d.bmi, w):.1f}"
    c["HbA1c, % (weighted mean)*"] = f"{wv(d.hba1c, w):.2f}"
    c["CKD, % (if both labs measured)*"] = f"{100*wv(d.ckd, w):.1f}"
    c["  n with both kidney labs"] = f"{int(d.ckd.notna().sum()):,}"
    return c
tab = pd.DataFrame({"Included (complete cases)": col(elig[elig.included]), "Excluded": col(elig[~elig.included])})
tab.index.name = "Characteristic"; tab.to_csv(RESULTS / "table_cc_vs_excluded.csv")
# weighted CKD difference with simple CI (descriptive)
# empirical missingness by PIR group (all interviewed adults with PIR)
e = elig[elig.pir_tier.notna()]
mrows = []
for t in ["low", "mid", "high"]:
    s = e[e.pir_tier == t]; w = s.WTINT2YR
    r = {"Income group": t, "N": len(s), "Not examined": 100 * wv(1 - s.examined, w)}
    for f in DELETABLE: r[f] = 100 * wv(s[f].isna().astype(float), w)
    for b, fs in BLOCKS.items():
        r[f"{b} block: any missing"] = 100 * wv(s[fs].isna().any(axis=1).astype(float), w)
        r[f"{b} block: all missing"] = 100 * wv(s[fs].isna().all(axis=1).astype(float), w)
    r["Any of 13 features missing"] = 100 * wv(s[DELETABLE].isna().any(axis=1).astype(float), w)
    mrows.append(r)
emp = pd.DataFrame(mrows); emp.to_csv(RESULTS / "empirical_missingness_by_income.csv", index=False)
# among examined adults only (missingness within the exam)
mrows2 = []
for t in ["low", "mid", "high"]:
    s = e[(e.pir_tier == t) & (e.examined == 1)]; w = s.WTMEC2YR
    r = {"Income group": t, "N": len(s)}
    for b, fs in BLOCKS.items(): r[f"{b} block: any missing"] = 100 * wv(s[fs].isna().any(axis=1).astype(float), w)
    r["Any of 13 features missing"] = 100 * wv(s[DELETABLE].isna().any(axis=1).astype(float), w)
    mrows2.append(r)
pd.DataFrame(mrows2).to_csv(RESULTS / "empirical_missingness_examined_only.csv", index=False)
# Table 1 (cohort by income)
cont = [("age", "Age, years"), ("bmi", "BMI, kg/m²"), ("waist", "Waist circumference, cm"), ("sbp", "Systolic BP, mmHg"),
        ("dbp", "Diastolic BP, mmHg"), ("hba1c", "HbA1c, %"), ("LBXSGL", "Serum glucose, mg/dL"), ("LBXSUA", "Uric acid, mg/dL"),
        ("LBXSATSI", "ALT, U/L"), ("LBXSTR", "Triglycerides, mg/dL"), ("LBXSAL", "Albumin, g/dL"), ("LBXTC", "Total cholesterol, mg/dL")]
g = df.pir_tier.to_numpy(); grp = [("Overall", np.ones(len(df), bool))] + [(t, g == t) for t in ["low", "mid", "high"]]
rows = [["N"] + [f"{m.sum():,}" for _, m in grp]]
for c, lab in cont:
    if c in ("LBXSATSI", "LBXSTR"): f = lambda s: f"{s.median():.0f} [{s.quantile(.25):.0f}–{s.quantile(.75):.0f}]"
    elif c in ("hba1c", "LBXSAL", "LBXSUA"): f = lambda s: f"{s.mean():.2f} ± {s.std():.2f}"
    else: f = lambda s: f"{s.mean():.1f} ± {s.std():.1f}"
    rows.append([lab] + [f(df.loc[m, c]) for _, m in grp])
for c, lab in [("female", "Female, %"), ("htn_told", "Told had hypertension, %"), ("ever_smoke", "Ever smoked ≥100 cigarettes, %")]:
    rows.append([lab] + [f"{100*df.loc[m, c].mean():.1f}" for _, m in grp])
rows.append(["CKD, n (%)"] + [f"{int(df.loc[m,'y'].sum()):,} ({100*df.loc[m,'y'].mean():.1f})" for _, m in grp])
rows.append(["CKD, survey-weighted %"] + [f"{100*np.average(df.loc[m,'y'], weights=df.loc[m,'w']):.1f}" for _, m in grp])
pd.DataFrame(rows, columns=["Characteristic", "Overall", "Low income", "Middle income", "High income"]).to_csv(RESULTS / "table1_cohort.csv", index=False)
json.dump(flow, open(RESULTS / "flow.json", "w"), indent=1, default=int)
print(json.dumps(flow, indent=1, default=int)); print(tab.to_string()); print(emp.round(1).T.to_string()); print(pd.DataFrame(mrows2).round(1).to_string())
