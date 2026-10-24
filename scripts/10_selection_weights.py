#!/usr/bin/env python3
"""v4 selection-weighting sensitivity (review point B). Among eligible interviewed adults (>= 20 y, not pregnant,
NHANES 2013-2018) WITH a recorded PIR, model P(included as a complete case | age, age^2, sex, race/ethnicity, PIR, PIR tier,
cycle, self-reported hypertension, ever smoking [with missing indicators]) by interview-weighted logistic regression.
Inverse-probability-of-complete-case weight for included participants: w_ipw = (WTINT2YR / 3) / p_hat, truncated at the
99th percentile. Target population: eligible interviewed adults with a recorded income (income groups need PIR).
Outputs: data/ipw_weights.csv (SEQN, w_ipw; not shipped), results/selection_model.json, results/table_selection_balance.csv."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from config import RESULTS, ROOT
from common import load_all, load_cohort
df = load_cohort(); allp = load_all()
el = allp[(allp.age >= 20) & (allp.pregnant != 1) & allp.pir_tier.notna()].copy()
el["inc"] = el.SEQN.astype(int).isin(set(df.SEQN.astype(int))).astype(int)
def design(d):
    X = pd.DataFrame({"age": (d.age - 50) / 17, "age2": ((d.age - 50) / 17) ** 2, "female": d.female, "pir": d.pir / 5})
    for r in ["NHBlack", "MexAm", "OtherHisp", "Other"]: X[f"race_{r}"] = (d.race5 == r).astype(float)
    for t in ["low", "mid"]: X[f"tier_{t}"] = (d.pir_tier.astype(str) == t).astype(float)
    for c in [2015, 2017]: X[f"cyc_{c}"] = (d.cycle == c).astype(float)
    for v in ["htn_told", "ever_smoke"]:
        X[f"{v}_miss"] = d[v].isna().astype(float); X[v] = d[v].fillna(0)
    return X
X = design(el); w = el.WTINT2YR.to_numpy(float)
lr = LogisticRegression(C=1e4, max_iter=5000).fit(X, el.inc, sample_weight=w / w.mean())
p = lr.predict_proba(X)[:, 1]; el["p"] = p
inc = el[el.inc == 1].copy(); raw = (inc.WTINT2YR / 3) / inc.p
cap = np.quantile(raw, 0.99); inc["w_ipw"] = np.minimum(raw, cap)
inc[["SEQN", "w_ipw", "p"]].to_csv(ROOT / "data" / "ipw_weights.csv", index=False)
ess = lambda ww: ww.sum() ** 2 / (ww ** 2).sum()
info = {"n_eligible_with_PIR": int(len(el)), "n_included": int(el.inc.sum()), "selection_auc_weighted": float(roc_auc_score(el.inc, p, sample_weight=w)),
        "p_hat_range_included": [float(inc.p.min()), float(inc.p.max())], "truncation_cap_99pct": float(cap), "n_truncated": int((raw > cap).sum()),
        "ess_mec_weights": float(ess(inc.WTMEC2YR)), "ess_ipw_weights": float(ess(inc.w_ipw)),
        "coefficients": dict(zip(X.columns, map(float, lr.coef_[0])))}
json.dump(info, open(RESULTS / "selection_model.json", "w"), indent=1); print(json.dumps(info, indent=1))
wv = lambda s, ww: np.average(s[s.notna()], weights=ww[s.notna()]) if s.notna().any() else np.nan
def col(d, ww):
    c = {"N": f"{len(d):,}", "Age, years (mean)": f"{wv(d.age, ww):.1f}", "Female, %": f"{100 * wv(d.female, ww):.1f}"}
    for lab, rc in [("Non-Hispanic White", "NHWhite"), ("Non-Hispanic Black", "NHBlack"), ("Mexican American", "MexAm"), ("Other Hispanic", "OtherHisp"), ("Other/multiracial", "Other")]:
        c[f"  {lab}, %"] = f"{100 * wv((d.race5 == rc).astype(float), ww):.1f}"
    for lab, t in [("Low income, %", "low"), ("Middle income, %", "mid"), ("High income, %", "high")]: c[lab] = f"{100 * wv((d.pir_tier.astype(str) == t).astype(float), ww):.1f}"
    c["Told had hypertension, %"] = f"{100 * wv(d.htn_told, ww):.1f}"; c["Ever smoked, %"] = f"{100 * wv(d.ever_smoke, ww):.1f}"
    c["CKD, % (where both kidney labs measured)"] = f"{100 * wv(d.ckd, ww):.1f}"
    return c
m = el.merge(inc[["SEQN", "w_ipw"]], on="SEQN", how="left")
I = m[m.inc == 1]
tab = pd.DataFrame({"Eligible adults with PIR (interview weights)": col(m, m.WTINT2YR),
                    "Included (interview weights)": col(I, I.WTINT2YR), "Included (MEC weights, main analysis)": col(I, I.WTMEC2YR),
                    "Included (IPW weights)": col(I, I.w_ipw), "Excluded (interview weights)": col(m[m.inc == 0], m[m.inc == 0].WTINT2YR)})
tab.index.name = "Characteristic"; tab.to_csv(RESULTS / "table_selection_balance.csv"); print(tab.to_string())
