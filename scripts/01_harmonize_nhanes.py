#!/usr/bin/env python3
"""Merge NHANES components into one table, one row per participant (interviewed AND examined; RIDSTATR kept).
Decisions: SBP/DBP = mean of available readings (DBP==0 -> missing; 2021-23 oscillometric readings 1-3); PIR = INDFMPIR
(top-coded at 5), tiers low <1.3, middle 1.3-<3.5, high >=3.5; race = RIDRETH1; yes/no items 1/0 (refused/don't know ->
missing). Lab weight: WTMEC2YR (2013-18); 2021-23 labs use the phlebotomy weight WTPH2YR when present."""
import numpy as np, pandas as pd
from config import RAW_DIR, HARMONIZED_CSV, CYCLES, EXT_CYCLES
def rd(yr, suf, comp, cols):
    d = pd.read_sas(RAW_DIR / f"{yr}_{comp}{suf}.xpt", format="xport")
    return d[[c for c in cols if c in d.columns]]
def main():
    parts = []
    for yr, suf in {**CYCLES, **EXT_CYCLES}.items():
        x = rd(yr, suf, "DEMO", ["SEQN", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDFMPIR", "RIDEXPRG",
                                 "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]).copy()
        x["cycle"] = yr
        if suf == "_L":
            bp = pd.read_sas(RAW_DIR / f"{yr}_BPXO{suf}.xpt", format="xport")
            sy = bp[["BPXOSY1", "BPXOSY2", "BPXOSY3"]]; di = bp[["BPXODI1", "BPXODI2", "BPXODI3"]].where(lambda v: v > 0)
        else:
            bp = pd.read_sas(RAW_DIR / f"{yr}_BPX{suf}.xpt", format="xport")
            sy = bp[[c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in bp]]
            di = bp[[c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in bp]].where(lambda v: v > 0)
        bp = pd.DataFrame({"SEQN": bp.SEQN, "sbp": sy.mean(axis=1), "dbp": di.mean(axis=1)})
        for t in [rd(yr, suf, "BMX", ["SEQN", "BMXBMI", "BMXWAIST"]), bp, rd(yr, suf, "GHB", ["SEQN", "LBXGH", "WTPH2YR"]),
                  rd(yr, suf, "BPQ", ["SEQN", "BPQ020"]), rd(yr, suf, "SMQ", ["SEQN", "SMQ020"]),
                  rd(yr, suf, "BIOPRO", ["SEQN", "LBXSCR", "LBXSGL", "LBXSUA", "LBXSATSI", "LBXSTR", "LBXSBU", "LBXSAL"]),
                  rd(yr, suf, "ALB_CR", ["SEQN", "URDACT"]), rd(yr, suf, "TCHOL", ["SEQN", "LBXTC"])]:
            x = x.merge(t, on="SEQN", how="left")
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    yn = lambda s: s.map({1: 1.0, 2: 0.0})
    df["examined"] = (df.RIDSTATR == 2).astype(int)
    df["female"] = (df.RIAGENDR == 2).astype(float); df["age"] = df.RIDAGEYR
    df["race5"] = df.RIDRETH1.map({1: "MexAm", 2: "OtherHisp", 3: "NHWhite", 4: "NHBlack", 5: "Other"})
    df["pir"] = df.INDFMPIR
    df["pir_tier"] = pd.cut(df.pir, [-0.01, 1.3, 3.5, 10], right=False, labels=["low", "mid", "high"])
    df["pregnant"] = (df.RIDEXPRG == 1).astype(float)
    df["hba1c"] = df.LBXGH; df["htn_told"] = yn(df.BPQ020); df["ever_smoke"] = yn(df.SMQ020)
    df["bmi"] = df.BMXBMI; df["waist"] = df.BMXWAIST
    HARMONIZED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HARMONIZED_CSV, index=False); print(df.shape, "->", HARMONIZED_CSV)
    print(df.groupby("cycle").agg(n=("SEQN", "size"), examined=("examined", "sum")))
if __name__ == "__main__":
    main()
