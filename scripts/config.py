"""Shared paths and constants (v4). Paths are relative to the repository root unless overridden by env vars."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path(os.environ.get("NHANES_RAW_DIR", ROOT / "data" / "raw"))
HARMONIZED_CSV = Path(os.environ.get("NHANES_CSV", ROOT / "data" / "nhanes_harmonized_all.csv.gz"))
RESULTS = ROOT / "results"; FIGURES = ROOT / "figures"
CYCLES = {2013: "_H", 2015: "_I", 2017: "_J"}          # development data
EXT_CYCLES = {2021: "_L"}                              # temporal external validation (Aug 2021-Aug 2023), no overlap
SEED = 20261005
FEATS = ["age", "female", "bmi", "waist", "sbp", "dbp", "htn_told", "ever_smoke",
         "hba1c", "LBXSGL", "LBXSUA", "LBXSATSI", "LBXSTR", "LBXSAL", "LBXTC"]
BINARY = ["female", "htn_told", "ever_smoke"]
PROTECTED = ["age", "female"]                          # never deleted in the v3 primary analyses
BLOCKS = {"exam": ["bmi", "waist", "sbp", "dbp"],
          "questionnaire": ["htn_told", "ever_smoke"],
          "lab": ["hba1c", "LBXSGL", "LBXSUA", "LBXSATSI", "LBXSTR", "LBXSAL", "LBXTC"]}
PROXIES = ["LBXSUA", "LBXSAL"]                         # partly kidney-related; ablation (BUN is never an input)
# Excluded from model inputs (label leakage): LBXSCR (creatinine), URDACT (UACR), eGFR, LBXSBU (BUN)
