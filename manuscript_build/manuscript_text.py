# -*- coding: utf-8 -*-
"""v4 manuscript text (JEI template order). Markup: *italic*, **bold**, ^{sup}, _{sub}. Citations: [@key1,key2] are numbered
in order of first use by the build script. Numbers are read directly from ../repo/results (v4 summaries for the main analyses,
v3 summaries for exploratory analyses carried over unchanged) so that text, tables, and figures cannot disagree."""
import json, math
from pathlib import Path
_R = Path(__file__).resolve().parent.parent / "repo" / "results"
_S = json.load(open(_R / "v4_summary.json")); _K = _S["keys"]; _C = _S["complete"]
for _k, _v in json.load(open(_R / "v4_summary_s0ce.json"))["keys"].items(): _K.setdefault(_k, _v)
_H = json.load(open(_R / "v4_summary_h2.json"))["keys"]
_V3 = json.load(open(_R / "v3_summary.json")); _K3 = _V3["keys"]
_E = json.load(open(_R / "v3_external.json")); _SEL = json.load(open(_R / "selection_model.json"))
_COV = json.load(open(_R / "v4_coverage_sim.json")); _DC = json.load(open(_R / "v3_dca.json"))
_SC = {"S0": "S7_cell_all15", "S1": "S1_cell", "S2": "S2_block", "S3": "S3_empirical", "S11": "S11_cell_unif_all15"}
def _f(x, d=1):
    s = f"{x:.{d}f}"
    if s == "-" + f"{0:.{d}f}": s = f"{0:.{d}f}"
    return s.replace("-", "−")
def V(sc, m, st="dFNR_low", th="sens80", ws="w", mod="xgb"):
    k = f"{_SC[sc]}|{mod}|{m}|{th}|{ws}"; return (_H if sc == "S11" else _K)[k][st]
def E(sc, m, st="dFNR_low", **kw): return _f(100 * V(sc, m, st, **kw)["est"])
def C(sc, m, st="dFNR_low", **kw):
    v = V(sc, m, st, **kw)
    if v.get("lo") is None or not math.isfinite(v["lo"]): return f"{_f(100 * v['est'])} pp (split range {_f(100 * v['split_min'])} to {_f(100 * v['split_max'])})"
    return f"{_f(100 * v['est'])} pp ({_f(100 * v['lo'])} to {_f(100 * v['hi'])})"
def CN(sc, m, st="dFNR_low", **kw):  # interval only
    v = V(sc, m, st, **kw); return f"{_f(100 * v['lo'])} to {_f(100 * v['hi'])}"
def B(sc, m, st="dFNR_low"):  # v3 pooled interval for comparison
    v = V(sc, m, st)["v3"]; return f"{_f(100 * v['lo'])} to {_f(100 * v['hi'])}"
def CF(st):
    v = _C["xgb|sens80|w"][st]; return f"{_f(100 * v['est'])}% ({_f(100 * v['lo'])} to {_f(100 * v['hi'])})"
def EX(k, st="dFNR_low"):
    v = _E["scenarios"][k][st]; return f"{_f(100 * v['est'])} pp ({_f(100 * v['lo'])} to {_f(100 * v['hi'])})"
_cv = _COV["coverage"]; _cw = _COV["mean_width_pp"]
COVTXT = dict(cons=f"{100 * _cv['cons']['dFNR_low']:.0f}", consg=f"{100 * _cv['cons']['gap']:.0f}", naive=f"{100 * _cv['naive']['dFNR_low']:.1f}",
              naiveg=f"{100 * _cv['naive']['gap']:.1f}", corr=f"{100 * _cv['corr']['dFNR_low']:.1f}", corrg=f"{100 * _cv['corr']['gap']:.1f}",
              wc=f"{_cw['cons']['dFNR_low']:.1f}", wn=f"{_cw['naive']['dFNR_low']:.1f}", sd=f"{_COV['diagnostics_pp']['empirical_sd_of_estimate'][0]:.1f}",
              bsd=f"{_COV['diagnostics_pp']['mean_sqrt2_boot_sd'][0]:.1f}", R=_COV["R"], truth=f"{_COV['truth_dFNR_low_pp']:.1f}")
NBOOT = _K["S2_block|xgb|mean|sens80|w"]["dFNR_low"]["n_boot"]

TITLE = "A simulation study of income-dependent missing data and kidney disease screening errors across income groups"
AUTHORS = "Pranav Mamidala^{1}, Samira Vedantam^{2}"
AFFIL = ["^{1} Monta Vista High School, Cupertino, California", "^{2} [MENTOR AFFILIATION]"]
STUDENTS = ["Pranav Mamidala, Grade 10"]
KEYWORDS = "Missing data, Simulation, Kidney, NHANES, Screening"
OVERVIEW = ("Lab and exam values may be missing more often for low-income patients. In a simulation using complete national survey "
            "records, we deleted values more often for low-income adults and found that common repairs made a kidney disease screening "
            "model miss more cases in that group when the assumed income difference in missing data was large.")

SUMMARY = (
"Screening algorithms must fill in missing lab and exam values before predicting, and records may be less complete for patients "
"with fewer financial resources. In this simulation study, we hypothesized that when values were deleted more often for low-income "
"adults, a chronic kidney disease (CKD) screening model repaired with chained-equation or nearest-neighbor imputation would miss more "
"low-income CKD cases than with complete data. Using 12,546 adults with complete records from the 2013–2018 National Health and "
"Nutrition Examination Survey (NHANES), we trained a model on 15 routine features and deleted test-set values for 30%, 10%, and 5% of "
f"low-, middle-, and high-income participants. In this preregistered design, the low-income false negative rate rose by {E('S0', 'det')} "
f"percentage points (95% CI, {CN('S0', 'det')}) with chained-equation and {E('S0', 'knn')} points ({CN('S0', 'knn')}) with "
f"nearest-neighbor imputation, versus about 1 point for high-income adults. In exploratory analyses, mean and median imputation and "
f"block deletion gave similar or larger increases, whereas stochastic iterative imputation with prediction averaging did not increase "
"missed cases but flagged more people without CKD. Missingness patterns observed in NHANES mainly reflect survey nonparticipation "
f"and nonmeasurement rather than clinical testing; with them, the low-minus-high difference changed by {E('S3', 'mean', 'gap')} points "
f"({CN('S3', 'mean', 'gap')}) for mean imputation. Weighting for complete-case selection did not materially change these results. In "
"the scenarios examined, a substantial increase in the false-negative disparity was observed under the larger assumed missingness "
"gradient, whereas no clear increase was detected under empirically calibrated NHANES patterns."
)

INTRO = [
"Chronic kidney disease (CKD) is a long-term loss of kidney function that affects about 14% of adults in the United States [@cdc]. "
"Because early CKD rarely causes symptoms, most adults with CKD do not know that they have it [@cdc,chu]. CKD is identified with two "
"laboratory measurements: the estimated glomerular filtration rate (eGFR), calculated from serum creatinine, and the urine "
"albumin-to-creatinine ratio (UACR) [@kdigo,inker]. Detecting CKD early matters, because blood pressure control and kidney-protective "
"medicines can slow its progression [@kdigo].",

"Lower socioeconomic status is linked to a higher risk of CKD [@nicholas], yet recommended kidney tests are often not ordered: in a "
"study of more than three million patients, only 35.1% of patients with diabetes and 4.1% of patients with hypertension had a UACR "
"test over two years [@shin]. Studies of electronic health record (EHR) data have reported that records can be less complete for "
"patients with fewer resources or less access to care [@gianf,getzen], that whether a value is recorded reflects how a patient "
"interacts with the healthcare system [@agniel], and that restricting analyses to patients with complete records selects a different, "
"often sicker, population [@weber,rusanov]. How large the income difference in missing data is in a given clinic is not known in "
"general, so in this study we treat it as an assumption to be varied rather than a measured fact.",

"Most machine learning (ML) models cannot accept a blank entry, so missing values are usually filled in, or imputed, before a "
"prediction is made [@wells,perez]. Single-value methods fill each blank with the training mean or median, or copy values from the "
"most similar records (k-nearest neighbors, KNN) [@troy]. Chained-equation (CE) imputation predicts each incomplete variable from the "
"others in turn; drawing several random imputations keeps some uncertainty about the missing value [@vanbuuren,rubin87]. Tree models "
"such as XGBoost can also route a missing value down a default branch without imputation [@xgb]. When missingness depends on income, "
"these choices might change how often a model misses disease in that group.",

"For a screening tool, the most harmful error is a false negative, a patient with disease who is not flagged. Comparing false negative "
"rates (FNRs) across groups is a standard fairness check called equal opportunity [@hardt,rajkomar,obermeyer]. Previous simulations "
"have shown that imputation choices can change the fairness of ML models [@caton,jeanselme] and that features under-reported for one "
"group can harm that group’s predictions [@akpinar], but these studies did not vary how much more often values were missing for "
"low-income patients in a disease screening task.",

"We used the National Health and Nutrition Examination Survey (NHANES), which measures exam and laboratory values on examined "
"participants [@nhanes], to run a simulation. Starting from complete records let us delete values on purpose and compare the same "
"people’s screening results before and after. Our study therefore asks what the consequences of an assumed income disparity in "
"missing data would be, not whether such a disparity exists in clinics. Our preregistered hypothesis was that when routine values were "
"deleted more often for low-income adults, repairing them with CE or KNN imputation would increase the FNR for low-income adults with "
"CKD compared with complete data. In the preregistered design, both repairs increased the low-income FNR under the large assumed "
"gradient; exploratory analyses showed when this did and did not happen.",
]

RESULTS = [
("sub", "Study cohort, complete-case selection, and observed missingness"),
"To compare screening results before and after deleting values, we needed participants whose true values were all known. Of 16,867 "
"interviewed, non-pregnant adults aged 20 years or older in NHANES 2013–2018, 12,546 had complete data for 15 routine features, "
"serum creatinine, UACR, and family income and formed the cohort (Figure 1). Their mean age was 49.6 ± 17.3 years, 51.0% were "
"female, and 2,142 (17.1%; survey-weighted 13.6%) met our CKD definition (Table 1). Low-income participants (poverty income ratio "
"[PIR] < 1.3; n = 3,869) had CKD more often than high-income participants (PIR ≥ 3.5; n = 3,848) (19.0% vs. 12.7%). Among the "
f"{_SEL['n_eligible_with_PIR']:,} eligible adults with a recorded income, the {_SEL['n_eligible_with_PIR'] - _SEL['n_included']:,} who were "
"excluded were more often low-income (27.3% vs. 20.9%, interview weights) and non-Hispanic Black (15.1% vs. 10.4%), and, among those "
"with both kidney tests, more often had CKD (23.2% vs. 13.5%). A model for inclusion separated included from "
f"excluded adults only modestly (area under the curve, {_SEL['selection_auc_weighted']:.2f}); after inverse-probability weighting, the "
"cohort matched the eligible population in income, race/ethnicity, and hypertension history within 0.5 percentage points (pp), and "
"its CKD prevalence (14.0%) was close to that of all eligible adults with kidney tests (14.1%).",

"In NHANES, every examined participant is offered the same measurements, so values are missing because a participant did not attend "
"the examination, declined or could not complete a component such as the blood draw, was ineligible for it, or because of operational "
"problems. This survey nonparticipation and nonmeasurement differs from clinical EHR missingness, where a test may never be ordered. "
"Using interview weights, at least one of the 13 deletable features was missing for 16.9% of low-income, 13.6% of "
"middle-income, and 11.9% of high-income adults; the exam block was incomplete for 12.5%, 10.0%, and 8.7%, and the laboratory block for "
"8.9%, 7.8%, and 7.3%. This observed gradient was much smaller than the 30%, 10%, and 5% gradient we had assumed.",

("sub", "Complete-data performance"),
"We repeated the analysis on 20 random splits of the 45 survey strata into training (mean n = 6,101), calibration (3,101), and "
"test (3,344) sets. On complete test data, XGBoost had a survey-weighted area under the receiver operating characteristic curve "
f"(AUROC) of {_V3['auroc']['complete|xgb']['all']:.3f} ({_V3['auroc']['complete|xgb']['low']:.3f} low-income, "
f"{_V3['auroc']['complete|xgb']['high']:.3f} high-income). At the threshold chosen on the calibration set for 80% sensitivity, XGBoost "
f"missed {CF('FNR_low')} of low-income and {CF('FNR_high')} of high-income CKD cases (Table 2). Here and below, 95% confidence "
"intervals (CIs) come from a hierarchical survey bootstrap that resamples whole primary sampling units (PSUs) and repeats the splits "
"and deletion masks inside each replicate (Methods).",

("sub", "Preregistered design (S0, confirmatory)"),
"In the preregistered design (S0), each test-set value of all 15 features was deleted with probability 30%, 10%, or 5% for low-, "
f"middle-, or high-income participants. CE imputation raised the low-income FNR by {C('S0', 'det')} and KNN imputation by "
f"{C('S0', 'knn')}, compared with {E('S0', 'det', 'dFNR_high')} and {E('S0', 'knn', 'dFNR_high')} pp for high-income participants "
f"(Table 2, Figure 2A). The low-minus-high gap changed by {C('S0', 'det', 'gap')} and {C('S0', 'knn', 'gap')}. Mean and median "
f"imputation, also prespecified, raised the low-income FNR by {C('S0', 'mean')} and {C('S0', 'median')}. As preregistered (H3), "
f"leaving values missing for XGBoost did not raise the FNR ({C('S0', 'native')}), but it raised the low-income false positive rate "
f"(FPR) by {E('S0', 'native', 'dFPR_low')} pp. In the preregistered control (H2), in which 15% of values were deleted for everyone "
f"(S11), the low-income FNR rose by {C('S11', 'det')} with CE and {C('S11', 'knn')} with KNN, and the low-minus-high gap was "
f"{C('S11', 'det', 'gap')} and {C('S11', 'knn', 'gap')}, so uniform deletion raised the FNR in every income group without a clear gap (Table 3).",

("sub", "Exploratory deletion scenarios"),
"The remaining scenarios were added after earlier results were seen and are exploratory. When age and sex were never deleted (S1), "
f"increases were smaller: {C('S1', 'mean')} for mean, {C('S1', 'det')} for CE, and {C('S1', 'knn')} for KNN imputation, "
"with the CE and KNN intervals including zero (Figure 2B). "
"When the lab, exam, and questionnaire blocks were each deleted as a unit at the same income-dependent rates (S2), mean, median, CE, "
f"and KNN imputation raised the low-income FNR by {C('S2', 'mean')}, {C('S2', 'median')}, {C('S2', 'det')}, and {C('S2', 'knn')}, "
f"versus {E('S2', 'knn', 'dFNR_high')} to {E('S2', 'mean', 'dFNR_high')} pp for high-income adults (Figure 2C). Stochastic iterative "
f"imputation with prediction averaging (SII) did not raise the low-income FNR in any scenario (S2: {C('S2', 'sii')}), but it raised the "
f"low-income FPR (S2: {C('S2', 'sii', 'dFPR_low')}); mean imputation instead lowered it ({C('S2', 'mean', 'dFPR_low')}).",

"With missingness patterns copied from NHANES for each income group (S3), deletion was much milder (6.4% of low-income and 5.2% of "
f"high-income values). Mean imputation raised the FNR by {C('S3', 'mean')} in low-income and {C('S3', 'mean', 'dFNR_high')} in "
f"high-income adults, so the gap changed by {C('S3', 'mean', 'gap')}; for the other repairs the gap estimates ranged from "
f"{min(E('S3', m, 'gap') for m in ['median', 'knn', 'det', 'sii', 'native'])} to {max(E('S3', m, 'gap') for m in ['median', 'knn', 'det', 'sii', 'native'])} "
"pp with intervals that included zero (Figure 2D). These intervals do not rule out gaps of about 2 to 3 pp, so S3 shows no clear "
"increase in the disparity rather than the absence of one.",

("sub", "Sensitivity analyses"),
f"Selection weighting changed the main estimates by at most {max(abs(100 * (V(sc, m, 'dFNR_low', ws='ipw')['est'] - V(sc, m)['est'])) for sc in ['S0', 'S1', 'S2', 'S3'] for m in ['mean', 'median', 'knn', 'sii', 'native']):.1f} pp "
f"(for example, S0 KNN, {C('S0', 'knn', ws='ipw')}; S2 mean, {C('S2', 'mean', ws='ipw')}) (Table 3, Figure 3). Handling binary "
f"features with logistic chained imputation (LCI) gave results close to SII (S0: {C('S0', 'lci')}; S2: {C('S2', 'lci')}), whereas "
f"keeping fractional imputed values moved SII toward zero (S0: {C('S0', 'sii_frac')}). Thresholds chosen by Youden’s index or for a 20% FPR gave larger "
f"increases for mean imputation in S2 ({C('S2', 'mean', th='youden')} and {C('S2', 'mean', th='fpr20')}). Random forest behaved like "
f"XGBoost (S2 mean, {C('S2', 'mean', mod='rf')}), but logistic regression was less sensitive ({C('S2', 'mean', mod='lr')}). Applied "
f"to the independent 2021–2023 cycle (AUROC {_E['auroc']['xgb']['all']:.3f}), the 2013–2018 model again showed increases under S2 "
f"for mean ({EX('S2_block|xgb|mean|sens80')}) and KNN ({EX('S2_block|xgb|knn|sens80')}) imputation, conditional on the fitted model.",

("sub", "Exploratory mechanism and net benefit"),
"For low-income CKD cases flagged with complete data but not after mean imputation in S1, deleted values had been replaced with "
"lower-risk values than the true ones, for example by 1.5 standard deviations for hypertension history and 1.2 for HbA1c, and these "
"cases lost predicted risk mainly through uric acid, HbA1c, systolic blood pressure, and hypertension history (Figure 4A). Deleting "
"the whole lab block for every low-income participant raised their FNR by 14.0 pp with mean imputation (Figure 4B). In decision "
"curve analysis, S2 mean imputation lowered low-income net benefit at a threshold probability of 0.10 from 0.085 to 0.073 (split "
"range of the difference, −0.017 to −0.009), whereas high-income net benefit changed by 0.001 or less (Figure 5). Subgroup results "
"by sex, age, and race/ethnicity were exploratory and based on few cases per cell, so they are not reported in detail.",
]

DISCUSSION = [
"In this simulation study, we asked what would happen to CKD screening errors if routine values were missing more often for "
"low-income adults. In the preregistered design, CE and KNN imputation raised the low-income FNR while high-income adults were "
"barely affected, so our confirmatory hypothesis was supported for that design. Exploratory analyses gave the same direction for mean "
"and median imputation and for block deletion, across thresholds, random forest, selection weighting, and a later survey cycle, but "
"smaller effects when age and sex were protected and for logistic regression. SII and native handling avoided extra missed cases "
"mainly by flagging more people without CKD.",

"The mechanism analysis suggests why single-value repairs caused harm. When a blank is filled with a typical value from a mostly healthy "
"training population, an abnormal blood pressure, HbA1c, or uric acid value is pulled toward normal, and cases near the threshold fall "
"below it. SII draws values with random variation and averages predictions over ten imputed datasets, which kept more patients above "
"the threshold, but at the cost of more false positives. This agrees with the view that missing data should be handled differently "
"for prediction than for description [@sperrin,groenwold].",

"The difference between assumed and empirical scenarios is central to interpretation. NHANES missingness comes from survey "
"nonparticipation and nonmeasurement, not from clinicians choosing which tests to order, and its income gradient was small. Under "
"those patterns, the estimated change in the low-minus-high gap was close to zero, but the intervals were wide enough to include "
"modest increases, so our results do not show that empirical missingness causes no disparity. Clinical missingness may follow a "
"larger or smaller gradient than either of our scenarios [@shin,agniel].",

"Our study had several limitations. First, the CKD label used a single eGFR and UACR measurement, whereas a clinical diagnosis "
"requires abnormalities lasting more than three months [@kdigo]. Second, we analyzed only adults with complete records; weighting "
"for measured characteristics did not change our results, but it cannot correct for unmeasured differences. Third, the assumed "
"scenarios did not let deletion depend on how sick a participant was. Fourth, each test set contained only about 199 low-income "
"and 129 high-income CKD cases, so intervals were wide. Our intervals cover PSU sampling and split and mask variability for this "
"pipeline; a coverage simulation suggested that they are conservative, but it held the model fixed, so we do not regard them as "
"validated. Finally, only the S0 low-income results for CE and KNN were confirmatory; everything else, including S1 and S2, was "
"added after earlier results were seen.",

"Our results do not show how any deployed tool performs. They suggest that developers could measure the income gradient of missing "
"data in their own records, run a stress test like ours with those rates, and report FNR and FPR separately by income group. Tools "
"could also prompt for the missing test instead of silently filling it with a typical value [@kdigo,shin]. In the "
"scenarios examined, a substantial increase in the false-negative disparity was observed under the larger assumed missingness "
"gradient, whereas no clear increase was detected under empirically calibrated NHANES patterns.",
]

METHODS = [
("sub", "Data source and study population"),
"Public data from NHANES 2013–2014, 2015–2016, and 2017–2018 (development) and August 2021–August 2023 (external validation) were "
"downloaded from the National Center for Health Statistics (NCHS) [@nhanes]. NHANES protocols were approved by the NCHS Ethics Review "
"Board, and participants provided informed consent [@erb]; we used only public, de-identified data. Adults aged 20 years or older who "
"completed the examination, were not pregnant, had serum creatinine and UACR, had all 15 features, and had a recorded PIR were "
"included. Examination weights were divided by three for the pooled six-year period [@guide].",

("sub", "CKD definition, predictors, and income groups"),
"The eGFR was calculated with the 2021 race-free CKD Epidemiology Collaboration creatinine equation [@inker], and CKD was defined as an "
"eGFR below 60 mL/min/1.73 m^{2} or a UACR of at least 30 mg/g from a single visit [@kdigo]. Creatinine, UACR, eGFR, and blood urea "
"nitrogen were not model inputs. The 15 features were age, sex, body mass index, waist circumference, systolic and diastolic blood "
"pressure, self-reported hypertension, ever smoking, HbA1c, glucose, uric acid, alanine aminotransferase, triglycerides, albumin, and "
"total cholesterol, forming an exam, a questionnaire, and a laboratory block. Income groups were low (PIR < 1.3), middle "
"(1.3 ≤ PIR < 3.5), and high (PIR ≥ 3.5); the preregistration’s question text mentioned PIR ≥ 4, but its design and all analyses used "
"these three groups. Income and race/ethnicity were not model inputs.",

("sub", "Splits, models, and thresholds"),
"For the point estimates, whole survey strata were randomly assigned 20 times to test (25%) and, from the rest, training and "
"calibration (2:1) sets (scikit-learn GroupShuffleSplit [@sklearn], seed 20261005 + 1,000 × split). XGBoost (300 trees, depth 3, "
"learning rate 0.05) [@xgb], standardized logistic regression, and random forest (300 trees, minimum leaf size 5) [@breiman] were fit "
"on the training set with class weighting. Thresholds were chosen on the calibration set for 80% weighted sensitivity (primary), "
"Youden’s index [@youden], or a 20% FPR.",

("sub", "Missingness scenarios"),
"Values were deleted only in the test set, ten masks per split, shared by all repairs. S0 (preregistered) deleted each of the 15 "
"features with probability 30%, 10%, or 5% by income group, and S11 (preregistered control) deleted 15% for everyone. S1 was S0 "
"with age and sex never deleted. S2 deleted the lab, exam, and questionnaire blocks as units at the same rates, with correlated "
"deletion across blocks (Gaussian copula, correlation 0.5). S3 sampled whole missingness patterns observed among NHANES adults of the "
"same income group with interview weights; these patterns reflect survey nonparticipation and nonmeasurement, not clinical test "
"ordering. Further exploratory scenarios (S4–S10) did not change the conclusions and are available with the code.",

("sub", "Repairs and handling of binary variables"),
"Imputers were fit on the training set. Mean and median imputation used the mode for the three binary features (sex, hypertension "
"history, ever smoking). KNN used five neighbors. CE, the preregistered “MICE” condition, was scikit-learn’s IterativeImputer with "
"Bayesian ridge regression and 10 iterations without posterior sampling, giving one deterministic imputation. SII ran the same "
"imputer ten times with posterior sampling and averaged the model’s predicted probabilities across the ten imputed datasets before "
"thresholding. SII is not conventional multiple imputation by chained equations, because no estimates are combined by Rubin’s rules "
"[@rubin87,vanbuuren]. For KNN, CE, and SII, imputed binary values were clipped to 0–1 and rounded, because the models were trained "
"only on 0/1 values and a fractional value would be sent down a tree branch by a split never meant for it. Two sensitivity analyses "
"tested this choice: keeping fractional values, and logistic chained imputation (LCI), which drew each binary feature from a "
"logistic regression on the other features and each continuous feature from a Bayesian ridge posterior predictive distribution "
"(10 iterations, ten imputations, predictions averaged). XGBoost also received unimputed data (native handling).",

("sub", "Estimand and confidence intervals"),
"The outcome was the change in survey-weighted FNR (ΔFNR) after deletion and repair compared with complete data for the same test "
"participants, by income group, and the low-minus-high difference (gap). The estimand is the ΔFNR of this whole pipeline (split, "
"training, threshold, deletion, repair), averaged over random stratum splits and masks, in the population represented by the "
"complete-case cohort. Point estimates average 20 splits × 10 masks. For 95% CIs, each of B = "
f"{NBOOT} outer bootstrap replicates drew two PSUs with replacement within each of the 45 strata of the full cohort. Inside each "
"replicate, the pipeline was rerun: strata were divided into four folds, each fold served once as the test set with models, imputers, "
"and thresholds refit on the other strata, and two new masks were drawn per fold; the replicate estimate averaged folds and masks. "
"The CI is the point estimate plus the 2.5th and 97.5th percentile deviations of replicates from their mean, multiplied by √2 "
"because with two PSUs per stratum the with-replacement bootstrap underestimates the variance by half [@raowu]. The interval therefore "
"covers PSU sampling variability plus remaining split and mask variability; it does not cover uncertainty from the choice of "
"scenario, CKD misclassification, or selection into the cohort. B was limited by computing time (about 100 s per replicate). In a "
f"coverage simulation with a known truth ({COVTXT['R']} samples from a pseudo-population of 450 PSUs, model held fixed), this interval "
f"covered the true low-income ΔFNR in {COVTXT['cons']}% of samples (mean width {COVTXT['wc']} pp), a plain percentile interval in "
f"{COVTXT['naive']}% (width {COVTXT['wn']} pp), and a noise-corrected interval in only {COVTXT['corr']}%. We "
"therefore describe the primary intervals as conservative. External-validation CIs (1,000 PSU bootstrap replicates of the 2021–2023 "
"test set) are conditional on the fitted model. Exploratory analyses carried over unchanged from our previous version are reported "
"as point estimates with split ranges. No p-values were used.",

("sub", "Selection weighting"),
"To assess whether complete-case selection affected the results, we fit an interview-weighted logistic model for inclusion among "
f"{_SEL['n_eligible_with_PIR']:,} eligible adults with a recorded PIR, using age, age squared, sex, PIR, income group, race/ethnicity, "
"survey cycle, hypertension history, and ever smoking (with missing-value indicators). Each included participant’s interview weight "
"was divided by the predicted inclusion probability and truncated at the 99th percentile; all thresholds and error rates were then "
"recomputed with these weights.",

("sub", "Decision timeline and analysis labels"),
"Each decision was dated and logged with the results already seen at that time. Before any model was fit (5 October 2026), we "
"preregistered the hypothesis (H1: low-income ΔFNR > 0 for CE and for KNN), the S0 design, XGBoost, the 80%-sensitivity threshold, "
"the uniform control (H2), and the native-handling comparison (H3). Only the S0 low-income ΔFNR for CE and KNN is confirmatory; H2 "
"and H3 are prespecified secondary comparisons. Deviations from the preregistration were the three-way split with a calibration-set "
"threshold, repeated splits and masks, and CIs in place of the preregistered bootstrap p-values. S1, S2, SII, and all other analyses "
"were added after earlier results were seen and are exploratory. Methods for exploratory mechanism (TreeSHAP [@lundberg]), decision "
"curve [@vickers], subgroup (sex, age, race/ethnicity), and external-validation (NHANES 2021–2023) analyses used the same "
"pipeline and thresholds as the main analysis.",

("sub", "Software"),
"Analyses used Python 3.13 with pandas 3.0 [@pandas], NumPy 2.5 [@numpy], SciPy 1.18, scikit-learn 1.9 [@sklearn], XGBoost 3.4 [@xgb], "
"and Matplotlib 3.11 [@mpl]. Code and derived results are available at [GITHUB REPOSITORY URL] [@repo].",
]

ACK = ("We thank our parents for their constant support and encouragement throughout this project. We also thank the National Center "
       "for Health Statistics and the NHANES participants for making these data publicly available. This research received no external funding.")


REFS = {
    'cdc': '“Chronic Kidney Disease in the United States.” *Centers for Disease Control and Prevention*. https://www.cdc.gov/kidney-disease/php/data-research/index.html. Accessed 17 Oct. 2026.',
    'chu': 'Chu, Chi D., et al. “CKD Awareness Among US Adults by Future Risk of Kidney Failure.” *American Journal of Kidney Diseases*, vol. 76, no. 2, Aug. 2020, pp. 174–183. https://doi.org/10.1053/j.ajkd.2020.01.007.',
    'kdigo': 'Stevens, Paul E., et al. “KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease.” *Kidney International*, vol. 105, no. 4S, Apr. 2024, pp. S117–S314. https://doi.org/10.1016/j.kint.2023.10.018.',
    'inker': 'Inker, Lesley A., et al. “New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race.” *New England Journal of Medicine*, vol. 385, no. 19, 4 Nov. 2021, pp. 1737–1749. https://doi.org/10.1056/NEJMoa2102953.',
    'nicholas': 'Nicholas, Susanne B., et al. “Socioeconomic Disparities in Chronic Kidney Disease.” *Advances in Chronic Kidney Disease*, vol. 22, no. 1, Jan. 2015, pp. 6–15. https://doi.org/10.1053/j.ackd.2014.07.002.',
    'shin': 'Shin, Jung-Im, et al. “Albuminuria Testing in Hypertension and Diabetes: An Individual-Participant Data Meta-Analysis in a Global Consortium.” *Hypertension*, vol. 78, no. 4, Oct. 2021, pp. 1042–1052. https://doi.org/10.1161/HYPERTENSIONAHA.121.17323.',
    'gianf': 'Gianfrancesco, Milena A., et al. “Potential Biases in Machine Learning Algorithms Using Electronic Health Record Data.” *JAMA Internal Medicine*, vol. 178, no. 11, Nov. 2018, pp. 1544–1547. https://doi.org/10.1001/jamainternmed.2018.3763.',
    'getzen': 'Getzen, Emily, et al. “Mining for Equitable Health: Assessing the Impact of Missing Data in Electronic Health Records.” *Journal of Biomedical Informatics*, vol. 139, Mar. 2023, p. 104269. https://doi.org/10.1016/j.jbi.2022.104269.',
    'agniel': 'Agniel, Denis, et al. “Biases in Electronic Health Record Data due to Processes within the Healthcare System: Retrospective Observational Study.” *BMJ*, vol. 361, 30 Apr. 2018, p. k1479. https://doi.org/10.1136/bmj.k1479.',
    'wells': 'Wells, Brian J., et al. “Strategies for Handling Missing Data in Electronic Health Record Derived Data.” *eGEMs*, vol. 1, no. 3, Dec. 2013, p. 7. https://doi.org/10.13063/2327-9214.1035.',
    'perez': 'Perez-Lebel, Alexandre, et al. “Benchmarking Missing-Values Approaches for Predictive Models on Health Databases.” *GigaScience*, vol. 11, Apr. 2022, p. giac013. https://doi.org/10.1093/gigascience/giac013.',
    'troy': 'Troyanskaya, Olga, et al. “Missing Value Estimation Methods for DNA Microarrays.” *Bioinformatics*, vol. 17, no. 6, June 2001, pp. 520–525. https://doi.org/10.1093/bioinformatics/17.6.520.',
    'vanbuuren': 'van Buuren, Stef and Karin Groothuis-Oudshoorn. “mice: Multivariate Imputation by Chained Equations in R.” *Journal of Statistical Software*, vol. 45, no. 3, 2011, pp. 1–67. https://doi.org/10.18637/jss.v045.i03.',
    'rubin76': 'Rubin, Donald B. “Inference and Missing Data.” *Biometrika*, vol. 63, no. 3, 1976, pp. 581–592. https://doi.org/10.1093/biomet/63.3.581.',
    'rubin87': 'Rubin, Donald B. *Multiple Imputation for Nonresponse in Surveys*. Wiley, 1987. https://doi.org/10.1002/9780470316696.',
    'xgb': 'Chen, Tianqi and Carlos Guestrin. “XGBoost: A Scalable Tree Boosting System.” *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, Aug. 2016, pp. 785–794. https://doi.org/10.1145/2939672.2939785.',
    'obermeyer': 'Obermeyer, Ziad, et al. “Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations.” *Science*, vol. 366, no. 6464, 25 Oct. 2019, pp. 447–453. https://doi.org/10.1126/science.aax2342.',
    'rajkomar': 'Rajkomar, Alvin, et al. “Ensuring Fairness in Machine Learning to Advance Health Equity.” *Annals of Internal Medicine*, vol. 169, no. 12, Dec. 2018, pp. 866–872. https://doi.org/10.7326/M18-1990.',
    'hardt': 'Hardt, Moritz, et al. “Equality of Opportunity in Supervised Learning.” *Advances in Neural Information Processing Systems*, vol. 29, 2016. https://arxiv.org/abs/1610.02413.',
    'caton': 'Caton, Simon, et al. “Impact of Imputation Strategies on Fairness in Machine Learning.” *Journal of Artificial Intelligence Research*, vol. 74, June 2022, pp. 1011–1035. https://doi.org/10.1613/jair.1.13197.',
    'jeanselme': 'Jeanselme, Vincent, et al. “Imputation Strategies Under Clinical Presence: Impact on Algorithmic Fairness.” *Proceedings of the 2nd Machine Learning for Health Symposium*, PMLR vol. 193, Nov. 2022, pp. 12–34. https://proceedings.mlr.press/v193/jeanselme22a.html.',
    'akpinar': 'Akpinar, Nil-Jana, et al. “The Impact of Differential Feature Under-Reporting on Algorithmic Fairness.” *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency*, June 2024, pp. 1355–1382. https://doi.org/10.1145/3630106.3658977.',
    'nhanes': '“National Health and Nutrition Examination Survey.” *Centers for Disease Control and Prevention, National Center for Health Statistics*. https://www.cdc.gov/nchs/nhanes/index.html. Accessed 17 Oct. 2026.',
    'groenwold': 'Groenwold, Rolf H. H. “Informative Missingness in Electronic Health Record Systems: The Curse of Knowing.” *Diagnostic and Prognostic Research*, vol. 4, no. 1, 2020, p. 8. https://doi.org/10.1186/s41512-020-00077-0.',
    'sperrin': 'Sperrin, Matthew, et al. “Missing Data Should Be Handled Differently for Prediction than for Description or Causal Explanation.” *Journal of Clinical Epidemiology*, vol. 125, Sept. 2020, pp. 183–187. https://doi.org/10.1016/j.jclinepi.2020.03.028.',
    'erb': '“Ethics Review Board Approval.” *Centers for Disease Control and Prevention, National Center for Health Statistics*. https://www.cdc.gov/nchs/nhanes/about/erb.html. Accessed 17 Oct. 2026.',
    'guide': '“NHANES Survey Methods and Analytic Guidelines.” *Centers for Disease Control and Prevention, National Center for Health Statistics*. https://wwwn.cdc.gov/nchs/nhanes/analyticguidelines.aspx. Accessed 17 Oct. 2026.',
    'sklearn': 'Pedregosa, Fabian, et al. “Scikit-learn: Machine Learning in Python.” *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html.',
    'field': 'Field, Christopher A. and Alan H. Welsh. “Bootstrapping Clustered Data.” *Journal of the Royal Statistical Society Series B: Statistical Methodology*, vol. 69, no. 3, June 2007, pp. 369–390. https://doi.org/10.1111/j.1467-9868.2007.00593.x.',
    'raowu': 'Rao, J. N. K. and C. F. J. Wu. “Resampling Inference with Complex Survey Data.” *Journal of the American Statistical Association*, vol. 83, no. 401, Mar. 1988, pp. 231–241. https://doi.org/10.1080/01621459.1988.10478591.',
    'wilson': 'Wilson, Edwin B. “Probable Inference, the Law of Succession, and Statistical Inference.” *Journal of the American Statistical Association*, vol. 22, no. 158, June 1927, pp. 209–212. https://doi.org/10.1080/01621459.1927.10502953.',
    'pandas': 'McKinney, Wes. “Data Structures for Statistical Computing in Python.” *Proceedings of the 9th Python in Science Conference*, 2010, pp. 56–61. https://doi.org/10.25080/Majora-92bf1922-00a.',
    'numpy': 'Harris, Charles R., et al. “Array Programming with NumPy.” *Nature*, vol. 585, no. 7825, 16 Sept. 2020, pp. 357–362. https://doi.org/10.1038/s41586-020-2649-2.',
    'mpl': 'Hunter, John D. “Matplotlib: A 2D Graphics Environment.” *Computing in Science & Engineering*, vol. 9, no. 3, 2007, pp. 90–95. https://doi.org/10.1109/MCSE.2007.55.',
    'repo': 'Mamidala, Pranav. “[REPOSITORY NAME].” *GitHub*. [GITHUB REPOSITORY URL]. Accessed [SUBMISSION DATE, e.g., 1 Nov. 2026].',
    'weber': 'Weber, Griffin M., et al. “Biases Introduced by Filtering Electronic Health Records for Patients with ‘Complete Data.’” *Journal of the American Medical Informatics Association*, vol. 24, no. 6, Nov. 2017, pp. 1134–1141. https://doi.org/10.1093/jamia/ocx071.',
    'rusanov': 'Rusanov, Alexander, et al. “Hidden in Plain Sight: Bias towards Sick Patients When Sampling Patients with Sufficient Electronic Health Record Data for Research.” *BMC Medical Informatics and Decision Making*, vol. 14, 2014, p. 51. https://doi.org/10.1186/1472-6947-14-51.',
    'vickers': 'Vickers, Andrew J. and Elena B. Elkin. “Decision Curve Analysis: A Novel Method for Evaluating Prediction Models.” *Medical Decision Making*, vol. 26, no. 6, Nov. 2006, pp. 565–574. https://doi.org/10.1177/0272989X06295361.',
    'lundberg': 'Lundberg, Scott M., et al. “From Local Explanations to Global Understanding with Explainable AI for Trees.” *Nature Machine Intelligence*, vol. 2, no. 1, Jan. 2020, pp. 56–67. https://doi.org/10.1038/s42256-019-0138-9.',
    'youden': 'Youden, W. J. “Index for Rating Diagnostic Tests.” *Cancer*, vol. 3, no. 1, 1950, pp. 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3.',
    'breiman': 'Breiman, Leo. “Random Forests.” *Machine Learning*, vol. 45, no. 1, Oct. 2001, pp. 5–32. https://doi.org/10.1023/A:1010933404324.',
}

ABBR = ("CKD, chronic kidney disease; FNR, false negative rate; FPR, false positive rate; PIR, poverty income ratio; KNN, k-nearest "
        "neighbors; CE, chained-equation imputation (preregistered, deterministic); SII, stochastic iterative imputation with prediction "
        "averaging (m = 10); PSU, primary sampling unit; CI, confidence interval; pp, percentage points.")
CIDEF = (f"95% CIs come from a hierarchical survey bootstrap (B = {NBOOT}): whole PSUs were resampled within strata, the splits "
         "(four-fold stratum cross-fitting with refitting) and masks were redone in each replicate, and the percentile interval was "
         "widened by √2 for two PSUs per stratum; a simulation suggested these intervals are conservative. ")
LABELS = ("Analysis labels: in S0, the low-income change for CE and KNN is confirmatory (H1); other S0 repairs are prespecified "
          "secondary conditions; S1, S2, S3, and SII are exploratory. ")
FIGS = [
("Figure1", "Figure 1. Design of the simulated missing-data study.",
 " Adults aged 20 years or older from NHANES 2013–2018 with complete data on 15 routine features, kidney tests, and family income were "
 "included. Whole survey strata were split 20 times into training, calibration, and test sets. Models and imputers were fit on the "
 "training set, thresholds were chosen on the calibration set, and values were deleted only in the test set. S0 and S11 are the "
 "preregistered design and control; S1–S3 are exploratory. The bottom box summarizes how CIs were computed. NHANES, National Health "
 "and Nutrition Examination Survey; eGFR, estimated glomerular filtration rate; UACR, urine albumin-to-creatinine ratio; LCI, logistic "
 "chained imputation; " + ABBR),
("Figure2", "Figure 2. Simulated income-dependent deletion increased low-income missed CKD cases for several repairs, with no clear gap under empirical NHANES patterns.",
 " Bar graphs of the change in FNR (after deletion and repair minus complete data) for low-, middle-, and high-income test participants "
 "with XGBoost at the 80%-sensitivity threshold. (A) Preregistered design: all 15 features deleted with probability 30%, 10%, and 5% by "
 "income group (S0). (B) As S0 but age and sex never deleted (S1, exploratory). (C) Correlated deletion of whole lab, exam, and "
 "questionnaire blocks at the same rates (S2, exploratory). (D) Missingness patterns observed in NHANES by income group, reflecting "
 "survey nonparticipation and nonmeasurement (S3, exploratory). Positive values mean more missed CKD cases. Bars show estimates over "
 "20 splits and 10 masks per split; error bars show " + CIDEF + LABELS + ABBR),
("Figure3", "Figure 3. Selection weighting and alternative thresholds did not change the direction of the low-income results.",
 " Forest plots of the low-income change in FNR (pp) for mean, KNN, CE, and SII repairs under (A) the preregistered design (S0) and "
 "(B) block deletion (S2, exploratory), for XGBoost at the 80%-sensitivity threshold with examination weights (main analysis) or "
 "inverse-probability-of-selection weights, and at the Youden and 20%-FPR thresholds. All rows except the S0 main-analysis CE and KNN "
 "points are exploratory. Points are estimates over 20 splits; bars show " + CIDEF + ABBR),
("Figure4", "Figure 4. Exploratory: single-value repairs lowered predicted risk mainly through blood pressure, hypertension history, HbA1c, and uric acid.",
 " (A) Mean change in TreeSHAP contribution (log-odds) of each feature for low-income CKD cases that were flagged with complete data but "
 "not after repair, under S1 (20 splits, first 10 masks). (B) Change in low-income FNR (pp) when one feature was deleted for every "
 "low-income test participant, averaged over 20 splits. Both analyses were added after earlier "
 "results were seen and come from our previous code version; no CIs are shown. BP, blood pressure; HbA1c, glycated hemoglobin; " + ABBR),
("Figure5", "Figure 5. Exploratory: block deletion lowered net benefit mainly for low-income adults.",
 " Decision curves for (A) low-income and (B) high-income test participants with complete data and after block deletion (S2) repaired "
 "by each method, using XGBoost probabilities recalibrated on the calibration set. Curves are means over 20 splits (10 masks per split); "
 "split ranges were narrow relative to differences between methods. “Flag everyone” and “flag no one” are reference strategies. " + ABBR),
]
TABLE1_TITLE = "Table 1. Characteristics of the study cohort by family income group."
TABLE1_CAP = (" Unweighted values for 12,546 NHANES 2013–2018 adults. Continuous variables are mean ± standard deviation, except ALT and "
 "triglycerides (median [interquartile range]). CKD was defined as eGFR < 60 mL/min/1.73 m^{2} or UACR ≥ 30 mg/g at one visit; "
 "survey-weighted prevalence uses examination weights. Excluded adults were more often low-income and had higher CKD prevalence (see Materials and Methods). "
 "NHANES, National Health and Nutrition Examination Survey; PIR, poverty income ratio; BMI, body mass index; BP, blood pressure; HbA1c, "
 "glycated hemoglobin; ALT, alanine aminotransferase; eGFR, estimated glomerular filtration rate; UACR, urine albumin-to-creatinine "
 "ratio; CKD, chronic kidney disease.")
TABLE2_TITLE = "Table 2. Change in missed CKD cases by income group in the preregistered design and three exploratory scenarios (simulation)."
TABLE2_CAP = (" First row: FNR (%) with complete data. Other rows: change in FNR (pp) after deletion and repair, XGBoost at the "
 "80%-sensitivity threshold, averaged over 20 splits and 10 masks per split. S0 deletes all 15 features with probability 30%, 10%, and "
 "5% for low-, middle-, and high-income participants; S1 is S0 with age and sex protected; S2 deletes whole blocks at the same rates; S3 "
 "uses missingness patterns observed in NHANES. *Confirmatory (H1, low-income column only). The last column is low-income minus "
 "high-income. Mean test set: n = 3,344; 199 low-, 250 middle-, and 129 high-income CKD cases. " + CIDEF + LABELS + ABBR)
TABLE3_TITLE = "Table 3. Prespecified control and exploratory sensitivity analyses."
TABLE3_CAP = (" Change in FNR (pp) for low- and high-income participants and their difference, XGBoost at the 80%-sensitivity threshold "
 "unless stated. The H2 control (S11) was prespecified; all other rows are exploratory. Selection weighting divides interview weights by "
 "the predicted probability of being a complete case. Values in parentheses are 95% CIs as in Table 2. Values in square brackets are "
 "point estimates with the range of the 20 split estimates (no bootstrap CI because these conditions were not rerun in the bootstrap). "
 "External validation used the 2013–2018 model on NHANES 2021–2023 (n = 4,381; 159 low- and 218 high-income CKD cases; 40 masks); its "
 "CIs (1,000 PSU bootstrap replicates) are conditional on the fitted model. Results for all scenarios, models, and thresholds are in the code repository. LCI, logistic chained "
 "imputation; other abbreviations as in Table 2.")

APPENDIX_JUSTIFICATION = (
"Statement of need for this appendix: Reviewers asked for a decision timeline, empirical missingness rates, comparisons of included, "
"excluded, and eligible participants with a selection-weighting analysis, a description and check of the confidence-interval method, "
"binary-variable sensitivity analyses, several model families and thresholds, missingness-aware training, a proxy ablation, decision "
"curves, external validation, a mechanism analysis, and subgroups. Reporting all of these in the main text would exceed JEI’s limit of "
"eight figures and tables combined and its ten-page limit for the Introduction through Materials and Methods. The main text reports "
"each analysis with its key numbers; this appendix gives the complete tables so that every number can be checked. Code is provided "
"in the linked GitHub repository.")
APPENDIX_INTRO = (
"All values are survey-weighted and come from the code in the repository (results/ and results/appendix/). ΔFNR and ΔFPR are in "
"percentage points (pp). Scenario labels: S0 preregistered cell deletion of all 15 features at 30/10/5% (code name S7_cell_all15); S1 "
"as S0 with age and sex protected; S2 block 30/10/5%; S3 empirical NHANES patterns by income (survey nonparticipation and "
"nonmeasurement); S4 empirical pooled; S5 cell uniform 15% (age/sex protected); S6 block uniform 15%; S8 lab only; S9 exam only; S10 "
"questionnaire only; S11 preregistered uniform 15% control (all 15 features). Repairs: CE, preregistered deterministic chained-equation "
"imputation; SII, stochastic iterative imputation with prediction averaging (called MICE in our earlier versions); LCI, logistic "
"chained imputation. Intervals: values in parentheses are conservative 95% CIs from the hierarchical PSU bootstrap (main text, "
"Methods) unless stated; values in square brackets are the point estimate with the range over 20 split estimates. Tables A10–A18 "
"come from our previous code version (same splits, 20 masks per split); their earlier pooled CIs double-counted the same participants "
"across splits and were withdrawn, so only point estimates and split ranges are shown. Analysis labels: only the S0 low-income ΔFNR "
"for CE and KNN (XGBoost, 80% sensitivity) is confirmatory; S11 (H2) and native handling (H3) are prespecified secondary comparisons; "
"everything else is exploratory. Methods for exploratory analyses: for the mechanism analysis, for low-income CKD cases flagged with "
"complete data, imputed values were compared with true values (in training standard deviations, signed so that negative means lower "
"risk) and TreeSHAP contributions between cases that lost and kept their flag; single-feature analyses deleted one feature or block "
"for every low-income participant. Net benefit used probabilities recalibrated on the calibration set by logistic recalibration. "
"Subgroups crossed income with sex, age group, and race/ethnicity. Missingness-aware training applied the same scenario to the "
"training and calibration sets and refit XGBoost with native handling or median imputation plus 13 missingness indicators. For "
"external validation, the model and imputers were fit on 2013–2018 data (75% of strata), thresholds on the remaining 25%, and all "
"were applied to complete 2021–2023 records with 40 masks per scenario.")
APPENDIX_TABLES = [
("A00_decision_timeline.csv", "Table A0. Timing of hypothesis and analysis decisions.", " Dates in 2026, Pacific Time. “Results seen” lists the analysis results available when each decision was made. Only decisions made before any results were seen are labeled confirmatory."),
("A1_empirical_missingness.csv", "Table A1. Observed NHANES 2013–2018 missingness by income group.", " Survey-weighted percentage (interview weights) of interviewed, non-pregnant adults aged 20 years or older with a recorded PIR who were missing each feature or block. Missingness here reflects survey nonparticipation and nonmeasurement (not attending the examination, declining or not completing a component, eligibility rules, operational reasons), not clinical test ordering. Used to build scenario S3."),
("A02b_selection_balance.csv", "Table A2. Eligible, included, and excluded adults, with selection weighting.", " Eligible adults: interviewed, non-pregnant, aged 20 years or older, with a recorded PIR. Included adults are shown with interview weights, examination (MEC) weights used in the main analysis, and inverse-probability-of-selection weights. CKD is among participants with both kidney tests."),
("A02c_selection_model.csv", "Table A2b. Selection (complete-case) model.", " Interview-weighted logistic regression for inclusion; age centered at 50 years and divided by 17, PIR divided by 5. Weights are interview weight / 3 divided by the predicted probability, truncated at the 99th percentile."),
("A2_included_vs_excluded.csv", "Table A3. Included versus all excluded adults, including those without a recorded income.", " Survey-weighted (interview weights) characteristics. *Among participants with the value measured. Exclusion reasons (first failing step): missing PIR 1,329; no creatinine or UACR 1,157; missing at least one feature 1,105; not examined 730."),
("realized_rates.csv", "Table A4. Realized deletion rates of each scenario.", " Percentage of the deletable cells deleted, percentage of participants with at least one deleted value, and percentage of age/sex cells deleted, by income group (one mask on the full cohort). S7_cell_all15 is S0. S11 deletes 15% of all cells by design."),
("A05_ci_comparison.csv", "Table A5. Confidence intervals in this version versus our previous version.", " The previous pooled CI combined bootstrap replicates from 20 splits of the same participants and has been withdrawn. Primary: conservative √2-rescaled percentile interval from the hierarchical bootstrap. Naive: unscaled percentile interval. Noise-corrected: Wald interval that subtracts the within-replicate split and mask variance (estimated from 12 repeated cross-fits of the full sample), shown for comparison only because it undercovered in Table A6."),
("A06_coverage_sim.csv", "Table A6. Coverage simulation for the interval methods.", " Pseudo-population: each of the 45 strata split into 10 pseudo-PSUs; each simulated sample drew two pseudo-PSUs per stratum. A fixed pre-trained XGBoost model and threshold, mean imputation, and S2 block deletion were used, so the simulation checks the design-based part of the interval but not model refitting. Point estimates and bootstrap replicates mimicked the main analysis (repeated stratum subsets and masks; cross-fitted replicates)."),
("A07_all_v4_results.csv", "Table A7. All results of this version.", " Every scenario, model, repair, threshold, and weighting. LR and RF rows and SII with fractional binaries have point estimates with split ranges only."),
("A07b_complete_baselines.csv", "Table A7b. Complete-data FNR by model, threshold, and weights.", " FNR in %."),
("A10_other_scenarios_v3.csv", "Table A10. Additional exploratory scenarios.", " XGBoost, 80% sensitivity; point estimate [range over 20 splits]."),
("A11_aware_training_v3.csv", "Table A11. Missingness-aware training (exploratory).", " Point estimate [range over 20 splits]; AUROC in the low-income group."),
("A12_proxy_ablation_v3.csv", "Table A12. Proxy ablation: uric acid and albumin removed (exploratory).", " 13-feature model (10 masks per split) versus the 15-feature model; point estimate [range over 20 splits]."),
("A13_external_2021_2023.csv", "Table A13. External validation in NHANES August 2021–August 2023 (exploratory).", " 2013–2018 model, imputers, and thresholds applied to 4,381 complete-case adults; 40 masks per scenario; 1,000 PSU bootstrap replicates within 15 strata. These CIs are conditional on the fitted model. Blood pressure was measured with an oscillometric device in this cycle."),
("A14_intersectional_v3.csv", "Table A14. Exploratory intersectional analysis.", " Change in FNR (pp), point estimate [range over 20 splits]. † fewer than 30 CKD cases per test set on average. NHWhite, non-Hispanic White; NHBlack, non-Hispanic Black; MexAm, Mexican American; OtherHisp, other Hispanic."),
("A15_mechanism_v3.csv", "Table A15. Mechanism: SHAP changes and imputed-minus-true values (exploratory).", " For low-income CKD cases flagged with complete data, mean change in TreeSHAP contribution (log-odds) for cases that lost their flag (flipped) and that kept it (stayed), and mean imputed-minus-true value of deleted cells in training standard deviations, signed so that negative means lower risk. 20 splits, first 10 masks."),
("A16_single_feature_v3.csv", "Table A16. Single-feature and single-block deletion for all low-income participants (exploratory).", " Change in low-income FNR (pp); mean over 20 splits with the 2.5th–97.5th percentile of split estimates."),
("A17_decision_curve_v3.csv", "Table A17. Net benefit at selected threshold probabilities (exploratory).", " XGBoost probabilities recalibrated on the calibration set; means over 20 splits (10 masks per split). The last column gives the change versus complete data at 0.10 with the 2.5th–97.5th percentile across splits."),
("A18_auroc_v3.csv", "Table A18. Survey-weighted AUROC by condition and income group.", " Means over 20 splits."),
]
