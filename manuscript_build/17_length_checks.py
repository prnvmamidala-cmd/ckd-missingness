#!/usr/bin/env python3
"""Length checks against JEI limits -> ../length_checks.json"""
import re, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE.parent; sys.path.insert(0, str(HERE)); import manuscript_text as T
strip = lambda t: re.sub(r"\[@[a-z0-9_,]+\]", "", re.sub(r"[*^_{}]", "", t))
wc = lambda items: sum(len(strip(x).split()) for x in items if isinstance(x, str))
pdf = next((OUT / "manuscript").glob("*.pdf"))
npages = lambda p: int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True).stdout).group(1))
pages = {}
for i in range(1, npages(pdf) + 1):
    t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"], capture_output=True, text=True).stdout
    for h in ["INTRODUCTION", "MATERIALS AND METHODS", "ACKNOWLEDGMENTS"]:
        if re.search(rf"^{h}$", t, re.M) and h not in pages: pages[h] = i
secs = {k: wc(getattr(T, k)) for k in ["INTRO", "RESULTS", "DISCUSSION", "METHODS"]}
D = {"Title": f"{len(T.TITLE)} characters (limit 110)", "Summary": f"{len(strip(T.SUMMARY).split())} words (limit < 250)",
     "Intro → Methods": f"pp. {pages['INTRODUCTION']}–{pages['ACKNOWLEDGMENTS']} of the PDF = {pages['ACKNOWLEDGMENTS'] - pages['INTRODUCTION'] + 1} pages "
                        f"(Methods ends on the page where Acknowledgments starts; limit 10); {sum(secs.values()):,} words "
                        f"(Intro {secs['INTRO']}, Results {secs['RESULTS']}, Discussion {secs['DISCUSSION']}, Methods {secs['METHODS']})",
     "Figures + tables": f"{len(T.FIGS)} figures + 3 tables = {len(T.FIGS) + 3} (limit 8)",
     "Manuscript PDF": f"{npages(pdf)} pages",
     "Required conclusion sentence in Summary and Discussion": str(all("In the scenarios examined, a substantial increase in the false-negative disparity was observed under the larger assumed missingness gradient, whereas no clear increase was detected under empirically calibrated NHANES patterns." in s for s in [T.SUMMARY, T.DISCUSSION[-1]]))}
json.dump(D, open(OUT / "length_checks.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(D, indent=1, ensure_ascii=False))
