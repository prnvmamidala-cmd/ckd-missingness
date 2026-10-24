#!/usr/bin/env python3
"""Download the public NHANES XPT files used in this study from CDC/NCHS (public domain).
2013-2018 (development) and 2021-2023 (temporal external validation). Files land in data/raw/."""
import time, urllib.request
from config import RAW_DIR, CYCLES, EXT_CYCLES
COMP = {"_H": ["DEMO", "BMX", "BPX", "GHB", "BPQ", "SMQ", "BIOPRO", "ALB_CR", "TCHOL"]}
COMP["_I"] = COMP["_J"] = COMP["_H"]
COMP["_L"] = ["DEMO", "BMX", "BPXO", "GHB", "BPQ", "SMQ", "BIOPRO", "ALB_CR", "TCHOL"]   # 2021-23 uses oscillometric BP
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for yr, suf in {**CYCLES, **EXT_CYCLES}.items():
        for comp in COMP[suf]:
            name = f"{comp}{suf}"; dest = RAW_DIR / f"{yr}_{name}.xpt"
            if dest.exists() and dest.stat().st_size > 1000: continue
            url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{yr}/DataFiles/{name}.xpt"
            for attempt in range(4):
                try:
                    data = urllib.request.urlopen(url, timeout=120).read()
                    assert data[:6] == b"HEADER", "not an XPT file"
                    dest.write_bytes(data); print("ok", dest.name, len(data)); break
                except Exception as e:
                    print("retry", name, e); time.sleep(3 * (attempt + 1))
            else:
                raise SystemExit(f"failed: {url}")
if __name__ == "__main__":
    main()
