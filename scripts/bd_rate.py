"""Bjontegaard delta-rate (BD-rate) of HEVC relative to H.264.

Classic method (VCEG-M33): fit log10(bitrate) as a cubic polynomial of PSNR
for each codec, integrate both fits over the overlapping PSNR range, and
convert the average log-rate difference to a percentage.
Negative BD-rate = HEVC needs that much LESS bitrate for the same PSNR.
"""
import numpy as np
import pandas as pd

from common import RESULTS


def bd_rate(rate_a, psnr_a, rate_b, psnr_b):
    la, lb = np.log10(rate_a), np.log10(rate_b)
    pa = np.polyfit(psnr_a, la, 3)
    pb = np.polyfit(psnr_b, lb, 3)
    lo = max(min(psnr_a), min(psnr_b))
    hi = min(max(psnr_a), max(psnr_b))
    ia, ib = np.polyint(pa), np.polyint(pb)
    avg_a = (np.polyval(ia, hi) - np.polyval(ia, lo)) / (hi - lo)
    avg_b = (np.polyval(ib, hi) - np.polyval(ib, lo)) / (hi - lo)
    return (10 ** (avg_b - avg_a) - 1) * 100


def main():
    rows = []
    for path in sorted(RESULTS.glob("rd_results*.csv")):
        tune = path.stem.replace("rd_results", "").lstrip("_").replace("tune-", "") or "psnr"
        rows += per_clip(pd.read_csv(path), tune)
    pd.DataFrame(rows).to_csv(RESULTS / "bd_rate.csv", index=False)


def per_clip(df, tune):
    rows = []
    for clip, g in df.groupby("clip"):
        a = g[g.codec == "libx264"].sort_values("psnr_y_db")
        b = g[g.codec == "libx265"].sort_values("psnr_y_db")
        bd = bd_rate(a.bitrate_kbps.values, a.psnr_y_db.values, b.bitrate_kbps.values, b.psnr_y_db.values)
        rows.append(dict(clip=clip, tune=tune, anchor="libx264", test="libx265",
                         bd_rate_pct=round(bd, 2)))
        print(f"{clip} (tune={tune}): HEVC BD-rate vs H.264 = {bd:+.2f}%")
    return rows


if __name__ == "__main__":
    main()
