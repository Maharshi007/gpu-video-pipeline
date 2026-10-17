"""H.264 (x264) vs HEVC (x265) rate-distortion sweep with the FFmpeg CLI.

For each clip x codec x CRF: encode (timed), measure bitrate, then Y-PSNR and
SSIM against the reference. Also cross-checks FFmpeg's PSNR against our own
C++ implementation (yuv_psnr). Writes results/rd_results.csv and
results/psnr_crosscheck.csv.
"""
import argparse
import re
import subprocess
import time

import pandas as pd

from common import OUT, RESULTS, clips, exe, ff_tool

CODECS = ["libx264", "libx265"]
CRFS = [18, 23, 28, 33, 38]


def probe_frames_fps(clip):
    out = subprocess.run([ff_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate",
                          "-of", "csv=p=0", str(clip)], capture_output=True, text=True, check=True)
    rate, frames = out.stdout.strip().split(",")
    num, den = rate.split("/")
    return int(frames), float(num) / float(den)


def quality(dist, ref):
    """One FFmpeg pass computing both PSNR and SSIM (distorted first, reference second)."""
    graph = "[0:v]split[a][b];[1:v]split[c][d];[a][c]psnr;[b][d]ssim"
    r = subprocess.run([ff_tool("ffmpeg"), "-hide_banner", "-i", str(dist), "-i", str(ref),
                        "-lavfi", graph, "-f", "null", "-"], capture_output=True, text=True, check=True)
    psnr_y = float(re.search(r"PSNR y:([\d.]+)", r.stderr).group(1))
    ssim = float(re.search(r"SSIM .*All:([\d.]+)", r.stderr).group(1))
    return psnr_y, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium")
    # Both encoders default to psycho-visual tuning (psy-rd, AQ), which
    # deliberately LOWERS PSNR for better perceived quality. For a PSNR-based
    # comparison, tune both for PSNR so neither is penalised by the metric.
    ap.add_argument("--tune", default="psnr", help="x264/x265 tune ('none' to disable)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []

    for clip in clips():
        frames, fps = probe_frames_fps(clip)
        duration = frames / fps
        for codec in CODECS:
            for crf in CRFS:
                out = OUT / f"rd_{clip.stem}_{codec}_crf{crf}_{a.tune}.mp4"
                cmd = [ff_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(clip), "-c:v", codec, "-crf", str(crf), "-preset", a.preset,
                       "-pix_fmt", "yuv420p"]
                if a.tune != "none":
                    cmd += ["-tune", a.tune]
                if codec == "libx265":
                    cmd += ["-x265-params", "log-level=error"]
                t0 = time.perf_counter()
                subprocess.run(cmd + [str(out)], check=True)
                enc_s = time.perf_counter() - t0
                kbps = out.stat().st_size * 8 / duration / 1000
                psnr_y, ssim = quality(out, clip)
                rows.append(dict(clip=clip.stem, codec=codec, crf=crf, tune=a.tune, bitrate_kbps=round(kbps, 1),
                                 psnr_y_db=psnr_y, ssim=ssim, encode_s=round(enc_s, 3),
                                 encode_fps=round(frames / enc_s, 2)))
                print(rows[-1], flush=True)

        # Cross-check our C++ PSNR against FFmpeg on one encode per clip.
        chk = OUT / f"rd_{clip.stem}_libx264_crf28_{a.tune}.mp4"
        own = subprocess.run([exe("yuv_psnr"), str(chk), str(clip)],
                             capture_output=True, text=True, check=True).stdout
        own_psnr = float(re.search(r"psnr_y_avg_mse=([\d.]+)", own).group(1))
        ff_psnr = next(r["psnr_y_db"] for r in rows
                       if r["clip"] == clip.stem and r["codec"] == "libx264" and r["crf"] == 28)
        checks.append(dict(clip=clip.stem, file=chk.name, own_psnr_y_db=own_psnr,
                           ffmpeg_psnr_y_db=ff_psnr, diff_db=round(abs(own_psnr - ff_psnr), 4)))
        print(checks[-1], flush=True)

    RESULTS.mkdir(exist_ok=True)
    # Primary results use tune=psnr; other tunings are kept in a separate file.
    suffix = "" if a.tune == "psnr" else f"_tune-{a.tune}"
    pd.DataFrame(rows).to_csv(RESULTS / f"rd_results{suffix}.csv", index=False)
    pd.DataFrame(checks).to_csv(RESULTS / f"psnr_crosscheck{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
