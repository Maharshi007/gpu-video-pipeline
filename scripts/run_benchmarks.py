"""Run video_pipeline over every clip x filter x mode and collect CSVs.

Modes: CPU single-thread, CPU all cores (OpenMP), GPU pageable, GPU pinned.
Validation (GPU vs CPU every frame) is a SEPARATE pass: running the slow CPU
reference between GPU frames leaves the GPU idle, so it down-clocks and the
timings become noisy. Also runs kernel_bench (isolated, back-to-back kernels).
Writes results/benchmark_video.csv, validation.csv, stage_breakdown.csv,
kernel_bench.csv.
"""
import argparse
import subprocess

import pandas as pd

from common import OUT, RESULTS, clips, exe

FILTERS = ["none", "blur_naive", "blur_separable", "blur_tiled", "sobel"]
MODES = [
    ("cpu_1t", ["--mode", "cpu", "--cpu-threads", "1"]),
    ("cpu_all", ["--mode", "cpu", "--cpu-threads", "0"]),
    ("gpu", ["--mode", "gpu"]),
    ("gpu_pinned", ["--mode", "gpu", "--pinned"]),
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    print(lines[-1] if lines else "", flush=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"run failed: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--preset", default="veryfast")
    a = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = RESULTS / "benchmark_video.csv"
    val_csv = RESULTS / "validation.csv"
    kb_csv = RESULTS / "kernel_bench.csv"
    for p in (csv, val_csv, kb_csv):
        p.unlink(missing_ok=True)  # fresh results every run - never mix old numbers in

    for clip in clips():
        print(f"== kernel_bench {clip.stem}", flush=True)
        r = subprocess.run([exe("kernel_bench"), "--input", str(clip), "--csv", str(kb_csv)],
                           capture_output=True, text=True, check=True)
        print(r.stdout, flush=True)
        for f in FILTERS:
            for label, flags in MODES:
                out = OUT / f"{clip.stem}_{label}_{f}.mp4"
                print(f"== {clip.stem} {f} {label}", flush=True)
                run([exe("video_pipeline"), "--input", str(clip), "--output", str(out),
                     "--filter", f, "--frames", str(a.frames), "--warmup", str(a.warmup),
                     "--preset", a.preset, "--csv", str(csv), *flags])
            print(f"== {clip.stem} {f} validate", flush=True)
            run([exe("video_pipeline"), "--input", str(clip), "--output", "none", "--mode", "gpu",
                 "--filter", f, "--frames", str(a.frames), "--validate", "--cpu-threads", "0",
                 "--csv", str(val_csv)])

    df = pd.read_csv(csv)
    gpu = df[df["mode"].str.startswith("gpu")]
    gpu[["clip", "mode", "filter", "h2d_ms", "kernel_ms", "d2h_ms", "process_ms"]].to_csv(
        RESULTS / "stage_breakdown.csv", index=False)
    print(df[["clip", "mode", "filter", "processing_fps", "end_to_end_fps", "validation"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
