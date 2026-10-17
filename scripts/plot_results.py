"""Generate every figure in results/figures/ from the measured CSVs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS

# Validated categorical palette (fixed order, never cycled) + neutral inks.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})

FILTERS = ["none", "blur_naive", "blur_separable", "blur_tiled", "sobel"]
MODES = [("cpu_1t", "CPU 1 thread", S1), ("cpu_all", "CPU all cores (OpenMP)", S2),
         ("gpu", "GPU (OpenCL)", S3), ("gpu_pinned", "GPU + pinned memory", S4)]


def cpu_all_label(df):
    """CSV stores the actual thread count, e.g. cpu_12t; map it back to cpu_all."""
    df = df.copy()
    df["m"] = df["mode"]
    df.loc[df["mode"].str.match(r"cpu_\d+t") & (df["cpu_threads"] > 1), "m"] = "cpu_all"
    df.loc[df["mode"] == "cpu_1t", "m"] = "cpu_1t"
    return df


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def fps_cpu_vs_gpu(df):
    clips = sorted(df["clip"].unique())
    fig, axes = plt.subplots(1, len(clips), figsize=(6.2 * len(clips), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(FILTERS))
    bw = 0.2
    for ax, clip in zip(axes, clips):
        d = df[df["clip"] == clip]
        for i, (m, label, color) in enumerate(MODES):
            vals = [d[(d.m == m) & (d["filter"] == f)].processing_fps.mean() for f in FILTERS]
            ax.bar(x + (i - 1.5) * bw, vals, bw, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1.5)
        # Selective direct label: GPU speed-up over the multi-core CPU baseline.
        for j, f in enumerate(FILTERS):
            g = d[(d.m == "gpu") & (d["filter"] == f)].processing_fps.mean()
            c = d[(d.m == "cpu_all") & (d["filter"] == f)].processing_fps.mean()
            ax.annotate(f"{g / c:.0f}x", (x[j] + 0.5 * bw, g), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_yscale("log")
        ax.set_xticks(x, [f.replace("_", "\n") for f in FILTERS])
        ax.set_title(clip)
    axes[0].set_ylabel("processing fps (log scale, higher is better)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(len(clips) / 2 + 0.05, 1.22), ncol=4)
    fig.suptitle("Frame processing throughput at 1080p - labels: GPU speed-up vs CPU all cores",
                 y=1.08, fontsize=11, color=INK2)
    save(fig, "fps_cpu_vs_gpu.png")


def stage_breakdown(kb):
    """From kernel_bench.csv (isolated, back-to-back runs -> steady GPU clocks)."""
    clip = "crowd_run_120f" if "crowd_run_120f" in kb["clip"].values else sorted(kb["clip"].unique())[0]
    d = kb[kb["clip"] == clip]
    rows = [(f, p) for f in FILTERS for p in (0, 1)]
    labels = [f"{f}  ({'pinned' if p else 'pageable'})" for f, p in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(rows))
    for col, label, color in [("h2d_ms", "Host -> device", S1), ("kernel_ms", "Kernels", S2),
                              ("d2h_ms", "Device -> host", S3)]:
        vals = np.array([d[(d["filter"] == f) & (d.pinned == p)][col].mean() for f, p in rows])
        ax.barh(labels, vals, left=left, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        left += vals
    for i, total in enumerate(left):
        ax.annotate(f"{total:.2f} ms", (total, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("median GPU time per 1080p frame (ms, OpenCL events, 100 runs)")
    ax.set_title(f"GPU per-stage breakdown - {clip}")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "stage_breakdown_gpu.png")


def blur_variants(kb):
    """Blur-only kernel time = total kernel time minus the colour-conversion
    kernels (measured by the 'none' filter), from kernel_bench.csv."""
    d = kb[kb.pinned == 0]
    clips = sorted(d["clip"].unique())
    variants = ["blur_naive", "blur_separable", "blur_tiled"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(variants))
    bw = 0.35
    for i, (clip, color) in enumerate(zip(clips, [S1, S2])):
        base = d[(d["clip"] == clip) & (d["filter"] == "none")].kernel_ms.mean()
        vals = [d[(d["clip"] == clip) & (d["filter"] == v)].kernel_ms.mean() - base for v in variants]
        bars = ax.bar(x + (i - 0.5) * bw, vals, bw, color=color, label=clip, edgecolor=SURFACE,
                      linewidth=2)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["naive 5x5\n(25 taps)", "separable\n(5 + 5 taps)", "tiled\n(__local memory)"])
    ax.set_ylabel("blur kernel time per frame (ms)")
    ax.set_title("Gaussian blur variants on GTX 1650 (1080p RGB)")
    ax.legend()
    save(fig, "blur_variants.png")


def end_to_end(df):
    clip = "crowd_run_120f" if "crowd_run_120f" in df["clip"].values else sorted(df["clip"].unique())[0]
    d = df[(df["clip"] == clip) & (df.m == "gpu")]
    x = np.arange(len(FILTERS))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (col, label, color) in enumerate([("processing_fps", "GPU processing only", S1),
                                             ("end_to_end_fps", "End-to-end (decode + GPU + x264 encode)", S2)]):
        vals = [d[d["filter"] == f][col].mean() for f in FILTERS]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, color=color, label=label, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x, [f.replace("_", "\n") for f in FILTERS])
    ax.set_ylabel("fps")
    ax.set_title(f"Why end-to-end fps is encoder-bound - {clip}")
    ax.legend()
    save(fig, "end_to_end_vs_processing.png")


def rd_plots(rd):
    clips = sorted(rd["clip"].unique())
    for metric, ylabel, name in [("psnr_y_db", "Y-PSNR (dB)", "rd_psnr.png"),
                                 ("ssim", "SSIM (All)", "rd_ssim.png")]:
        fig, axes = plt.subplots(1, len(clips), figsize=(5.5 * len(clips), 4))
        for ax, clip in zip(np.atleast_1d(axes), clips):
            for codec, label, color in [("libx264", "H.264 (x264)", S1), ("libx265", "HEVC (x265)", S2)]:
                g = rd[(rd["clip"] == clip) & (rd.codec == codec)].sort_values("bitrate_kbps")
                ax.plot(g.bitrate_kbps / 1000, g[metric], marker="o", markersize=6, color=color,
                        label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_xscale("log")
            # Plain-number ticks; matplotlib's default minor log labels overlap.
            lo, hi = g.bitrate_kbps.min() / 1000, rd[rd["clip"] == clip].bitrate_kbps.max() / 1000
            ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if lo * 0.7 <= t <= hi * 1.3]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel("bitrate (Mbps, log scale)")
            ax.set_ylabel(ylabel)
            ax.set_title(clip)
            ax.legend()
        tune = rd["tune"].iloc[0] if "tune" in rd else "psnr"
        fig.suptitle(f"Rate-distortion: CRF 18-38, preset medium, -tune {tune}", y=1.02,
                     fontsize=11, color=INK2)
        save(fig, name)

    fig, axes = plt.subplots(1, len(clips), figsize=(5.5 * len(clips), 4), sharey=True)
    for ax, clip in zip(np.atleast_1d(axes), clips):
        for codec, label, color in [("libx264", "H.264 (x264)", S1), ("libx265", "HEVC (x265)", S2)]:
            g = rd[(rd["clip"] == clip) & (rd.codec == codec)].sort_values("crf")
            ax.plot(g.crf, g.encode_fps, marker="o", markersize=6, color=color, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(sorted(rd.crf.unique()))
        ax.set_xlabel("CRF")
        ax.set_title(clip)
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel("encode fps (CPU, preset medium)")
    fig.suptitle("Encode speed: x264 vs x265 (higher is faster)", y=1.02, fontsize=11, color=INK2)
    save(fig, "encode_speed.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    bench = RESULTS / "benchmark_video.csv"
    if bench.exists():
        df = cpu_all_label(pd.read_csv(bench))
        fps_cpu_vs_gpu(df)
        end_to_end(df)
    kb = RESULTS / "kernel_bench.csv"
    if kb.exists():
        kb = pd.read_csv(kb)
        stage_breakdown(kb)
        blur_variants(kb)
    rd = RESULTS / "rd_results.csv"
    if rd.exists():
        rd_plots(pd.read_csv(rd))


if __name__ == "__main__":
    main()
