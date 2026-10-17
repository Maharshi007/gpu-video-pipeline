"""Create the test clips in data/ and list their resolution / frame count.

  python scripts/prepare_clips.py              # synthetic + Xiph crowd_run (120 frames)
  python scripts/prepare_clips.py --no-xiph    # synthetic only (no download)

The Xiph clip is streamed over HTTP and cut after N frames, so only ~370 MB
is downloaded instead of the full 1.5 GB sequence.
"""
import argparse
import subprocess

from common import DATA, ff_tool

XIPH = "http://media.xiph.org/video/derf/y4m/crowd_run_1080p50.y4m"


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--no-xiph", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    ff = ff_tool("ffmpeg")

    synth = DATA / "testsrc_1080p.y4m"
    if not synth.exists():
        run([ff, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=size=1920x1080:rate=30", "-frames:v", str(a.frames),
             "-pix_fmt", "yuv420p", str(synth)])
    real = DATA / f"crowd_run_{a.frames}f.y4m"
    if not a.no_xiph and not real.exists():
        run([ff, "-hide_banner", "-loglevel", "error", "-y", "-i", XIPH,
             "-frames:v", str(a.frames), str(real)])

    probe = ff_tool("ffprobe")
    for clip in sorted(DATA.glob("*.y4m")):
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                              "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                              "-of", "csv=p=0", str(clip)], capture_output=True, text=True).stdout
        print(f"{clip.name}: width,height,fps,frames = {out.strip()}")


if __name__ == "__main__":
    main()
