"""Shared paths/helpers for the analysis scripts."""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
BIN = ROOT / "build" / "bin"
OUT = ROOT / "build" / "out"  # encoded videos (gitignored)


def exe(name: str) -> str:
    """Path to one of our built executables."""
    p = BIN / (name + (".exe" if os.name == "nt" else ""))
    if not p.exists():
        raise SystemExit(f"{p} not found - build the project first")
    return str(p)


def ff_tool(name: str) -> str:
    """ffmpeg/ffprobe: prefer the bundled Windows build, else PATH."""
    local = ROOT / "third_party" / "ffmpeg" / "bin" / (name + ".exe")
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} not found")
    return found


def clips():
    found = sorted(DATA.glob("*.y4m"))
    if not found:
        raise SystemExit("no .y4m clips in data/ - run scripts/prepare_clips.py")
    return found
