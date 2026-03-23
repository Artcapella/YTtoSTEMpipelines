#!/usr/bin/env python3
"""
batch_run.py — Run yt_to_stems.py on every URL listed in urls.txt.

Usage:
    1. Add one YouTube URL per line to urls.txt (lines starting with # are ignored).
    2. Press the Run button in VSCode (or run: python batch_run.py).

Configuration:
    Edit the constants below to change model, output directory, etc.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
URL_FILE      = "urls.txt"   # Path to the file containing YouTube URLs
MODEL         = "auto"       # auto | htdemucs | htdemucs_ft | htdemucs_6s | mdx_extra_q
                             #   auto: runs 4-stem first, upgrades to 6-stem if
                             #         guitar/piano are detected in the "other" stem
DEVICE        = "auto"       # auto | cuda | cpu
SCENE_JSON    = True         # Generate a scene.json for each track
OUTPUT_DIR    = None         # None = auto-name from video title; or e.g. "stems/my_batch"
KEEP_ORIGINAL       = False  # Keep the downloaded WAV alongside the stems
THRESHOLD           = 0.25   # Guitar/piano sensitivity for --model auto (0.0-1.0)
                             #   lower = triggers 6-stem more often
                             #   higher = only upgrades on very guitar/piano-heavy tracks
NO_PRUNE            = False  # Set True to keep all stems, even silent/artifact ones
SILENCE_THRESHOLD   = -40.0  # dBFS floor for silent stem pruning (default: -40.0)
# ──────────────────────────────────────────────────────────────────────────────

import subprocess
import sys
from pathlib import Path


def load_urls(path: str) -> list[str]:
    urls = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)
    except FileNotFoundError:
        print(f"ERROR: URL file not found: {path}")
        print(f"Create '{path}' and add one YouTube URL per line.")
        sys.exit(1)
    return urls


def run_one(url: str, index: int, total: int) -> bool:
    print(f"\n{'='*60}")
    print(f"  [{index}/{total}] {url}")
    print(f"{'='*60}")

    cmd = [sys.executable, "yt_to_stems.py", url, "--model", MODEL, "--device", DEVICE]

    if SCENE_JSON:
        cmd.append("--scene-json")
    if OUTPUT_DIR:
        cmd += ["--output", OUTPUT_DIR]
    if KEEP_ORIGINAL:
        cmd.append("--keep-original")
    if MODEL == "auto":
        cmd += ["--threshold", str(THRESHOLD)]
    if NO_PRUNE:
        cmd.append("--no-prune")
    if SILENCE_THRESHOLD != -40.0:
        cmd += ["--silence-threshold", str(SILENCE_THRESHOLD)]

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    urls = load_urls(URL_FILE)

    if not urls:
        print(f"No URLs found in '{URL_FILE}'. Add at least one YouTube URL and re-run.")
        sys.exit(0)

    total = len(urls)
    print(f"Batch run: {total} URL{'s' if total != 1 else ''} found in '{URL_FILE}'")

    succeeded = []
    failed = []

    for i, url in enumerate(urls, start=1):
        ok = run_one(url, i, total)
        if ok:
            succeeded.append(url)
        else:
            failed.append(url)
            print(f"  !! Failed: {url}")

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE  —  {len(succeeded)}/{total} succeeded")
    if failed:
        print(f"\nFailed URLs ({len(failed)}):")
        for url in failed:
            print(f"  {url}")
    print()


if __name__ == "__main__":
    main()
