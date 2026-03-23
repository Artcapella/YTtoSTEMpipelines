# YTtoSTEMpipelines

Download audio from YouTube videos and separate it into individual instrument stems using [Demucs](https://github.com/facebookresearch/demucs).

## Features

- **YouTube to stems in one command** — downloads audio via yt-dlp and runs Demucs stem separation
- **Auto model selection** — analyzes the audio to decide between 4-stem and 6-stem separation based on guitar/piano detection
- **Silent stem pruning** — automatically removes stems that are mostly silence (separation artifacts)
- **Stem normalization** — ensures all stems are stereo and at a consistent sample rate
- **Batch processing** — process multiple URLs from a text file
- **Scene JSON generation** — produces a starter config for the ConductorSBN adaptive mixer

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) (must be on PATH)

### Python packages

```
pip install yt-dlp demucs soundfile numpy librosa
```

> **Note:** A CUDA-capable GPU is recommended for faster stem separation but not required. The tool will automatically fall back to CPU.

## Usage

### Single video

```bash
# Basic — 4-stem separation (vocals, drums, bass, other)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Auto model — runs 4-stem first, upgrades to 6-stem if guitar/piano detected
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model auto

# Fine-tuned model (slower but better quality)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_ft

# 6-stem model (adds guitar and piano stems)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_6s

# Generate a scene.json for ConductorSBN
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --scene-json

# Custom output directory
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" -o my_scene/

# Download only (skip separation)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-only

# Separate an existing audio file (skip download)
python yt_to_stems.py --input existing_file.mp3 --scene-json
```

### Batch processing

1. Add YouTube URLs to `urls.txt` (one per line, `#` for comments).
2. Run:

```bash
python batch_run.py
```

Edit the constants at the top of `batch_run.py` to configure model, device, thresholds, and other options.

## Models

| Model | Stems | Speed | Quality | Notes |
|---|---|---|---|---|
| `htdemucs` | 4 (vocals, drums, bass, other) | Fast | Good | Default Demucs model |
| `htdemucs_ft` | 4 | ~4x slower | Better | Fine-tuned version (default) |
| `htdemucs_6s` | 6 (+ guitar, piano) | Moderate | Good | Piano quality can be weak |
| `mdx_extra_q` | 4 | Fast | Decent | Smaller download, quantized |
| `auto` | 4 or 6 | Variable | Best | Runs 4-stem first, analyzes "other" stem for guitar/piano content, re-runs with 6-stem if detected |

## Options

| Flag | Description |
|---|---|
| `--model`, `-m` | Demucs model to use (default: `htdemucs_ft`) |
| `--output`, `-o` | Output directory (default: `./stems/<video_title>/`) |
| `--device` | `auto`, `cuda`, or `cpu` (default: `auto`) |
| `--scene-json` | Generate a `scene.json` for ConductorSBN |
| `--bpm` | BPM to set in scene.json (default: 120) |
| `--input`, `-i` | Path to existing audio file (skips download) |
| `--download-only` | Download audio without separating |
| `--keep-original` | Keep the downloaded WAV alongside stems |
| `--threshold` | Guitar/piano detection sensitivity for `auto` mode (0.0–1.0, default: 0.3) |
| `--no-prune` | Keep all stems, even silent/artifact ones |
| `--silence-threshold` | dBFS floor for silent stem pruning (default: -40.0) |
| `--verbose`, `-v` | Print detailed output |

## Output structure

```
stems/
  Video Title/
    vocals.wav
    drums.wav
    bass.wav
    other.wav
    guitar.wav      # only with 6-stem model
    piano.wav       # only with 6-stem model
    scene.json      # only with --scene-json
```

## How auto model selection works

When using `--model auto`, the tool:

1. Runs the fine-tuned 4-stem model (`htdemucs_ft`)
2. Analyzes the "other" stem using spectral features (onset density, spectral flatness, percussive energy ratio, and pitch clarity)
3. If the composite score exceeds the threshold, re-runs with the 6-stem model (`htdemucs_6s`) to extract guitar and piano separately
4. Otherwise, keeps the 4-stem result

The `--threshold` flag controls sensitivity — lower values trigger 6-stem separation more often.
