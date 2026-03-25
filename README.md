# YTtoSTEMpipelines

A Python pipeline that downloads audio from YouTube and separates it into individual instrument stems — ready for use in adaptive music systems, game audio, remixing, or music analysis.

Built as the audio acquisition and preprocessing layer for the **ConductorSBN** adaptive mixer.

---

## What it does

Given a YouTube URL, the pipeline:

1. Downloads the best available audio stream via **yt-dlp**
2. Converts it to uncompressed WAV with **ffmpeg**
3. Separates it into instrument stems using **Facebook's Demucs** neural network
4. Normalizes all stems to a consistent sample rate and channel layout
5. Prunes silent/artifact stems automatically
6. Optionally generates a `scene.json` config for the ConductorSBN adaptive mixer

All of this happens in a single command.

---

## Technical highlights

### Intelligent model selection (`--model auto`)

Rather than always running the heavier 6-stem model, the pipeline runs a lightweight 4-stem pass first and then analyzes the "other" stem using a multi-feature spectral classifier to decide whether guitar or piano are present:

| Feature | What it detects |
|---|---|
| **Onset density** | Note attack rate — guitar/piano produce dense, distinct transients; pads and strings do not |
| **Spectral flatness** | Plucked/struck instruments have broader spectral energy than sustained tones |
| **Percussive energy ratio** | librosa HPSS separates transient vs. sustained energy — guitar/piano score high here |
| **Pitch clarity in guitar/piano ranges** | Checks for strong pitched content in the 80–4200 Hz band typical of these instruments |

These are combined into a composite score (0.0–1.0). If the score exceeds `--threshold` (default 0.3), the pipeline re-runs with `htdemucs_6s` to extract guitar and piano as separate stems. Otherwise the 4-stem result is kept, saving significant compute time.

### Silent stem pruning

Demucs always outputs every stem it was trained for, even when an instrument isn't present — a vocal stem on an instrumental track becomes faint ghosting artifacts, not silence. The pruner catches this with two criteria applied per stem:

- **Overall RMS** below a configurable dBFS floor
- **Active frame ratio** below 5% (fewer than 1 in 20 analysis frames above the floor)

Stems that fail either check are deleted, preventing downstream systems from loading meaningless files.

### Dual-mode Demucs integration

The pipeline targets the **Demucs Python API** (`demucs.api.Separator`) for clean, in-process separation with direct tensor access. If the API isn't available (older Demucs versions), it automatically falls back to the CLI (`demucs.separate.main`) and normalizes the output directory structure to match, so the rest of the pipeline is unaffected.

### Rate limit detection

YouTube throttles download IPs under heavy load. The pipeline detects HTTP 429 / Too Many Requests errors from yt-dlp and surfaces them as a distinct, actionable warning rather than a generic failure — including suggested mitigations (wait period, VPN, cookie-based auth, batch pacing).

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) on PATH

```
pip install yt-dlp demucs soundfile numpy librosa
```

> A CUDA GPU is recommended for faster separation but not required — the pipeline detects and falls back to CPU automatically.

---

## Usage

### Single video

```bash
# 4-stem separation (vocals, drums, bass, other)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Auto model — 4-stem first, upgrades to 6-stem if guitar/piano detected
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model auto

# Fine-tuned model (slower but better quality)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_ft

# 6-stem model (adds guitar and piano stems explicitly)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_6s

# Generate a scene.json for ConductorSBN
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --scene-json

# Custom output directory
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" -o my_scene/

# Separate an existing audio file (skip download)
python yt_to_stems.py --input existing_file.mp3 --scene-json

# Download only (skip separation)
python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-only
```

### Batch processing

1. Add YouTube URLs to `urls.txt` (one per line, `#` to comment out lines).
2. Run:

```bash
python batch_run.py
```

Configure model, device, thresholds, and output directory via the constants at the top of `batch_run.py`. The batch runner streams live output for each video and prints a summary of succeeded/failed URLs on completion.

---

## Models

| Model | Stems | Speed | Notes |
|---|---|---|---|
| `htdemucs` | 4 (vocals, drums, bass, other) | Fast | Default Demucs model |
| `htdemucs_ft` | 4 | ~4x slower | Fine-tuned — better quality (pipeline default) |
| `htdemucs_6s` | 6 (+ guitar, piano) | Moderate | Piano separation quality is limited |
| `mdx_extra_q` | 4 | Fast | Smaller download, quantized |
| `auto` | 4 or 6 | Variable | Runs 4-stem, analyzes "other" stem, upgrades only if needed |

---

## Options

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `htdemucs_ft` | Demucs model to use |
| `--output`, `-o` | `./stems/<title>/` | Output directory |
| `--device` | `auto` | `auto`, `cuda`, or `cpu` |
| `--scene-json` | off | Generate a `scene.json` for ConductorSBN |
| `--bpm` | `120` | BPM to embed in scene.json |
| `--input`, `-i` | — | Existing audio file (skips download) |
| `--download-only` | off | Download WAV without separating |
| `--keep-original` | off | Keep the downloaded WAV alongside stems |
| `--threshold` | `0.3` | Guitar/piano detection sensitivity for `auto` mode (0.0–1.0) |
| `--no-prune` | off | Keep all stems, including silent/artifact ones |
| `--silence-threshold` | `-40.0` | dBFS floor for silent stem pruning |
| `--verbose`, `-v` | off | Print detailed output including per-stem energy metrics |

---

## Output structure

```
stems/
  Video Title/
    vocals.wav
    drums.wav
    bass.wav
    other.wav
    guitar.wav      # 6-stem model only
    piano.wav       # 6-stem model only
    scene.json      # --scene-json only
```

All stems are normalized to 44.1 kHz stereo WAV.

---

## How auto model selection works

```
URL → 4-stem separation (htdemucs_ft)
           │
           ▼
    Analyze "other" stem
    ┌─────────────────────────────┐
    │ onset density               │
    │ spectral flatness           │  → composite score (0.0–1.0)
    │ percussive energy ratio     │
    │ pitch clarity (80–4200 Hz)  │
    └─────────────────────────────┘
           │
    score > threshold?
    ├── No  → keep 4-stem result
    └── Yes → re-run with htdemucs_6s → 6-stem result
```

The `--threshold` flag shifts the decision boundary. Lower values (e.g. `0.1`) trigger 6-stem separation more aggressively; higher values (e.g. `0.6`) reserve it for tracks with prominent, sustained guitar or piano parts.

---

## Troubleshooting

### Rate limited by YouTube (HTTP 429)

YouTube throttles IPs that make too many download requests in a short window. The pipeline detects this and prints a `WARNING: RATE LIMITED` banner rather than a generic failure.

**Fixes, in order of simplicity:**

1. **Wait and retry** — 15–30 minutes is usually enough for the throttle to lift.
2. **Use a VPN or different network** — the rate limit is per IP, so switching networks bypasses it immediately.
3. **Pass browser cookies to yt-dlp** — authenticated requests are far less likely to be throttled:
   ```bash
   # Add to yt_to_stems.py ydl_opts:
   "cookiesfrombrowser": ("chrome",),   # or "firefox", "edge", etc.
   ```
   Or export cookies manually and pass `"cookiefile": "cookies.txt"`.
4. **Reduce batch speed** — if running `batch_run.py` on many URLs, add a `time.sleep(10)` between calls in `run_one` to space out requests.

---

### `No WAV file found after download`

ffmpeg ran but produced no output. Common causes:

- **ffmpeg not on PATH** — verify with `ffmpeg -version`. On Windows, the ffmpeg `bin/` folder must be added to your system PATH.
- **Unsupported URL** — yt-dlp may not support the video's region or format. Run with `--verbose` to see the raw yt-dlp output.
- **Private or age-gated video** — requires passing cookies (see above).

---

### Demucs separation is very slow

- Separation runs on CPU by default if no CUDA GPU is detected. Check `Device: cpu` vs `Device: cuda` in the output.
- If you have an NVIDIA GPU, ensure PyTorch is installed with CUDA support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
- `htdemucs_ft` (the default) is ~4x slower than `htdemucs`. Use `--model htdemucs` for faster results at slightly lower quality.

---

### Demucs downloads a model on first run

Each Demucs model (~80–150 MB) is downloaded from the internet on first use and cached locally. This is normal. Subsequent runs use the cache and start immediately.

---

### All stems are pruned / output directory is empty

The silence pruner removed everything, which usually means separation failed silently or produced a near-zero output.

- Run with `--no-prune` to skip pruning and inspect the raw stems.
- Run with `--verbose` to see per-stem dBFS readings.
- If all stems are genuinely near-silent, the source audio may be corrupted or extremely quiet — check the downloaded WAV before separation.

---

### `stem_paths` missing `guitar` or `piano` with `--model auto`

These stems only appear when the auto classifier scores the "other" stem above `--threshold`. If you expect guitar or piano but they aren't being extracted:

- Lower the threshold: `--threshold 0.1`
- Or force the 6-stem model directly: `--model htdemucs_6s`

---

## Dependencies

| Package | Role |
|---|---|
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube download and audio extraction |
| [Demucs](https://github.com/facebookresearch/demucs) | Neural source separation (Facebook Research) |
| [librosa](https://librosa.org/) | Spectral feature extraction for auto model selection |
| [soundfile](https://python-soundfile.readthedocs.io/) | WAV read/write for normalization and pruning |
| [NumPy](https://numpy.org/) | Array operations throughout |
| [ffmpeg](https://ffmpeg.org/) | Audio conversion and resampling |
