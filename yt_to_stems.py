#!/usr/bin/env python3
"""
yt_to_stems.py — Download audio from a YouTube URL and separate it into stems.

Downloads the best available audio via yt-dlp, converts to WAV, then runs
Demucs stem separation to produce individual instrument tracks ready for
use as layers in the ConductorSBN adaptive mixer.

Requirements:
    pip install yt-dlp demucs soundfile numpy librosa
    # Also needs ffmpeg installed on your system:
    #   Ubuntu/Debian: sudo apt install ffmpeg
    #   macOS:         brew install ffmpeg
    #   Windows:       https://ffmpeg.org/download.html (add to PATH)

Usage:
    # Basic — downloads and splits into 4 stems (vocals, drums, bass, other)
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Auto-detect whether to use 4-stem or 6-stem model:
    # Runs 4-stem first, analyzes the "other" stem for guitar/piano content,
    # and re-runs with the 6-stem model if those instruments are prominent.
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model auto

    # Use the fine-tuned model (slower but better quality)
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_ft

    # Use the 6-source model (adds guitar and piano stems)
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_6s

    # Specify output directory
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" -o my_scene/

    # Also generate a starter scene.json for the adaptive mixer
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --scene-json

    # Use CPU only (no CUDA GPU)
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --device cpu

    # Download only (skip stem separation)
    python yt_to_stems.py "https://www.youtube.com/watch?v=VIDEO_ID" --download-only

    # Separate an already-downloaded file (skip download)
    python yt_to_stems.py --input existing_file.mp3 --scene-json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_dependencies():
    """Verify required tools are available."""
    errors = []

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        errors.append("yt-dlp not found. Install with: pip install yt-dlp")

    try:
        import demucs  # noqa: F401
    except ImportError:
        errors.append("demucs not found. Install with: pip install demucs")

    try:
        import soundfile  # noqa: F401
    except ImportError:
        errors.append("soundfile not found. Install with: pip install soundfile")

    try:
        import librosa  # noqa: F401
    except ImportError:
        errors.append("librosa not found. Install with: pip install librosa")

    # Check ffmpeg
    if shutil.which("ffmpeg") is None:
        errors.append(
            "ffmpeg not found on PATH. Install it:\n"
            "  Ubuntu/Debian: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg\n"
            "  Windows:       https://ffmpeg.org/download.html"
        )

    if errors:
        print("Missing dependencies:\n")
        for err in errors:
            print(f"  • {err}\n")
        sys.exit(1)


def download_audio(url: str, output_dir: str, verbose: bool = False) -> str:
    """
    Download audio from a YouTube URL using yt-dlp.

    Downloads the best available audio stream and converts it to WAV format
    at the original sample rate. WAV is used because Demucs works best with
    uncompressed input.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded file.
        verbose: Print yt-dlp output.

    Returns:
        Path to the downloaded WAV file.
    """
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    # Use a temp name to avoid filesystem issues with special characters in titles
    # We'll rename afterward using the sanitized title
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    print(f"Downloading audio from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")

    # Find the downloaded WAV file
    # yt-dlp sanitizes the filename, so we search for .wav files in the output dir
    wav_files = list(Path(output_dir).glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV file found in {output_dir} after download. "
            "Check that ffmpeg is installed and working."
        )

    # Use the most recently modified WAV file
    downloaded_path = str(max(wav_files, key=lambda p: p.stat().st_mtime))

    print(f"Downloaded: {downloaded_path}")
    print(f"Title: {title}")

    return downloaded_path


def separate_stems(
    audio_path: str,
    output_dir: str,
    model: str = "htdemucs_ft",
    device: str = "auto",
    verbose: bool = False,
) -> dict[str, str]:
    """
    Run Demucs stem separation on an audio file.

    Uses the Demucs Python API (demucs.api.Separator) for clean integration.
    Falls back to the CLI if the API isn't available (older demucs versions).

    Args:
        audio_path: Path to the input audio file (WAV, MP3, FLAC, etc.).
        output_dir: Directory to save the separated stems.
        model: Demucs model name. Options:
            - "htdemucs"     — Default, 4 stems (vocals, drums, bass, other). Fast.
            - "htdemucs_ft"  — Fine-tuned, 4 stems. ~4x slower but better quality.
            - "htdemucs_6s"  — 6 stems (adds guitar, piano). Piano quality is weak.
            - "mdx_extra_q"  — Quantized MDX model. Smaller download, decent quality.
        device: "auto" (use GPU if available), "cuda", or "cpu".
        verbose: Print progress.

    Returns:
        Dict mapping stem names to their file paths, e.g.:
        {"vocals": "/path/to/vocals.wav", "drums": "/path/to/drums.wav", ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = str(Path(audio_path).resolve())
    track_name = Path(audio_path).stem

    print(f"\nSeparating stems with model '{model}'...")
    print(f"Input: {audio_path}")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Device: {device}")

    # Try the Python API first (demucs >= 4.0)
    try:
        import demucs.api

        separator = demucs.api.Separator(model=model, device=device)
        print("Using Demucs Python API...")
        origin, separated = separator.separate_audio_file(audio_path)

        stem_paths = {}
        for stem_name, stem_audio in separated.items():
            stem_filename = f"{stem_name}.wav"
            stem_path = os.path.join(output_dir, stem_filename)
            demucs.api.save_audio(stem_audio, stem_path, samplerate=separator.samplerate)
            stem_paths[stem_name] = stem_path
            print(f"  Saved: {stem_filename}")

        return stem_paths

    except (ImportError, AttributeError):
        # Fall back to CLI approach for older demucs versions
        print("Demucs API not available, falling back to CLI...")

    # CLI fallback — uses demucs.separate.main()
    cli_output_dir = os.path.join(output_dir, "_demucs_raw")
    try:
        import demucs.separate

        cli_args = [
            "--name", model,
            "--out", cli_output_dir,
            "--device", device,
        ]
        if not verbose:
            cli_args.append("--verbose")  # demucs CLI verbose is actually needed for progress
        cli_args.append(audio_path)

        demucs.separate.main(cli_args)
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}") from e

    # Find the output stems from the CLI
    # Demucs CLI outputs to: {out_dir}/{model_name}/{track_name}/{stem}.wav
    stems_dir = Path(cli_output_dir) / model / track_name
    if not stems_dir.exists():
        # Some models use different naming — search for the track name
        for candidate in Path(cli_output_dir).rglob("*.wav"):
            if candidate.parent.name == track_name or track_name in str(candidate):
                stems_dir = candidate.parent
                break

    if not stems_dir.exists():
        raise FileNotFoundError(
            f"Demucs output not found. Expected at {stems_dir}. "
            f"Contents of {cli_output_dir}: {list(Path(cli_output_dir).rglob('*'))}"
        )

    # Move stem files to the output directory and build the result dict
    stem_paths = {}
    for stem_file in sorted(stems_dir.glob("*.wav")):
        stem_name = stem_file.stem  # e.g., "vocals", "drums", "bass", "other"
        dest = Path(output_dir) / f"{stem_name}.wav"
        shutil.copy2(str(stem_file), str(dest))
        stem_paths[stem_name] = str(dest)
        print(f"  Saved: {stem_name}.wav")

    # Clean up the intermediate directory
    shutil.rmtree(cli_output_dir, ignore_errors=True)

    return stem_paths


def normalize_stems(stem_paths: dict[str, str], target_sr: int = 44100):
    """
    Ensure all stems have the same sample rate and are stereo.
    Overwrites files in place.
    """
    import soundfile as sf
    import numpy as np

    print("\nNormalizing stems...")

    for stem_name, path in stem_paths.items():
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        changed = False

        # Ensure stereo
        if data.shape[1] == 1:
            data = np.column_stack([data[:, 0], data[:, 0]])
            changed = True
            print(f"  {stem_name}: mono → stereo")

        if data.shape[1] > 2:
            data = data[:, :2]
            changed = True
            print(f"  {stem_name}: {data.shape[1]}ch → stereo")

        # Check sample rate (resampling with ffmpeg if needed)
        if sr != target_sr:
            print(f"  {stem_name}: resampling {sr} → {target_sr} Hz")
            temp_path = path + ".tmp.wav"
            sf.write(temp_path, data, sr, subtype="FLOAT")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", temp_path,
                    "-ar", str(target_sr),
                    "-ac", "2",
                    path,
                ],
                capture_output=True,
            )
            os.remove(temp_path)
            changed = True
        elif changed:
            sf.write(path, data, sr, subtype="FLOAT")

        if not changed:
            print(f"  {stem_name}: OK ({sr} Hz, {data.shape[1]}ch, {data.shape[0]/sr:.1f}s)")


def prune_silent_stems(
    stem_paths: dict[str, str],
    silence_threshold_db: float = -40.0,
    min_active_ratio: float = 0.05,
    verbose: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """
    Remove stems that are mostly or entirely silent (separation artifacts).

    A stem is considered silent if either:
      - Its overall RMS energy is below silence_threshold_db, OR
      - Less than min_active_ratio of its frames exceed the silence floor.

    This catches cases like a "vocals" stem on an instrumental track, which
    Demucs will populate with faint ghosting artifacts rather than true silence.

    Args:
        stem_paths: Dict mapping stem names to file paths.
        silence_threshold_db: RMS floor in dBFS. Stems below this are removed.
            Default -40 dB is conservative — real content is typically > -30 dB.
        min_active_ratio: Minimum fraction of frames that must exceed the
            silence floor. Default 0.05 means at least 5% of frames must be
            above the threshold or the stem is removed.
        verbose: Print per-stem energy metrics.

    Returns:
        Tuple of (surviving_stem_paths, removed_stem_names).
    """
    import soundfile as sf
    import numpy as np

    print("\nPruning silent stems...")

    surviving = {}
    removed = []

    for stem_name, path in stem_paths.items():
        data, _ = sf.read(path, dtype="float32", always_2d=True)

        # Overall RMS in dBFS
        rms = float(np.sqrt(np.mean(data ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-9)

        # Frame-level active ratio: fraction of 2048-sample frames above threshold
        frame_size = 2048
        mono = data.mean(axis=1)
        n_frames = len(mono) // frame_size
        if n_frames > 0:
            frames = mono[: n_frames * frame_size].reshape(n_frames, frame_size)
            frame_rms_db = 20.0 * np.log10(
                np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9
            )
            active_ratio = float(np.mean(frame_rms_db > silence_threshold_db))
        else:
            active_ratio = 0.0

        is_silent = rms_db < silence_threshold_db or active_ratio < min_active_ratio

        if verbose:
            status = "SILENT" if is_silent else "OK"
            print(
                f"  {stem_name:>8}: {rms_db:+.1f} dBFS, "
                f"{active_ratio*100:.1f}% active frames  [{status}]"
            )

        if is_silent:
            try:
                os.remove(path)
            except OSError:
                pass
            removed.append(stem_name)
            print(f"  Pruned (silent): {stem_name}.wav  ({rms_db:+.1f} dBFS, "
                  f"{active_ratio*100:.1f}% active frames)")
        else:
            surviving[stem_name] = path
            if not verbose:
                print(f"  Kept: {stem_name}.wav  ({rms_db:+.1f} dBFS)")

    if removed:
        print(f"  Removed {len(removed)} silent stem(s): {', '.join(removed)}")
    else:
        print("  No silent stems found.")

    return surviving, removed


def analyze_other_stem(
    other_stem_path: str,
    guitar_piano_threshold: float = 0.3,
    verbose: bool = False,
) -> dict:
    """
    Analyze the "other" stem from a 4-source separation to determine whether
    guitar and/or piano are prominent enough to warrant re-running with the
    6-source model.

    The approach uses librosa to extract spectral features that distinguish
    percussive/plucked instruments (guitar, piano) from sustained instruments
    (strings, synths, pads):

    1. Onset density: Guitar and piano produce distinct note attacks. Strings
       and synths tend toward sustained, legato textures. High onset density
       relative to the track's tempo suggests plucked/struck instruments.

    2. Spectral flatness: Measures how noise-like vs. tonal the signal is.
       Guitar (especially acoustic) has higher spectral flatness than
       bowed strings due to the broadband energy of plucked transients.

    3. Percussive energy ratio: librosa's harmonic-percussive separation
       estimates how much of the signal is transient vs. sustained. Guitar
       and piano have strong percussive transients from picking/hammering.
       Pads and bowed strings are almost entirely harmonic.

    4. Pitch clarity in guitar/piano ranges: Checks whether strong pitched
       content exists in the fundamental frequency ranges typical of guitar
       (~80-1200 Hz) and piano (~27-4186 Hz but especially mid-range).

    We combine these into a composite "guitar/piano prominence" score.

    Args:
        other_stem_path: Path to the "other" stem WAV file.
        guitar_piano_threshold: Score above which we recommend 6-source model.
            Range 0.0-1.0. Default 0.3 is moderately conservative — it avoids
            triggering on orchestral music with pizzicato strings but catches
            tracks with sustained guitar or piano parts.
        verbose: Print detailed analysis metrics.

    Returns:
        Dict with:
            "recommend_6s": bool — whether to re-run with htdemucs_6s
            "score": float — composite guitar/piano prominence (0.0-1.0)
            "onset_density": float — note onsets per second
            "percussive_ratio": float — fraction of energy in transients
            "spectral_flatness_mean": float — avg spectral flatness
            "details": str — human-readable summary
    """
    import librosa
    import numpy as np

    print("\nAnalyzing 'other' stem for guitar/piano content...")

    # Load audio (mono, at native sample rate)
    y, sr = librosa.load(other_stem_path, sr=None, mono=True)
    duration = len(y) / sr

    # --- Feature 1: Onset density ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    onset_density = len(onset_times) / duration  # onsets per second

    # Normalize onset density: ~0-2 onsets/sec is pad-like, 3+ is plucked/struck
    # Map to 0-1 range with sigmoid-like curve centered around 2.5 onsets/sec
    onset_score = 1.0 / (1.0 + np.exp(-2.0 * (onset_density - 2.5)))

    # --- Feature 2: Spectral flatness ---
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    flatness_mean = float(np.mean(spectral_flatness))

    # Strings/pads typically have flatness < 0.01, guitar 0.02-0.10
    # Map: 0.005 -> 0.0, 0.05 -> 1.0
    flatness_score = np.clip((flatness_mean - 0.005) / 0.045, 0.0, 1.0)

    # --- Feature 3: Percussive energy ratio ---
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(y_harmonic ** 2))
    percussive_energy = float(np.sum(y_percussive ** 2))
    total_energy = harmonic_energy + percussive_energy

    if total_energy > 0:
        percussive_ratio = percussive_energy / total_energy
    else:
        percussive_ratio = 0.0

    # Sustained instruments: percussive_ratio < 0.15
    # Guitar/piano: percussive_ratio > 0.3
    # Map: 0.10 -> 0.0, 0.35 -> 1.0
    percussive_score = np.clip((percussive_ratio - 0.10) / 0.3, 0.0, 1.0)

    # --- Feature 4: Pitch clarity in guitar/piano range ---
    # Check if there's strong pitched content in the mid-range (200-2000 Hz)
    # where guitar and piano fundamentals live, using the chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # High chroma variance across time = distinct note changes (plucked/struck)
    # Low variance = sustained chords/pads
    chroma_variance = float(np.mean(np.var(chroma, axis=1)))

    # Normalize: pads ~0.01-0.03, guitar/piano ~0.04-0.15
    chroma_score = np.clip((chroma_variance - 0.02) / 0.08, 0.0, 1.0)

    # --- Composite score ---
    # Weighted combination. Percussive ratio and onset density are the strongest
    # indicators; spectral flatness and chroma variance are supporting evidence.
    composite = (
        0.35 * percussive_score
        + 0.30 * onset_score
        + 0.20 * flatness_score
        + 0.15 * chroma_score
    )
    composite = float(np.clip(composite, 0.0, 1.0))

    recommend = composite >= guitar_piano_threshold

    # Build human-readable summary
    if recommend:
        detail_parts = []
        if percussive_score > 0.3:
            detail_parts.append("strong transient attacks (plucked/struck)")
        if onset_score > 0.3:
            detail_parts.append(f"high onset density ({onset_density:.1f}/sec)")
        if flatness_score > 0.3:
            detail_parts.append("broadband spectral character")
        if chroma_score > 0.3:
            detail_parts.append("distinct pitch changes")
        details = (
            f"Guitar/piano content DETECTED (score: {composite:.2f}). "
            f"Indicators: {', '.join(detail_parts) if detail_parts else 'composite threshold exceeded'}. "
            f"Recommending re-run with htdemucs_6s."
        )
    else:
        details = (
            f"Guitar/piano content NOT prominent (score: {composite:.2f}). "
            f"The 'other' stem appears to be primarily sustained/orchestral content. "
            f"Keeping 4-stem separation."
        )

    if verbose:
        print(f"  Onset density:     {onset_density:.2f}/sec (score: {onset_score:.2f})")
        print(f"  Spectral flatness: {flatness_mean:.4f} (score: {flatness_score:.2f})")
        print(f"  Percussive ratio:  {percussive_ratio:.3f} (score: {percussive_score:.2f})")
        print(f"  Chroma variance:   {chroma_variance:.4f} (score: {chroma_score:.2f})")
        print(f"  Composite score:   {composite:.2f} (threshold: {guitar_piano_threshold})")

    print(f"  → {details}")

    return {
        "recommend_6s": recommend,
        "score": composite,
        "onset_density": onset_density,
        "percussive_ratio": percussive_ratio,
        "spectral_flatness_mean": flatness_mean,
        "chroma_variance": chroma_variance,
        "details": details,
    }


def separate_with_auto_model(
    audio_path: str,
    output_dir: str,
    device: str = "auto",
    guitar_piano_threshold: float = 0.3,
    verbose: bool = False,
) -> tuple[dict[str, str], str]:
    """
    Two-pass model selection: run htdemucs first, analyze the "other" stem,
    and re-run with htdemucs_6s if guitar/piano are prominent.

    Args:
        audio_path: Path to input audio.
        output_dir: Final output directory for stems.
        device: Compute device.
        guitar_piano_threshold: Analysis threshold for triggering 6-source model.
        verbose: Print detailed output.

    Returns:
        Tuple of (stem_paths dict, model_used string).
    """
    # Pass 1: Run 4-stem separation
    print("\n" + "=" * 60)
    print("AUTO MODEL SELECTION — Pass 1: 4-stem separation")
    print("=" * 60)

    pass1_dir = os.path.join(output_dir, "_auto_pass1")
    os.makedirs(pass1_dir, exist_ok=True)

    stem_paths_4 = separate_stems(
        audio_path=audio_path,
        output_dir=pass1_dir,
        model="htdemucs_ft",
        device=device,
        verbose=verbose,
    )

    # Analyze the "other" stem
    other_path = stem_paths_4.get("other")
    if other_path is None or not os.path.exists(other_path):
        print("Warning: No 'other' stem found. Keeping 4-stem results.")
        final_paths = {}
        for name, path in stem_paths_4.items():
            dest = os.path.join(output_dir, f"{name}.wav")
            shutil.move(path, dest)
            final_paths[name] = dest
        shutil.rmtree(pass1_dir, ignore_errors=True)
        return final_paths, "htdemucs_ft"

    analysis = analyze_other_stem(
        other_path,
        guitar_piano_threshold=guitar_piano_threshold,
        verbose=verbose,
    )

    if not analysis["recommend_6s"]:
        # Keep 4-stem results — move to final output dir
        print("\nKeeping 4-stem separation.")
        final_paths = {}
        for name, path in stem_paths_4.items():
            dest = os.path.join(output_dir, f"{name}.wav")
            shutil.move(path, dest)
            final_paths[name] = dest
        shutil.rmtree(pass1_dir, ignore_errors=True)
        return final_paths, "htdemucs_ft"

    # Pass 2: Re-run with 6-stem model
    print("\n" + "=" * 60)
    print("AUTO MODEL SELECTION — Pass 2: 6-stem separation")
    print("=" * 60)

    shutil.rmtree(pass1_dir, ignore_errors=True)

    stem_paths_6 = separate_stems(
        audio_path=audio_path,
        output_dir=output_dir,
        model="htdemucs_6s",
        device=device,
        verbose=verbose,
    )

    return stem_paths_6, "htdemucs_6s"


def generate_scene_json(
    stem_paths: dict[str, str],
    output_dir: str,
    scene_name: str = "Untitled Scene",
    bpm: int = 120,
):
    """
    Generate a starter scene.json for the ConductorSBN adaptive mixer.

    Maps Demucs stem names to mixer layer groups with sensible defaults:
        - "other" → base layer (always on) — contains melodic/harmonic content
        - "bass"  → base layer (always on) — low end foundation
        - "vocals" → peaceful/melodic layer — choir, vocals, melody
        - "drums" → combat/intensity layer — percussion
        - "guitar" → tension layer (if 6-source model)
        - "piano"  → peaceful layer (if 6-source model)
    """
    stem_layer_map = {
        "other":  {"layer": "base",    "default_volume": 0.6, "always_on": True,
                   "description": "Melodic and harmonic content (strings, synths, etc.)"},
        "bass":   {"layer": "base",    "default_volume": 0.4, "always_on": True,
                   "description": "Bass instruments and low-end foundation"},
        "vocals": {"layer": "melody",  "default_volume": 0.5, "always_on": False,
                   "description": "Vocal parts, choir, or lead melody"},
        "drums":  {"layer": "combat",  "default_volume": 0.7, "always_on": False,
                   "description": "Percussion and rhythmic elements"},
        "guitar": {"layer": "tension", "default_volume": 0.5, "always_on": False,
                   "description": "Guitar parts"},
        "piano":  {"layer": "peaceful","default_volume": 0.4, "always_on": False,
                   "description": "Piano parts"},
    }

    stems_config = {}
    layer_groups = {}

    for stem_name, stem_path in stem_paths.items():
        filename = Path(stem_path).name
        mapping = stem_layer_map.get(stem_name, {
            "layer": stem_name,
            "default_volume": 0.5,
            "always_on": False,
            "description": f"Separated {stem_name} stem",
        })

        stems_config[stem_name] = {
            "file": filename,
            "layer": mapping["layer"],
            "default_volume": mapping["default_volume"],
            "always_on": mapping["always_on"],
            "description": mapping["description"],
        }

        layer = mapping["layer"]
        if layer not in layer_groups:
            intensity_map = {"base": 0, "peaceful": 1, "melody": 1, "tension": 2, "combat": 3}
            layer_groups[layer] = {
                "stems": [],
                "intensity": intensity_map.get(layer, 1),
            }
        layer_groups[layer]["stems"].append(stem_name)

    scene = {
        "name": scene_name,
        "bpm": bpm,
        "key": "unknown",
        "time_signature": [4, 4],
        "stems": stems_config,
        "layer_groups": layer_groups,
        "effects": {
            "other": {"reverb_room_size": 0.4},
        },
        "_notes": [
            "This scene.json was auto-generated by yt_to_stems.py.",
            "You should adjust: bpm, key, layer assignments, volumes, and effects.",
            "The layer mapping is a starting point — reorganize based on the actual music.",
            "Tip: listen to each stem individually to decide which layers make sense.",
        ],
    }

    scene_path = os.path.join(output_dir, "scene.json")
    with open(scene_path, "w") as f:
        json.dump(scene, f, indent=4)

    print(f"\nGenerated: {scene_path}")
    print("Review and adjust the layer assignments, BPM, and key for your scene.")

    return scene_path


def print_summary(stem_paths: dict[str, str], output_dir: str, scene_json: bool,
                  model_used: str = ""):
    """Print a summary of what was created."""
    import soundfile as sf

    print("\n" + "=" * 60)
    print("STEM SEPARATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    if model_used:
        print(f"Model used: {model_used}")
    print(f"\nStems created ({len(stem_paths)}):")

    for stem_name, path in sorted(stem_paths.items()):
        try:
            info = sf.info(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {stem_name:>8}.wav  —  {info.duration:.1f}s, "
                  f"{info.samplerate}Hz, {info.channels}ch, {size_mb:.1f}MB")
        except Exception:
            print(f"  {stem_name:>8}.wav")

    if scene_json:
        print(f"\n  scene.json  —  Starter config for ConductorSBN adaptive mixer")

    print(f"\nNext steps:")
    print(f"  1. Listen to each stem to understand what's in it")
    if scene_json:
        print(f"  2. Edit scene.json — set the correct BPM, key, and adjust layer assignments")
        print(f"  3. Copy the folder to assets/music/scenes/ in ConductorSBN")
    else:
        print(f"  2. Run again with --scene-json to generate a mixer config")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and separate into stems for ConductorSBN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID"
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" --model auto --scene-json
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" --model htdemucs_ft --scene-json
  %(prog)s "https://www.youtube.com/watch?v=VIDEO_ID" -o my_scene/ --model htdemucs_6s
  %(prog)s --input existing_track.mp3 --model auto --scene-json

Models:
  auto          Two-pass auto-detection. Runs 4-stem first, analyzes the
                "other" stem for guitar/piano content, re-runs with 6-stem
                if those instruments are prominent. Best quality decision
                but ~2x processing time when 6-stem is triggered.
  htdemucs      Default. 4 stems (vocals, drums, bass, other). Fast.
  htdemucs_ft   Fine-tuned. 4 stems. ~4x slower, slightly better quality.
  htdemucs_6s   6 stems (adds guitar + piano). Piano quality is mediocre.
  mdx_extra_q   Quantized MDX. Smaller download. Decent quality.
        """,
    )

    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="YouTube video URL to download and split.",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Path to an existing audio file to split (skips download).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for stems. Default: ./stems/<video_title>/",
    )
    parser.add_argument(
        "--model", "-m",
        default="htdemucs_ft",
        choices=["auto", "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q", "mdx_extra", "mdx_q"],
        help="Demucs model to use (default: htdemucs_ft). 'auto' runs 4-stem first, "
             "analyzes the 'other' stem, and re-runs with 6-stem if guitar/piano detected.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for Demucs inference (default: auto — uses GPU if available).",
    )
    parser.add_argument(
        "--scene-json",
        action="store_true",
        help="Generate a starter scene.json for the ConductorSBN adaptive mixer.",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="BPM to set in scene.json (default: 120). Adjust after inspecting the track.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download audio only, skip stem separation.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the downloaded WAV file alongside the stems.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Guitar/piano detection threshold for --model auto (0.0-1.0, default: 0.3). "
             "Lower = more sensitive (triggers 6-stem more often). "
             "Higher = more conservative (only triggers on very guitar/piano-heavy tracks).",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable automatic pruning of silent stems. By default, stems that "
             "are mostly silence (separation artifacts) are deleted.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=-40.0,
        help="RMS silence floor in dBFS for stem pruning (default: -40.0). "
             "Stems below this level with fewer than 5%% active frames are removed.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output from yt-dlp and Demucs.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.url is None and args.input is None:
        parser.error("Provide either a YouTube URL or --input <file>.")

    if args.url is not None and args.input is not None:
        parser.error("Provide either a URL or --input, not both.")

    check_dependencies()

    # Determine output directory
    if args.output:
        output_dir = args.output
    elif args.input:
        output_dir = os.path.join("stems", Path(args.input).stem)
    else:
        output_dir = os.path.join("stems", "_downloading")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get the audio file
    if args.input:
        audio_path = str(Path(args.input).resolve())
        if not os.path.exists(audio_path):
            print(f"Error: File not found: {audio_path}")
            sys.exit(1)
        print(f"Using existing file: {audio_path}")
    else:
        # Download from YouTube
        download_dir = tempfile.mkdtemp(prefix="yt_stems_dl_")
        try:
            audio_path = download_audio(args.url, download_dir, verbose=args.verbose)
        except Exception as e:
            print(f"\nDownload failed: {e}")
            shutil.rmtree(download_dir, ignore_errors=True)
            sys.exit(1)

        # Rename output dir to match video title if we used a temp name
        if args.output is None:
            title = Path(audio_path).stem
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
            if safe_title:
                new_output_dir = os.path.join("stems", safe_title)
                if os.path.exists(new_output_dir):
                    i = 2
                    while os.path.exists(f"{new_output_dir}_{i}"):
                        i += 1
                    new_output_dir = f"{new_output_dir}_{i}"
                output_dir = new_output_dir
                os.makedirs(output_dir, exist_ok=True)

    if args.download_only:
        dest = os.path.join(output_dir, Path(audio_path).name)
        if str(Path(audio_path).parent) != str(Path(output_dir)):
            shutil.copy2(audio_path, dest)
        print(f"\nDownloaded audio saved to: {dest}")
        if args.url and 'download_dir' in locals():
            shutil.rmtree(download_dir, ignore_errors=True)
        return

    # Step 2: Separate stems
    model_used = args.model
    try:
        if args.model == "auto":
            stem_paths, model_used = separate_with_auto_model(
                audio_path=audio_path,
                output_dir=output_dir,
                device=args.device,
                guitar_piano_threshold=args.threshold,
                verbose=args.verbose,
            )
        else:
            stem_paths = separate_stems(
                audio_path=audio_path,
                output_dir=output_dir,
                model=args.model,
                device=args.device,
                verbose=args.verbose,
            )
    except Exception as e:
        print(f"\nStem separation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Step 3: Normalize stems
    normalize_stems(stem_paths)

    # Step 4: Prune silent stems (artifact-only stems from separation)
    if not args.no_prune:
        stem_paths, _ = prune_silent_stems(
            stem_paths,
            silence_threshold_db=args.silence_threshold,
            verbose=args.verbose,
        )

    # Step 5: Keep or discard original download
    if args.keep_original and args.url:
        original_dest = os.path.join(output_dir, "original_" + Path(audio_path).name)
        shutil.copy2(audio_path, original_dest)
        print(f"\nOriginal audio kept at: {original_dest}")

    # Clean up temp download dir
    if args.url and 'download_dir' in locals():
        shutil.rmtree(download_dir, ignore_errors=True)

    # Step 6: Generate scene.json if requested
    if args.scene_json:
        scene_name = Path(output_dir).name.replace("_", " ").title()
        generate_scene_json(
            stem_paths=stem_paths,
            output_dir=output_dir,
            scene_name=scene_name,
            bpm=args.bpm,
        )

    # Print summary
    print_summary(stem_paths, output_dir, args.scene_json, model_used=model_used)


if __name__ == "__main__":
    main()
