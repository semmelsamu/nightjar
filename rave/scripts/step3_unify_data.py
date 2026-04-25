"""
Step 3 – Unify Sample Rate & Verify Dataset
────────────────────────────────────────────
Reads:   data/preprocessing_steps12/**/*.wav   (output of steps 1 & 2)
Writes:  data/training_data/**/*.wav   (same relative paths)

- Scans every WAV and tallies sample rates.
- The most common sample rate becomes the target.
- Every file is resampled to that rate (files already at target are simply copied).
- A final report confirms: sample rate, channel count, and total duration.

Run AFTER step1_prepare_birds.py and step2_prepare_songs.py.
"""

import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torchaudio
import torchaudio.transforms as T
import torch

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("data/unprepared_data/preprocessing_steps12")
OUTPUT_DIR = Path("data/training_data")

# ── Scan ───────────────────────────────────────────────────────────────────────
def scan(root: Path) -> tuple[dict[int, list[Path]], list[tuple[Path, str]]]:
    """Return {sample_rate: [paths]} and [(bad_path, error_msg)]."""
    rate_map: dict[int, list[Path]] = defaultdict(list)
    errors: list[tuple[Path, str]] = []

    files = list(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No .wav files found under {root}")

    for f in files:
        try:
            info = torchaudio.info(f)
            rate_map[info.sample_rate].append(f)
        except Exception as exc:
            errors.append((f, str(exc)))

    return rate_map, errors

# -- Normalize audio ---------------
def normalize_rms(wav: torch.Tensor, target_rms: float = 0.1, eps: float = 1e-8):
    rms = torch.sqrt(torch.mean(wav**2))
    if rms > eps:
        wav = wav * (target_rms / rms)
    return wav

# ── Resample & copy ────────────────────────────────────────────────────────────
def process(src: Path, dst: Path, src_sr: int, tgt_sr: int) -> None:
    wav, sr = torchaudio.load(src)

    # Sanity-check: must be mono after steps 1/2
    if wav.shape[0] != 1:
        raise ValueError(f"Expected mono, got {wav.shape[0]} channels")

    wav = wav - wav.mean()

    if sr != tgt_sr:
        wav = T.Resample(orig_freq=sr, new_freq=tgt_sr)(wav)

    wav = normalize_rms(wav, 0.1) # normalize audio
    wav = wav.clamp(-1.0, 1.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), wav, tgt_sr)

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning prepared_data …\n")
    rate_map, scan_errors = scan(INPUT_DIR)

    # ── Sample-rate report ──────────────────────────────────────────────────
    print("=== SAMPLE RATE DISTRIBUTION ===")
    for sr, files in sorted(rate_map.items(), key=lambda x: -len(x[1])):
        print(f"  {sr:>7} Hz  →  {len(files):>5} files")

    target_sr: int = Counter({sr: len(f) for sr, f in rate_map.items()}).most_common(1)[0][0]
    total_files = sum(len(v) for v in rate_map.values())
    need_resample = total_files - len(rate_map.get(target_sr, []))

    print(f"\n  Target sample rate : {target_sr} Hz  (most common)")
    print(f"  Files to resample  : {need_resample} / {total_files}")

    if scan_errors:
        print(f"\n  ⚠ {len(scan_errors)} file(s) could not be scanned:")
        for f, e in scan_errors:
            print(f"      {f}  →  {e}")

    # ── Process ─────────────────────────────────────────────────────────────
    print("\nProcessing …\n")
    saved = skipped = failed = 0

    for sr, files in rate_map.items():
        for src in files:
            rel  = src.relative_to(INPUT_DIR)
            dst  = OUTPUT_DIR / rel
            try:
                process(src, dst, sr, target_sr)
                action = "resampled" if sr != target_sr else "copied"
                print(f"  [{action:>9}]  {rel}")
                saved += 1
            except Exception as exc:
                print(f"  [ERR]       {rel}  →  {exc}")
                failed += 1

    # ── Final verification ───────────────────────────────────────────────────
    print("\n=== VERIFICATION ===")
    out_files = list(OUTPUT_DIR.rglob("*.wav"))
    rate_counts: Counter = Counter()
    chan_counts: Counter = Counter()
    total_duration_s = 0.0
    duration_by_split: defaultdict = defaultdict(float)  # e.g. "birds", "songs"
    verify_errors = []

    for f in out_files:
        try:
            info = torchaudio.info(f)
            rate_counts[info.sample_rate] += 1
            chan_counts[info.num_channels] += 1
            dur = info.num_frames / info.sample_rate
            total_duration_s += dur
            # first folder level under OUTPUT_DIR is the split name (birds / songs)
            split = f.relative_to(OUTPUT_DIR).parts[0]
            duration_by_split[split] += dur
        except Exception as exc:
            verify_errors.append((f, str(exc)))

    print(f"  Total files        : {len(out_files)}")
    print(f"  Sample rates found : {dict(rate_counts)}")
    print(f"  Channel counts     : {dict(chan_counts)}")
    print(f"  Total duration     : {total_duration_s / 60:.1f} min")
    for split, dur_s in sorted(duration_by_split.items()):
        print(f"    └ {split:<10} : {dur_s / 60:.1f} min")

    if len(rate_counts) == 1 and target_sr in rate_counts:
        print(f"\n  ✔ All files are {target_sr} Hz mono WAV.")
    else:
        print("\n  ✖ Inconsistencies detected – check the lists above.")
        sys.exit(1)

    if verify_errors:
        print(f"\n  ⚠ {len(verify_errors)} file(s) could not be verified.")

    print(f"\nDone. Output → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()