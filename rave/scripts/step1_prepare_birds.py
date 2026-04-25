"""
Step 1 – Prepare Bird Dataset
─────────────────────────────
Reads:   data/unprepared_data/kaggle_birds_data/<species>/*.mp3
Writes:  data/unprepared_data/preprocessing_steps12/birds/<species>_<i>.wav

For each species:
  - picks up to MAX_FILES_PER_CLASS files at random (seeded for reproducibility)
  - trims to MAX_DURATION_MS if longer; leaves short files at their original length
  - converts to mono
  - saves as 16-bit PCM WAV  (sample rate kept native; unified in step 3)
"""

import random
from pathlib import Path
from pydub import AudioSegment

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR           = Path("data/unprepared_data/kaggle_birds_data")
OUTPUT_DIR          = Path("data/unprepared_data/preprocessing_steps12/birds")
MAX_FILES_PER_CLASS = 3
MAX_DURATION_MS     = 20_000   # 20 s
SEED                = 42

# ── Helpers ────────────────────────────────────────────────────────────────────
def to_mono_wav(audio: AudioSegment, duration_ms: int) -> AudioSegment:
    """Trim to max length if needed, then collapse to mono. Short files kept as-is."""
    if len(audio) > duration_ms:
        audio = audio[:duration_ms]
    return audio.set_channels(1)

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    species_dirs = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not species_dirs:
        raise FileNotFoundError(f"No species subdirectories found in {INPUT_DIR}")

    total_saved = 0
    total_skipped = 0

    for species_dir in sorted(species_dirs):
        files = list(species_dir.glob("*.mp3"))
        if not files:
            print(f"  [SKIP] {species_dir.name} – no .mp3 files found")
            continue

        random.shuffle(files)
        selected = files[:MAX_FILES_PER_CLASS]

        for i, src in enumerate(selected):
            out_path = OUTPUT_DIR / f"{species_dir.name}_{i}.wav"
            try:
                audio = AudioSegment.from_file(src)
                audio = to_mono_wav(audio, MAX_DURATION_MS)
                audio.export(out_path, format="wav")
                print(f"  [OK]   {out_path.name}")
                total_saved += 1
            except Exception as exc:
                print(f"  [ERR]  {src.name} → {exc}")
                total_skipped += 1

    print(f"\nDone. Saved {total_saved} files, skipped {total_skipped}.")


if __name__ == "__main__":
    main()
