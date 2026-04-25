"""
Step 2 – Prepare Song Dataset
──────────────────────────────
Reads:   data/unprepared_data/song_data/*.mp3   (and .wav)
Writes:  data/unprepared_data/preprocessing_steps12/songs/<basename>_off<X>s_part<i>.wav

For each song and each configured start offset:
  - slices the audio into consecutive 20-second windows beginning at `offset`
  - tail snippets shorter than MIN_SNIPPET_MS are discarded; those between
    MIN_SNIPPET_MS and 20 s are kept at their natural length (no zero-padding)
  - converts each window to mono
  - saves as WAV  (sample rate kept native; unified in step 3)

Three passes are made per song:
  offset = 0 s  → straight non-overlapping 20-s chunks
  offset = 5 s  → same chunks but starting 5 s later
  offset = 10 s → same chunks but starting 10 s later
"""

from pathlib import Path
from pydub import AudioSegment

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR       = Path("data/unprepared_data/song_data")
OUTPUT_DIR      = Path("data/unprepared_data/preprocessing_steps12/songs")
SNIPPET_MS      = 20_000          # 20 s per window
OFFSETS_MS      = [0, 5_000, 10_000]   # start offsets → 0 s, 5 s, 10 s lag
MIN_SNIPPET_MS  = 5_000           # discard tail snippets shorter than this
SUPPORTED_EXT   = {".mp3", ".wav"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def slice_audio(audio: AudioSegment,
                offset_ms: int,
                snippet_ms: int,
                min_ms: int) -> list[AudioSegment]:
    """Return consecutive non-overlapping windows starting at offset_ms.
    Tail chunks shorter than min_ms are discarded; those between min_ms and
    snippet_ms are kept at their natural length (no zero-padding)."""
    segments = []
    start = offset_ms
    while start < len(audio):
        chunk = audio[start : start + snippet_ms]
        if len(chunk) < min_ms:
            break          # discard very short tail
        segments.append(chunk.set_channels(1))
        start += snippet_ms
    return segments

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    src_files = [f for f in INPUT_DIR.iterdir()
                 if f.suffix.lower() in SUPPORTED_EXT]

    if not src_files:
        raise FileNotFoundError(f"No audio files found in {INPUT_DIR}")

    total_saved = 0

    for src in sorted(src_files):
        print(f"\nProcessing: {src.name}")
        try:
            audio = AudioSegment.from_file(src)
        except Exception as exc:
            print(f"  [ERR] Could not load {src.name}: {exc}")
            continue

        base = src.stem

        for offset_ms in OFFSETS_MS:
            snippets = slice_audio(audio, offset_ms, SNIPPET_MS, MIN_SNIPPET_MS)

            for i, snippet in enumerate(snippets):
                offset_s = offset_ms // 1000
                out_name = f"{base}_off{offset_s}s_part{i}.wav"
                out_path = OUTPUT_DIR / out_name
                try:
                    snippet.export(out_path, format="wav")
                    print(f"  [OK]  {out_name}")
                    total_saved += 1
                except Exception as exc:
                    print(f"  [ERR] {out_name}: {exc}")

    print(f"\nDone. Saved {total_saved} snippets total.")


if __name__ == "__main__":
    main()
