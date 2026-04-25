"""
generate.py – Random sampling from a trained RAVE .ts model
─────────────────────────────────────────────────────────────
Usage:
    python generate.py --model path/to/model.ts
    python generate.py --model path/to/model.ts --duration 10 --seed 42
    python generate.py --model path/to/model.ts --duration 30 --temperature 1.5 --output out.wav

Arguments:
    --model         Path to the exported RAVE model (.ts file)   [required]
    --duration      Length of generated audio in seconds         [default: 10]
    --temperature   Scaling factor for latent noise.
                    < 1.0 → more conservative/tonal output
                    = 1.0 → standard sampling
                    > 1.0 → wilder/more chaotic output           [default: 1.0]
    --seed          Random seed for reproducibility              [default: 42]
    --output        Output .wav file path                        [default: generated_audio/generated_<timestamp>.wav]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate audio from a RAVE .ts model via random latent sampling."
    )
    parser.add_argument("--model",       type=str,   required=True,
                        help="Path to the exported RAVE model (.ts file)")
    parser.add_argument("--duration",    type=float, default=30.0,
                        help="Duration of generated audio in seconds (default: 10)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Latent noise scale: <1 conservative, >1 chaotic (default: 1.0)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--output",      type=Path,   default=Path("generated_audio"),
                        help="Output .wav path (default: generated_audio/generated_<timestamp>.wav)")
    return parser.parse_args()


# ── Model helpers ──────────────────────────────────────────────────────────────
def load_model(model_path: str) -> torch.nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix.lower() != ".ts":
        raise ValueError(f"Expected a .ts file, got: {path.suffix}")

    print(f"Loading model: {path}")
    model = torch.jit.load(str(path)).eval()
    return model


def get_model_info(model: torch.nn.Module) -> tuple[int, int]:
    """
    Read sample rate and latent dimension from the model's registered buffers.
    RAVE exports store these as `sampling_rate` and `latent_size` (or `z_channels`).
    Falls back to common defaults if the buffers are not found.
    """
    # sample rate
    try:
        sr = int(model.sampling_rate)
    except AttributeError:
        sr = 44100
        print(f"  [WARN] Could not read sampling_rate from model, assuming {sr} Hz")

    # latent dimension
    latent_dim = None
    for attr in ("latent_size", "z_channels", "n_latents"):
        try:
            latent_dim = int(getattr(model, attr))
            break
        except AttributeError:
            continue

    if latent_dim is None:
        # probe the decoder with a tiny tensor to infer latent dim
        for dim in [16, 8, 32, 4, 64]:
            try:
                test = torch.zeros(1, dim, 1)
                model.decode(test)
                latent_dim = dim
                print(f"  [INFO] Inferred latent dimension: {latent_dim} (probed)")
                break
            except Exception:
                continue

    if latent_dim is None:
        latent_dim = 16
        print(f"  [WARN] Could not determine latent dimension, defaulting to {latent_dim}")

    return sr, latent_dim


# ── Generation ─────────────────────────────────────────────────────────────────
def generate(
    model:       torch.nn.Module,
    sr:          int,
    latent_dim:  int,
    duration_s:  float,
    temperature: float,
    seed:        int | None,
) -> np.ndarray:
    """
    Sample random Gaussian noise in latent space and decode it to audio.

    RAVE's decoder expects: (batch=1, latent_dim, time_steps)
    where time_steps ≈ duration_s * sr / compression_ratio.

    We estimate the compression ratio from a short probe run, then generate
    the full sequence in one forward pass.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # ── Estimate compression ratio ─────────────────────────────────────────
    # Decode a single latent frame and measure how many audio samples come out.
    probe_steps = 16
    probe_z = torch.zeros(1, latent_dim, probe_steps)
    with torch.inference_mode():
        probe_out = model.decode(probe_z)
    compression_ratio = probe_out.shape[-1] / probe_steps
    print(f"  Compression ratio  : {compression_ratio:.0f}x  ({probe_out.shape[-1]} samples per {probe_steps} latent frames)")

    # ── Build random latent sequence ───────────────────────────────────────
    target_samples = int(duration_s * sr)
    latent_steps   = max(1, round(target_samples / compression_ratio))

    z = torch.randn(1, latent_dim, latent_steps) * temperature
    print(f"  Latent shape       : {list(z.shape)}  (temperature={temperature})")

    # ── Decode ─────────────────────────────────────────────────────────────
    t0 = time.time()
    with torch.inference_mode():
        audio = model.decode(z)          # (1, channels, samples)  or  (1, samples)
    elapsed = time.time() - t0

    # Flatten to (samples,) mono
    audio_np = audio.squeeze().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)   # stereo → mono average

    actual_dur = len(audio_np) / sr
    print(f"  Generated          : {actual_dur:.2f}s  (decoded in {elapsed:.2f}s, {actual_dur/elapsed:.1f}x realtime)")

    return audio_np


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Output path ────────────────────────────────────────────────────────
    output_path = Path(args.output)

    # If it's a directory OR has no .wav extension → generate filename
    if output_path.suffix != ".wav":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / f"generated_{timestamp}.wav"

    # ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    model = load_model(args.model)
    sr, latent_dim = get_model_info(model)

    print(f"\n  Sample rate        : {sr} Hz")
    print(f"  Latent dimension   : {latent_dim}")
    print(f"  Target duration    : {args.duration}s")
    print(f"  Seed               : {args.seed if args.seed is not None else 'random'}")
    print()

    # ── Generate ───────────────────────────────────────────────────────────
    audio = generate(
        model=model,
        sr=sr,
        latent_dim=latent_dim,
        duration_s=args.duration,
        temperature=args.temperature,
        seed=args.seed,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()