# Nightjar RAVE Midpoints

Interactive near-anchor routes through the trained Nightjar RAVE latent space.

## Contents

- `index.html` is the standalone route player with 2D and 3D views.
- `audio/` contains the rendered route variants used by the player.
- `model/epoch_2000000.ckpt.ts` is the exported RAVE model, stored with Git LFS.
- `generate.py` rebuilds the route audio and player.
- `01_generate.ipynb` runs the generator and embeds the player for inspection.

The raw source song and training data are not committed.

## Playback

Open `index.html` in a browser. The dark points and black path are encoded Nightjar windows, the colored paths are generated near-anchor routes, and the white point follows audio playback.

## Regeneration

Install the audio requirements from `../requirements.txt`, then provide the source Nightjar song locally:

```bash
NIGHTJAR_SOURCE_WAV=/path/to/nightjar.wav python generate.py
```

Alternatively, place the source song at `data/nightjar.wav` inside this folder. Running the generator refreshes `audio/*.wav` and rewrites `index.html` with matching route data.
