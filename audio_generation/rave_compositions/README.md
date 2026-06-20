# Nightjar RAVE Compositions

Composition notebook and rendered takes for the Nightjar RAVE experiment.

- `nightjar_composition.ipynb` assembles complete takes from the cached local clip pool and the original Nightjar recording.
- Local `audio/` and `data/` cache inputs are not included in this repository snapshot.
- `outputs/` contains the rendered composition takes and route figure.
- The notebook does not run RAVE by default; it works from precomputed clips.

## Arrangement

The rendered takes use the same four-section form:

| Section | Starts at | Role |
| --- | ---: | --- |
| birds intro | 0:00 | opening bird phrase |
| original Nightjar | 0:34 | full source recording fades in |
| AI-generated Nightjar | 3:00 | precomputed Nightjar pool route fades in |
| birds outro | 4:10 | the opening bird phrase returns in reverse |

The total rendered length is about 4:46. Section boundaries are crossfaded, so each listed time is the start of the fade into that section.

## Route Selection

Each AI clip is a precomputed 3-second Nightjar pool node with an anchor index, variation index, PCA position, RMS, and spectral centroid. The notebook does not synthesize new clips during composition; it chooses a route through the existing pool.

- `close` mostly follows the original anchor order and strongly prefers low-variation nodes, so it stays closest to the source song.
- `free` can move forward, backward, stay near the same anchor, or skip a few anchors. It still uses anchor indices as a map, but it is not locked to the original order.
- `wild` favors wider variants and larger anchor jumps, while still keeping enough local continuity to avoid hard cuts.
- `outer` explicitly prefers nodes with a high distance from their source anchor and from the main anchor cloud. It leaves the main Nightjar zone, makes a wider loop, then bends back toward the final Nightjar anchor before the bird outro.

Recent nodes and anchors are penalized in every AI mode, which keeps the route from collapsing into a tiny repeated loop.
