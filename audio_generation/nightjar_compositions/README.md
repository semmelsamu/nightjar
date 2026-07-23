# Nightjar Compositions

Rendered Nightjar composition takes built from original source audio and local RAVE-generated Nightjar clips.

This published set contains:

- `README.md`
- `outputs/compositions/*.wav`
- `outputs/figures/*.png`

The local working inputs are not part of the published set:

- `audio/`
- `data/`
- `training_data/`
- `generated_route_cache/`
- `nightjar_composition.ipynb`

## Composition Form

Every take uses the same four-section arrangement.

| Section | Fade-in start | Source |
| --- | ---: | --- |
| birds intro | `0:00` | original bird training WAVs |
| original Nightjar | `0:34` | original Nightjar source recording |
| AI Nightjar route | `3:00` | short RAVE-generated Nightjar clips selected by route |
| birds outro | `4:10` | the intro bird phrase in reverse order |

The full render is about `4:46` (`285.71` seconds) at `44.1 kHz`.

The listed times are fade-in starts. Section joins are crossfaded, so the sections overlap slightly instead of cutting on the exact timestamp.

## How The Takes Are Built

1. A bird intro phrase is assembled from original bird training WAVs.
2. The original Nightjar recording fades in after the bird intro.
3. The AI section is assembled from a route: a sequence of short RAVE-generated Nightjar clips.
4. The bird outro uses the same bird phrase as the intro, but in reverse order.
5. The full track is crossfaded and normalized into one rendered WAV.

The bird and original-song sections stay as source audio in every take. The AI section changes by route.

## Route Space

The routes are designed in a latent map made from RAVE features.

RAVE encodes a short audio clip as an 8-channel latent sequence over time:

`8 latent channels x time steps`

For the route map, each original Nightjar anchor is encoded and averaged over time. That gives one `8D latent mean` per anchor.

The visible figures use `PC1`, `PC2`, and `PC3`. These are not the first three RAVE latent channels. They are fitted map coordinates: each PC coordinate is a weighted mixture of the eight latent-channel means.

The fitted relation is:

`pc_coord ~= intercept + latent_mean @ basis`

- `latent_mean`: one 8-number summary for a clip
- `basis`: the fitted weights from the 8D latent mean into PC1/PC2/PC3
- `intercept`: the fitted offset of the map

Each AI route point has a matching local RAVE-generated audio clip. Some clips were generated earlier into the local Nightjar pool, and some were generated for the shaped route cache. In both cases the rendered composition uses the clip attached to the plotted route point.

## Route Modes

All route modes follow the same audio rule: route points are mapped to local RAVE-generated Nightjar clips. The difference between modes is the path through the latent map.

### `close`

This route follows the main Nightjar anchor order and prefers low-variation nodes near the original anchor path. It is the nearest take to the source map.

Output:

- `outputs/compositions/nightjar_composition_close.wav`
- `outputs/figures/routes/nightjar_route_close.png`

### `free`

This route still uses the Nightjar anchor map, but the anchor movement is looser. It can move forward, backward, stay near the same anchor, or skip to another anchor.

Output:

- `outputs/compositions/nightjar_composition_free.wav`
- `outputs/figures/routes/nightjar_route_free.png`

### `wild_high`, `wild_side`, `wild_low`

These three takes use the same route idea in different directions. Each one draws a flattened loop in the visible PC2/PC3 map:

- leave the dense Nightjar cloud
- bend around a rounded loop
- return to the starting region

The three directions create three different route regions:

- `wild_high`: points toward a higher PC3 region
- `wild_side`: points laterally across the map
- `wild_low`: points toward a lower PC2/PC3 region

Outputs:

- `outputs/compositions/nightjar_composition_wild_high.wav`
- `outputs/compositions/nightjar_composition_wild_side.wav`
- `outputs/compositions/nightjar_composition_wild_low.wav`
- `outputs/figures/routes/nightjar_route_wild_high.png`
- `outputs/figures/routes/nightjar_route_wild_side.png`
- `outputs/figures/routes/nightjar_route_wild_low.png`

### `wild_8d`

This take builds several generated loops from 8D latent-mean route points. The figures show the PC1/PC2/PC3 projection of those route points.

The difference from `wild_high`, `wild_side`, and `wild_low` is the route construction:

- the other wild takes are one flattened visible loop each
- `wild_8d` is several smaller generated loops built from 8D latent-mean points

Output:

- `outputs/compositions/nightjar_composition_wild_8d.wav`
- `outputs/figures/routes/nightjar_route_wild_8d.png`

### `outer`

This route is not a one-direction trip. It starts near the dense Nightjar cloud, moves outward, loops around the cloud, and returns.

The route ignores the large original-song outliers when choosing the plotting and loop area. The loop is built around the dense region where most Nightjar route points sit.

Output:

- `outputs/compositions/nightjar_composition_outer.wav`
- `outputs/figures/routes/nightjar_route_outer.png`

## Figures

Each route figure shows three projections:

- `PC1 / PC2`
- `PC1 / PC3`
- `PC2 / PC3`

The small arrows show route direction. The star marks the first route point.

The combined timing figure is:

- `outputs/figures/nightjar_composition_routes.png`

## Output Files

Rendered compositions:

- `outputs/compositions/nightjar_composition_close.wav`
- `outputs/compositions/nightjar_composition_free.wav`
- `outputs/compositions/nightjar_composition_wild_high.wav`
- `outputs/compositions/nightjar_composition_wild_side.wav`
- `outputs/compositions/nightjar_composition_wild_low.wav`
- `outputs/compositions/nightjar_composition_wild_8d.wav`
- `outputs/compositions/nightjar_composition_outer.wav`

Route figures:

- `outputs/figures/routes/nightjar_route_close.png`
- `outputs/figures/routes/nightjar_route_free.png`
- `outputs/figures/routes/nightjar_route_wild_high.png`
- `outputs/figures/routes/nightjar_route_wild_side.png`
- `outputs/figures/routes/nightjar_route_wild_low.png`
- `outputs/figures/routes/nightjar_route_wild_8d.png`
- `outputs/figures/routes/nightjar_route_outer.png`
- `outputs/figures/nightjar_composition_routes.png`

## Local Route Audio

Each plotted AI route point has a matching local RAVE-generated audio clip.

The local route audio lives in ignored working folders:

- `audio/`: earlier-generated Nightjar pool clips used by routes such as `close` and `free`
- `generated_route_cache/`: per-step clips generated for shaped routes such as `wild_low`, `wild_8d`, and `outer`

For route-cache clips, the files are stored as:

- `step_000.wav`
- `step_001.wav`
- `step_002.wav`
- ...

The route manifest links those step files back to the plotted route points.

The local route audio folders are ignored by Git. The rendered WAV files and route figures live under `outputs/`.
