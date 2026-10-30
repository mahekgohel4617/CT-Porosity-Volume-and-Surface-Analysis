# CT Pore Analyzer

Automatic ring detection and 3-D pore analysis for stacks of CT slices, in a single C++17 / OpenCV file.

For specimens that appear as a bright circular ring in each slice, with pores as dark gaps in the ring, the program:

1. finds the ring (the region of interest) in every slice automatically, and skips slices that have no ring,
2. thresholds dark pixels inside a thin band around the fitted circle,
3. links pores that overlap in consecutive slices into 3-D pores (union-find),
4. reports each pore's volume, size, surface area, sphericity, shape and position on the ring, and writes a CSV.

## Build

Requires g++ 9+ (C++17) and OpenCV 4.

```bash
# Ubuntu / WSL2
sudo apt install g++ pkg-config libopencv-dev
# macOS
brew install opencv pkg-config

g++ -O2 -std=c++17 pore_analyzer.cpp -o pore_analyzer $(pkg-config --cflags --libs opencv4)
```

## Usage

Each argument is one stack: a folder of slices (sorted by file name) or a multi-page TIFF.

```bash
./pore_analyzer stackA stackB --dx 0.5 --dy 0.5 --dz 1.0 --unit um --csv pores.csv --debug overlays
./pore_analyzer scan.tif --verbose
```

Set `--dx --dy --dz` to your voxel size, otherwise results are in pixels.

| Option | Default | Meaning |
|---|---|---|
| `--pore-frac` | 0.5 | pore if gray < frac x ring brightness |
| `--band-shrink` | 0.6 px | ring edge ignored (suppresses false pores on thin rings) |
| `--open` | 0 | 3x3 opening iterations on the 2-D pore mask |
| `--min-voxels` | 4 | smallest pore kept |
| `--min-slices` | 2 | fewest slices a pore must span |
| `--min-sphericity` | 0.70 | "spherical" flag threshold |
| `--min-aspect` | 0.50 | "spherical" flag also needs c/a above this |
| `--ring-tol` | 3 px | ring inlier tolerance |
| `--min-coverage` | 0.70 | fraction of the circumference that must be visible |
| `--csv`, `--debug`, `--verbose` | | CSV file, overlay images, per-slice log |

## Output

Console table per stack (ID, z-range, voxels, volume, equivalent diameter, sphericity, c/a, fill ratio, centroid, angle, shape flag), plus an optional CSV with one row per pore. The CSV can be used as a feature table for machine learning.

Green = fitted ring, red = detected pore with its ID.

## How it works

- **Ring (ROI):** Otsu threshold, Kasa least-squares circle fit, 6 robust refits (tolerance max(3 px, 3 x 1.4826 x MAD)), then validity checks (pixel count, inlier fraction, 72-sector coverage, radius, thickness).
- **Pores:** dark pixels inside the ring band, relative to the median ring brightness of the same slice.
- **3-D linking:** 8-connected 2-D blobs; blobs sharing an (x, y) pixel in consecutive slices are merged. A slice with no ring breaks the chain.
- **Measurements:** volume = N x dx x dy x dz; equivalent diameter; surface area from exposed voxel faces x 2/3; Wadell sphericity; principal axes from the covariance matrix; aspect ratio c/a; fill ratio; angle on the ring.

## Results

Tested on a synthetic 60-slice stack (1000 x 1000 px, 5 pores of known size, 12 slices with no ring) made by `make_synthetic_stack.py`:

- Ring 10 px wide: 5 of 5 pores found with voxel counts equal to the ground truth; the elongated pore (c/a = 0.33) is flagged irregular; the 12 ring-less slices are skipped.
- Edge shrink: on one real slice, false one-pixel pores dropped from 24 to 0.
- Speed: about 0.85 s for the 60-slice stack on one core (warm file cache), about 120 MB peak memory.

```bash
python3 make_synthetic_stack.py synth_stack 10     # ring width in px
./pore_analyzer synth_stack
```

## Limitations

- **Pores larger than the ring are clipped.** Pores are measured only inside the ring band. With a 3.5 px ring, the same synthetic pores were found but their volumes were 36-81% too small.
- **Not validated on annotated real scans.** The only real slice used has no pores, and the synthetic test shares the generator's assumptions.
- Surface area and sphericity are estimates, calibrated for round objects (a perfect cube scores 1.0). Pores of only a few voxels are unreliable.
- Assumes a bright circular ring, darker pores, and pores that continue across neighbouring slices.

## Files

| File | Purpose |
|---|---|
| `code.cpp` | the whole program |
