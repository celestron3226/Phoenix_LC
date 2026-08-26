# phoenix/training

The pipeline that produced the paper's Phoenix land-cover map, in run order:
training tiles, model training (the paper's benchmark), prediction tiles,
per-quad prediction, and the final map products.

There are two tile scripts with different roles; do not mix them up:

| script | role | labels? |
|--------|------|---------|
| `prepare_tiles_training.py` | cut the hand-labeled bundles into train/val `.npz` tiles for model training | yes |
| `prepare_tiles_prediction.py` | cut the FULL metro NAIP + NDVI + CHM into 256 px `.npz` tiles for city-wide prediction | no |

`phoenix_common.py` holds the shared normalization code, so training and
prediction preprocessing stay byte identical.

## 1. Training tiles (CPU, `geopandas` env)

```bash
sbatch prepare_tiles_training.sh     # runs prepare_tiles_training.py
```

Converts the hand-modified label groups (from `phoenix/label/`) into 128 and
256 px training tiles in one run, with a per-source-tile train/val split and
the Tree-over-Building overlap rule.

## 2. Train: the benchmark (GPU, `pytorch_gpu` env)

`encoder_experiment_week13_sol.py` runs the paper's grid on SLURM: encoder
(ResNet18 / ResNet50 / Swin-v2-Base) x pre-trained weights (random / ImageNet /
RSC / Chesapeake LC / SatlasPretrain) x patch size (128 / 256), three seeds,
one SLURM array task per run.

```bash
python encoder_experiment_week13_sol.py --check          # inputs present?
python encoder_experiment_week13_sol.py --predownload    # ImageNet weights (login node)
python encoder_experiment_week13_sol.py --list-jobs      # index -> run mapping
mkdir -p slurm_out
sbatch --array=0-77%8 encoder_experiment_week13_sol.sh
python encoder_experiment_week13_sol.py --aggregate      # rebuild results.csv
```

The best run (SatlasPretrain aerial + Swin-v2-Base, patch 256, coord none)
provides the `model_best.pth` used below.

## 3. Prediction tiles (CPU array, `geopandas` env)

```bash
sbatch prepare_tiles_prediction.sh   # runs prepare_tiles_prediction.py, one quad per task
```

Cuts every NAIP + NDVI + CHM quad into non-overlapping 256 px prediction tiles,
copying the normalization constants from the training tile config so
prediction preprocessing matches training exactly.

## 4. Predict (GPU array, `pytorch_gpu` env)

```bash
sbatch predict_lc.sh                 # runs predict_phoenix_lc.py, one quad per task
```

Loads `model_best.pth` into the Swin-v2-B U-Net and writes one
`landcover_<stem>.tif` per quad (uint8 codes 1..7, 0 = nodata).

## 5. Final maps (CPU, `geopandas` env)

```bash
sbatch make_maps.sh
```

Mosaics all per-quad rasters, clips the mosaic to the City of Phoenix boundary
(after repairing the boundary geometry with ogr2ogr -makevalid), and
polygonizes both products to GeoPackage:
`phoenix_lc_full.tif` / `phoenix_lc_city.tif` and
`phoenix_lc_full.gpkg` / `phoenix_lc.gpkg`.

## Notes

- Every stage is resume-safe: finished quads/runs are detected and skipped, so
  any script can simply be resubmitted after a timeout.
- Paths appear as `/path/to/...` placeholders; edit the `CONFIG` blocks (or the
  CLI flags of the benchmark script) before running.
- The `.sh` scripts use `YOUR_SLURM_ACCOUNT` and `YOUR_EMAIL@example.com`;
  replace before submitting.
