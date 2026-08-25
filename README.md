# Phoenix_LC

Deep learning pipeline for land-cover classification in the City of Phoenix.
The model classifies seven categories at 0.3 m resolution from NAIP imagery, a
LiDAR-derived canopy height model (CHM), and NDVI, using a U-Net segmentation architecture with several pre-trained backbones.

<img width="700" alt="Phoenix land cover map" src="https://github.com/user-attachments/assets/af13893b-5128-4382-b5db-cb60cd4feac6" />

## Environment setup

Three conda environments, one per kind of work:

```bash
conda env create -f environment_pytorch_gpu.yml   # Model training and prediction
conda env create -f environment_geopandas.yml     # Data prep and labeling (CPU): includes GDAL and PDAL
conda env create -f environment_segment.yml       # GPU vegetation seeds (DeepForest + SAM3), Pass 2 of labeling only
```

The `segment` environment needs a prior `hf auth login` so SAM3 can download
its checkpoint from Hugging Face.

## Repository layout

```
Phoenix_LC/
├── environment_pytorch_gpu.yml     Model training and prediction env
├── environment_geopandas.yml       Data preparing / labeling env (GDAL, PDAL)
├── environment_segment.yml         GPU vegetation-seed env (DeepForest, SAM3)
├── chesapeake/                     Encoder pretraining on ChesapeakeCVPR
│   └── train_chesapeake_swin.py
├── phoenix/
│   ├── data/                       Build the input rasters (NAIP, NDVI, CHM)
│   │   ├── naip/  jp2_combine
│   │   ├── ndvi/  ndvi_per_tile
│   │   └── chm/   download_laz, LAZ_to_CHM
│   ├── label/                      Label generation for ArcGIS editing
│   │   ├── 01_make_patches
│   │   ├── 02_make_veg_seeds
│   │   └── 03_build_bundles
│   ├── training/                   Train and predict pipeline
│       ├── phoenix_common.py
│       ├── prepare_tiles_phoenix.py
│       ├── train_phoenix.py
│       └── predict_phoenix.py
└── LICENSE
```

Each `.py` under `phoenix/data/` and `phoenix/label/` ships with a matching
`.sh` SLURM submission script. Each folder has its own README with the exact
commands.

## Pipeline overview

The full workflow runs in five stages.

1. Pretrain the encoder (`chesapeake/`). Train a U-Net on ChesapeakeCVPR and
   keep the best `best_encoder_*.pth`. This checkpoint is the transfer source
   for the Phoenix model.
2. Build the input rasters (`phoenix/data/`). Convert NAIP JP2 to tiled
   GeoTIFF, compute per-tile NDVI, and produce a per-NAIP-tile CHM from USGS
   LiDAR. Output is a 1:1:1 set of NAIP / NDVI / CHM tiles on the same grid.
3. Generate labels (`phoenix/label/`). Sample training patches, auto-seed
   vegetation and built-surface polygons, and bundle them for hand-editing in
   ArcGIS. The hand-corrected shapefiles are the training labels.
4. Train and predict (`phoenix/training/`). Convert the hand-modified labels
   into tiles, fine-tune the U-Net from the Chesapeake encoder, and run
   city-wide prediction clipped to the Phoenix boundary.

## Data

The raw inputs are not distributed with this repository (they are large and
publicly available at the source):

- NAIP 0.3 m 4-band (RGBN) imagery for the City of Phoenix (https://earthexplorer.usgs.gov/).
- USGS 3DEP LiDAR (LAZ) for the same area (https://apps.nationalmap.gov/lidar-explorer/#/).
- The Phoenix Council District boundary shapefile.
- Microsoft Building Footprints, road centerlines, and wetland polygons (used
  only to seed built-surface and water labels during labeling).

## Configuring paths

The scripts were written for a SLURM cluster. Absolute paths appear as
`/path/to/...` placeholders: edit the marked `CONFIG` block at the top of each
script, or pass the corresponding command-line argument where one exists (the
training scripts accept `--tile-root`, `--train-root`, `--encoder-ckpt`, and so
on). The `.sh` submission scripts also use `YOUR_SLURM_ACCOUNT` and
`YOUR_EMAIL@example.com`; replace these before submitting.

## Acknowledgments

Supported by NSF grant DEB-2224662 (CAP LTER). Computing resources provided by
Research Computing at Arizona State University.

## License

MIT. See [LICENSE](LICENSE).
