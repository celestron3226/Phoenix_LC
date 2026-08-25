# phoenix/label

Generates training labels in three passes, then hands off to manual editing in
ArcGIS. The hand-corrected shapefiles are the labels used for training.

## Order

1. `01_make_patches.py` samples N patches whose centers fall inside the Phoenix
   boundary and writes, per patch, a NAIP window plus CHM / NDVI clipped to the
   same grid and a seed `label.shp` (Building, Road, Water from vector sources).
2. `02_make_veg_seeds.py` (GPU) adds vegetation seed polygons per patch:
   grass and shrub from NDVI + CHM rules, and tree crowns from two methods
   (a CHM rule plus DeepForest), each run through SAM3 and saved separately so
   they can be compared and pruned.
3. `03_build_bundles.py` packages every 50 patches into one downloadable
   `group_XXXX-YYYY.tar.gz` with reference VRTs and a single merged
   `label_group.shp` for editing.

## Manual step

Open each bundle in ArcGIS, correct the pre-made polygons, and digitize what is
missing. The corrected shapefiles become the `Label_modified` groups that
`phoenix/training/prepare_tiles_phoenix.py` reads.

## Run (SLURM)

```bash
sbatch 01_make_patches.sh
sbatch 02_make_veg_seeds.sh      # GPU job array
sbatch 03_build_bundles.sh
```

## Requirements

- Passes 1 and 3 run in the `geopandas` conda environment
  (`environment_geopandas.yml`).
- Pass 2 runs on GPU in the `segment` environment (`environment_segment.yml`),
  which ships DeepForest and segment-geospatial (SAM3). It needs a prior
  `hf auth login` so SAM3 can download its checkpoint.
- Edit the `CONFIG` paths at the top of each script before running.
