# phoenix/data

Builds the three input rasters the model consumes, all aligned 1:1:1 per NAIP
tile on the same grid.

## Order

1. `naip/jp2_combine.py` converts NAIP JP2 to tiled GeoTIFF (cheap windowed
   reads for everything downstream). No pixel values change.
2. `ndvi/ndvi_per_tile.py` computes NDVI = (NIR - Red) / (NIR + Red) per tile.
3. `chm/download_laz.py` then `chm/LAZ_to_CHM.py` download USGS LiDAR and build
   a per-NAIP-tile canopy height model (CHM = DSM - DTM), reprojected onto the
   NAIP grid at 0.3 m.

## Run (SLURM)

```bash
sbatch naip/jp2_combine.sh
sbatch ndvi/ndvi_per_tile.sh
sbatch chm/download_laz.sh
sbatch chm/LAZ_to_CHM.sh
```

Every stage is resume-friendly: existing outputs are skipped, so a job can be
resubmitted after a timeout.

## Requirements

Run these in the `geopandas` conda environment (`environment_geopandas.yml`).
It provides GDAL for all stages; the CHM step (`chm/LAZ_to_CHM.py`) additionally
needs PDAL on PATH. Edit the `CONFIG` paths at the top of each script before
running.
