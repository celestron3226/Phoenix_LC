# phoenix/training

The core end-to-end pipeline (Plan A): turn the hand-modified labels into tiles,
fine-tune the U-Net from the Chesapeake encoder, and predict city-wide.

`phoenix_common.py` holds the normalization and coordinate-encoding code shared
by training and prediction, so both use byte-identical preprocessing.

## Run

```bash
# 1. Build .npz tiles from the hand-modified label groups (CPU).
python prepare_tiles_phoenix.py
#    -> prints the timestamped tile folder to pass as --tile-root below.

# 2. Fine-tune the U-Net (ResNet18 encoder from the Chesapeake checkpoint).
python train_phoenix.py --bs 16 --lr 1e-4 --seed 1 --coord xy \
    --tile-root  /path/to/Phoenix/Result/tiles_<stamp>_n<count> \
    --encoder-ckpt /path/to/chesapeake_pretrain/best_encoder_e032_miou0.7219.pth

# 3. City-wide prediction over all NAIP quads (SLURM array, or serial).
python predict_phoenix.py --run-dir <RUN_DIR>

# 4. Mosaic the per-quad outputs.
cd <OUT_DIR>
gdalbuildvrt landcover_phoenix.vrt landcover_*.tif
gdalbuildvrt veg_phoenix.vrt veg_*.tif
```

## Coordinate encoding

`--coord xy` (default) adds two CoordConv channels of per-pixel normalized
coordinates. `--coord none` disables them (ablation) and `--coord sincos` uses
multi-scale Fourier features. Prediction reads the mode from the run's
`run_config.json`, so it can never mismatch training.

## Notes

- Run these in the `pytorch_gpu` conda environment
  (`environment_pytorch_gpu.yml`).
- Paths appear as `/path/to/...` placeholders (edit the `CONFIG` block or pass
  the matching CLI argument).
- `predict_phoenix.py` picks the highest-mIoU `best_unet_*.pth` in the run dir
  automatically; override with `--model`.
