# chesapeake

Pretrains the U-Net encoder on the ChesapeakeCVPR land-cover dataset. The best
encoder checkpoint produced here is the transfer source for the Phoenix model
(`phoenix/training/train_phoenix.py --encoder-ckpt ...`).

## Run

```bash
# ResNet18 encoder (the backbone used by the headline Phoenix model)
python train_chesapeake_swin.py --arch resnet18 --bs 128 --lr 1e-4 --seed 1

# Swin-v2-B encoder (used by the encoder-comparison experiments)
python train_chesapeake_swin.py --arch swin_b --bs 64 --accum 2 --lr 1e-4 --seed 1
```

## Output

Each run writes to `<TRAIN_ROOT>/<arch>_bs<BS>_lr<LR>_seed<SEED>/<timestamp>/`:

- `best_unet_e<NNN>_miou<X.XXXX>.pth` full model at each new best.
- `best_encoder_e<NNN>_miou<X.XXXX>.pth` encoder only, keys prefixed `encoder.`.
  This is the file to pass to the Phoenix training scripts.
- `train_log.csv`, `training_curve.png`, `latest_checkpoint.pth`.

## Notes

- Run in the `pytorch_gpu` conda environment (`environment_pytorch_gpu.yml`).
- Edit `ROOT` and `TRAIN_ROOT` at the top of the script for your dataset and
  output locations.
- On compute nodes without internet, pre-download the ImageNet weights on a
  login node first (see the script header for the exact command).
