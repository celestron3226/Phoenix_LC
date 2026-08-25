# phoenix/experiments

Ablations and alternative methods kept for the record. These are not part of the
headline pipeline but reproduce the comparisons behind it.

| file | what it does |
|------|--------------|
| `encoder_experiment_week12.py` | Encoder-initialization comparison (random / ImageNet / Chesapeake / SatlasPretrain) across ResNet and Swin-v2-B backbones. |
| `encoder_experiment_week13.py` | Coordinate-encoding x patch-size grid on top of the Week 12 encoder arms. |
| `train_variant.py` | Trains the two alternative methods: `hybrid` (DL detects vegetation, CHM rule splits it) and `planb` (weakly supervised on rule pseudo-labels). |
| `predict_variant.py` | City-wide prediction for the hybrid and Plan-B models. |
| `predict_local.py` | Single-quad local prediction for quick visual checks. |

## Notes

- `train_variant.py` and `predict_variant.py` import shared code from
  `../training/phoenix_common.py`; each adds that folder to `sys.path` at the
  top, so they run from this directory without any copying.
- `encoder_experiment_week12.py`, `encoder_experiment_week13.py`, and
  `predict_local.py` were run on a local Windows workstation and still contain
  local `C:\...` paths in their `CONFIG` sections. Edit those before running.
- The variant scripts share the same `/path/to/...` placeholders and CLI
  arguments as the core training scripts, and run in the `pytorch_gpu`
  conda environment.
