# E2GAN-for-StyleGAN2-ADA

E2GAN-inspired selective LoRA fine-tuning extension for [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) (PyTorch).

This repository preserves the **original StyleGAN2-ADA backbone** and adds a minimal, modular, parameter-efficient fine-tuning path on top of it.

> **Note:** When LoRA is disabled (the default), this repo behaves identically to the official NVLabs StyleGAN2-ADA.

---

## Quick Start

### 1. Prepare Dataset (unchanged from StyleGAN2-ADA)

```bash
python dataset_tool.py --source=~/datasets/my_images --dest=~/datasets/my_images.zip
```

### 2. Fine-Tune with LoRA

```bash
python train.py \
  --outdir=training-runs \
  --data=~/datasets/my_images.zip \
  --resume=pretrained_base.pkl \
  --use-lora=true \
  --lora-rank=4 \
  --lora-alpha=1.0 \
  --freeze-g-backbone=true \
  --gpus=1 --kimg=100 --snap=10
```

This will:
- Load the pretrained base model
- Inject LoRA adapters into the generator's affine/style layers
- Freeze all non-LoRA generator parameters
- Train only the LoRA adapters
- Save adapter-only `.pt` checkpoints alongside full snapshots

### 3. Generate with LoRA Adapter

```bash
python generate.py \
  --network=pretrained_base.pkl \
  --lora-ckpt=training-runs/your-run/network-snapshot-000100_lora.pt \
  --seeds=0-9 \
  --outdir=generated
```

### 4. Generate without LoRA (original mode)

```bash
python generate.py \
  --network=pretrained_base.pkl \
  --seeds=0-9 \
  --outdir=generated
```

### 5. Inspect Trainable Parameters

```bash
python scripts/inspect_trainable_params.py \
  --network=pretrained_base.pkl \
  --lora-rank=4
```

---

## CLI Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--use-lora` | bool | `false` | Enable LoRA adapters on the generator |
| `--lora-rank` | int | `4` | Rank of the low-rank decomposition |
| `--lora-alpha` | float | `1.0` | LoRA scaling factor (effective scaling = alpha/rank) |
| `--freeze-g-backbone` | bool | `true` | Freeze all non-LoRA generator parameters |

When `--use-lora` is not set, training behaves exactly like the original StyleGAN2-ADA.

---

## Artifact Formats

| Artifact | Format | Description |
|---|---|---|
| `network-snapshot-*.pkl` | pickle | Full G/D/G_ema state. Extension-specific; only works inside this repo. |
| `network-snapshot-*_lora.pt` | torch dict | **Primary portable artifact.** Adapter-only weights + metadata. |

To use a LoRA-adapted model elsewhere: take the original base `.pkl` + the adapter `.pt`, and re-inject.

---

## Smoke Test

```bash
python tests/test_lora_smoke.py
```

CPU-only; no GPU or dataset required.

---

## Architecture

```
adapters/
  __init__.py          # Package init
  lora_layers.py       # LoRALinear module
  inject.py            # inject_lora, extract/load utilities
scripts/
  inspect_trainable_params.py  # Dry-run param inspection
tests/
  test_lora_smoke.py   # Automated smoke test
```

### LoRA Target Layers (MVP)

The MVP targets **affine/style layers** (`SynthesisLayer.affine` and `ToRGBLayer.affine`) — the fully-connected layers that transform W-space latents into per-channel style modulation coefficients.

These are small (w_dim × channels), high-leverage layers that control feature-map statistics.

### Modified Backbone Files

Only three backbone files have small, guarded additions:
- `train.py` — CLI flags
- `training/training_loop.py` — LoRA injection, optimizer, snapshotting
- `generate.py` — `--lora-ckpt` option

All additions are guarded by `if lora_kwargs:` and have zero effect when LoRA is disabled.

---

## License

The original StyleGAN2-ADA code is licensed by NVIDIA Corporation. See [LICENSE.txt](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).  
The E2GAN-LoRA extension code is provided under the MIT license.

Still under building...