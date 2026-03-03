# OC20 Knowledge Distillation

This repository contains the implementation used to distill the EquiformerV2 model on the OC20 dataset.

## Environment setup

```bash
mamba env create -f env.yml
conda activate ark
cd fairchem
pip install -e .
cd ..
```

## Data setup (OC20)

Use official OC20 data instructions or prepare local LMDB paths that match the YAML configs.

Example O-adsorbate download script included in this repo:

```bash
cd fairchem
python scripts/download_data_Oads.py --task is2re --split Oads --num-workers 8 --ref-energy
cd ..
```

## Training and inference

### ARK distillation training

```bash
python main_oc20.py \
  --mode train \
  --config-yml oc20/configs/s2ef/2M/equiformer_v2/153M_exp_Oads.yml \
  --run-dir models \
  --print-every 200 \
  --amp \
  --checkpoint save_models/eq2_153M_ec4_allmd.pt
```

### Validate a checkpoint

```bash
python main_oc20.py \
  --mode validate \
  --config-yml oc20/configs/s2ef/2M/equiformer_v2/153M_exp.yml \
  --run-dir models \
  --amp \
  --checkpoint <path-to-checkpoint.pt>
```