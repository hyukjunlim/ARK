# ARK: Angular Relational Knowledge Distillation

This repository contains the implementation used in:

**Angular relational knowledge distillation of machine learning interatomic potentials for scalable catalyst exploration**  
(`paperworks/main.tex`)

ARK distills relational physics from a large teacher MLIP into a compact student by aligning **edge-level relational vectors** with a contrastive objective.

<p align="center">
  <img src="docs/CRACK_Overview.png" width="90%">
</p>

## What this repository currently contains

- OC20-focused ARK training/inference pipeline built on EquiformerV2.
- Teacher-student distillation in a single model (`nets/equiformer_v2/equiformer_v2_oc20.py`).
- ARK contrastive edge loss and n2n loss in `oc20/trainer/forces_trainer_v2.py`.
- Configs and scripts for OC20 O-adsorbate / 200k experiments under `oc20/configs` and `scripts/train`.
- Bundled `fairchem/` codebase for OCP tooling.

## Important naming note

The method name in the paper is **ARK**.  
Some code/config keys still use older internal names:

- `crack_loss(...)` in `oc20/trainer/forces_trainer_v2.py` is the ARK relational contrastive loss.
- `optim.crack_coefficient` in YAML controls ARK loss weight.

## Repository map

- `main_oc20.py`: main entrypoint for train/validate/predict.
- `oc20/configs/s2ef/...`: experiment YAMLs.
- `oc20/trainer/forces_trainer_v2.py`: losses, train loop, evaluation, visualization.
- `nets/equiformer_v2/equiformer_v2_oc20.py`: EquiformerV2 teacher + 2-layer student branch.
- `scripts/train/oc20/s2ef/equiformer_v2/*.sh`: example launch scripts.
- `paperworks/main.tex`: manuscript source.

## Environment setup

```bash
conda env create -f env/env_equiformer_v2.yml
conda activate equiformer_v2
export PYTHONNOUSERSITE=True
cd fairchem
pip install -e .
cd ..
```

If `main_oc20.py` fails with missing packages (for example `submitit`), install them in this environment.

## Data setup (OC20)

Use official OC20 data instructions or prepare local LMDB paths that match the YAML configs.

Example O-adsorbate download script included in this repo:

```bash
cd fairchem
python scripts/download_data_Oabs.py --task is2re --split Oabs --num-workers 8 --ref-energy
cd ..
```

Provided configs expect these paths:

- `oc20/configs/s2ef/all_md/equiformer_v2/31M_exp.yml`
- `oc20/configs/s2ef/all_md/equiformer_v2/153M_exp.yml`
- `dataset.train.src: datasets/oc20/Oabs_train/`
- `dataset.val.src: datasets/oc20/Oabs_val/`

Some configs under `oc20/configs/s2ef/2M/equiformer_v2/` still contain machine-specific absolute paths (`/DATA/...`). Update them before running.

## Training and inference

### 1) ARK distillation training (example: 83M config)

```bash
python main_oc20.py \
  --mode train \
  --config-yml oc20/configs/s2ef/2M/equiformer_v2/83M_exp.yml \
  --run-dir models \
  --print-every 200 \
  --amp \
  --checkpoint save_models/eq2_153M_ec4_allmd.pt
```

Equivalent script:

```bash
sh scripts/train/oc20/s2ef/equiformer_v2/83M_exp.sh
```

### 2) ARK distillation training (example: 153M Oabs config)

```bash
python main_oc20.py \
  --mode train \
  --config-yml oc20/configs/s2ef/2M/equiformer_v2/153M_exp_Oabs.yml \
  --run-dir models \
  --print-every 200 \
  --amp \
  --checkpoint save_models/eq2_153M_ec4_allmd.pt
```

Equivalent script:

```bash
sh scripts/train/oc20/s2ef/equiformer_v2/153M_exp.sh
```

### 3) Validate a checkpoint

```bash
python main_oc20.py \
  --mode validate \
  --config-yml oc20/configs/s2ef/2M/equiformer_v2/83M_exp.yml \
  --run-dir models \
  --amp \
  --checkpoint <path-to-checkpoint.pt>
```

### 4) Predict

```bash
python main_oc20.py \
  --mode predict \
  --config-yml oc20/configs/s2ef/all_md/equiformer_v2/31M_exp.yml \
  --run-dir models \
  --amp \
  --checkpoint <path-to-checkpoint.pt>
```

## ARK-related optimization keys

From the YAML `optim` block:

- `crack_coefficient`: ARK relational contrastive loss weight.
- `n2n_coefficient`: node-to-node feature matching loss weight.
- `energy_coefficient`: energy loss weight.
- `energy_coefficient2`: per-atom energy consistency loss weight.
- `lr_initial_e`, `lr_initial_f`, `lr_initial_student`: separate learning rates for parameter groups.

## Reported paper highlights

From `paperworks/main.tex`:

- ARK validates across OC20, OMat24, and SPICE benchmarks.
- On OC20, ARK-trained student reports `231.7 meV` energy MAE and `5.8 meV/Angstrom` force MAE.
- In the ORR screening case study, ARK enables up to `11.9x` acceleration in the multi-fidelity workflow.

## Acknowledgement

This repository builds on:

- EquiformerV2: https://github.com/atomicarchitects/equiformer
- FAIR-Chem / OCP tooling: https://github.com/FAIR-Chem/fairchem
