# ARK: Angular Relational Knowledge Distillation

This repository contains the implementation used in:

**Angular relational knowledge distillation of machine learning interatomic potentials for scalable catalyst exploration**  

ARK distills relational physics from a large teacher MLIP into a compact student by aligning **edge-level relational vectors** with a contrastive objective.

<p align="center">
  <img src="assets/overview.png" width="90%">
</p>

## Reported paper highlights

- ARK validates across OC20, OMat24, and SPICE benchmarks.
- On OC20, ARK-trained student reports `231.7 meV` energy MAE and `5.8 meV/Angstrom` force MAE.
- In the ORR screening case study, ARK enables up to `11.9x` acceleration in the multi-fidelity workflow.

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

Equivalent script:

```bash
sh scripts/train/oc20/s2ef/equiformer_v2/153M_exp.sh
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

### Predict

```bash
python main_oc20.py \
  --mode predict \
  --config-yml oc20/configs/s2ef/all_md/equiformer_v2/31M_exp.yml \
  --run-dir models \
  --amp \
  --checkpoint <path-to-checkpoint.pt>
```

## Acknowledgement

This repository builds on:

- EquiformerV2: https://github.com/atomicarchitects/equiformer_v2
- FAIR-Chem / OCP tooling: https://github.com/FAIR-Chem/fairchem
