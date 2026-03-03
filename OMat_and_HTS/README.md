# CRACK

## Installation

```bash
conda create -n crack python=3.12 -y
conda activate crack
git clone https://github.com/hyukjunlim/CRACK_fairchem.git
cd CRACK_fairchem
pip install -e packages/fairchem-core
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

## Training

```bash
sh run_slurm.sh
```