# OMat24 Knowledge Distillation and High-throughput Screening

## Installation

```bash
conda create -n ark python=3.12 -y
conda activate ark
pip install -e packages/fairchem-core
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

## Training

```bash
sh run_slurm.sh
```

## Checkpoints

Checkpoints for OMat24 and high-throughput screening experiment are available on the [Google Drive](https://drive.google.com/drive/folders/1s5qrevUchYDRIou1Y8SnTIKZZq-AGLyq?usp=sharing).