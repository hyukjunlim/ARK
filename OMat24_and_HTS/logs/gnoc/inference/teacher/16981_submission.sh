#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=3
#SBATCH --error=/home/hyukjunlim03/ccel/fairchem/logs/%j/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=''
#SBATCH --mem=80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/hyukjunlim03/ccel/fairchem/logs/%j/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=14400
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/hyukjunlim03/ccel/fairchem/logs/%j/%j_%t_log.out --error /home/hyukjunlim03/ccel/fairchem/logs/%j/%j_%t_log.err /home/hyukjunlim03/anaconda3/envs/ark/bin/python -u -m submitit.core._submit /home/hyukjunlim03/ccel/fairchem/logs/%j
