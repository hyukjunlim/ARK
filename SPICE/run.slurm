#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 40
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:volta:2
#SBATCH -t 72:00:00
#SBATCH -o /home/gridsan/hlim/mlff/MLFF-distill/output/escaip.%j.out
#SBATCH -e /home/gridsan/hlim/mlff/MLFF-distill/output/escaip.%j.err
#SBATCH -J escaip

output_dir="/home/gridsan/hlim/mlff/MLFF-distill/output"
mkdir -p $output_dir

num_gpus=2
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /etc/profile
source /state/partition1/llgrid/pkg/anaconda/anaconda3-2023a/etc/profile.d/conda.sh
conda deactivate
conda activate escaip

choice=2

case "$choice" in
    0) # monomers
        config_path="configs/SPICE/monomers/hessian/gemnet-dT-small-ark.yml"
        ;;
    1) # solvated_amino_acids
        config_path="configs/SPICE/solvated_amino_acids/hessian/gemnet-dT-small-ark.yml"
        ;;
    2) # iodine
        config_path="configs/SPICE/iodine/hessian/gemnet-dT-small-ark.yml"
        ;;
    *)
        echo "Invalid choice: $choice"
        echo "Please use one of: 0 (monomers), 1 (solvated_amino_acids), 2 (iodine)"
        exit 1
        ;;
esac

python main.py --mode train --config-yml "$config_path" --run-dir 'models'