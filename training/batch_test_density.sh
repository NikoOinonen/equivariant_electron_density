#!/bin/bash
#SBATCH --time=00-08:00:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPU(s)
#SBATCH -p gpu-h100-80g,gpu-a100-80g    # Request specific GPU partitions
#SBATCH --mem=64G               # Memory
#SBATCH -c 4                    # Number of cores
#SBATCH -J e3nn_test_density    # Job name
#SBATCH -o logs/test_%j.log     # Output file
#SBATCH --exclude dgx[4-7]      # The dgx nodes are somehow slow

# Load modules
module load mamba
source activate e3nn_density

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME

# Print environment info
which python
python --version
conda info --envs
conda list
pip list

# Run script
python -u test_density.py \
    --dataset ../generate_density_datasets/dataset_val.pickle ../generate_density_datasets/dataset_train.pickle \
    --run_dir "runs/Oct28_20-29-16_gpu11.int.triton.aalto.fi_gpu4_avg2_lr2e-2_warmup4000_decay10000_irreps125-40-25-15x7_density_input_exclude15" \
    --include_elements "15"

    # --dataset ../data/water_density_testset.pkl \
