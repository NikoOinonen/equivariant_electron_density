#!/bin/bash
#SBATCH --time=02-00:00:00      # Job time allocation
#SBATCH --gres=gpu:4            # Request GPU(s)
#SBATCH -p gpu-h100-80g,gpu-a100-80g    # Request specific GPU partitions
#SBATCH --mem=64G               # Memory
#SBATCH -c 8                    # Number of cores
#SBATCH -J e3nn_train_density   # Job name
#SBATCH -o logs/train_%j.log    # Output file

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

num_gpus=$(echo "$SLURM_JOB_GPUS" | sed -e $'s/,/\\\n/g' | wc -l)
echo "Number of GPUs: $num_gpus"

# Run script
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --max_restarts 0 \
    train_density.py \
        --dataset ../generate_density_datasets/dataset_train.pickle \
        --testset ../generate_density_datasets/dataset_val.pickle \
        --num_epochs 40 \
        --test_interval 1 \
        --batch_average 2 \
        --lr 1.5e-3 \
        --lr_warm 8000 \
        --lr_decay 350e3 \
        --irreps_hidden "128-128-128-128" \
        --num_layers 6 \
        --correlation_order 3 \
