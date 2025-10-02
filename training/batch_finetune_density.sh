#!/bin/bash
#SBATCH --time=00-16:00:00              # Job time allocation
#SBATCH --gres=gpu:2                    # Request GPU(s)
#SBATCH -p gpu-h100-80g,gpu-a100-80g    # Request specific GPU partitions
#SBATCH --mem=64G                       # Memory
#SBATCH -c 4                            # Number of cores
#SBATCH -J e3nn_ft_density              # Job name
#SBATCH -o logs/finetune_%j.log         # Output file

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
    finetune_density.py \
        --base_model "runs/250923-165703_bs8_ns72864_lr1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3_exc15" \
        --dataset ../generate_density_datasets/data_list_ccsd-cid_train.json \
        --testset ../generate_density_datasets/data_list_ccsd-cid_test.json \
        --num_epochs 1000 \
        --test_interval 10 \
        --batch_size 4 \
        --lr 1e-4 \
        --lr_warm  8000\
        --lr_decay 350e3 \
        --include_elements "15" \
        --finetune_method "restart-all"
