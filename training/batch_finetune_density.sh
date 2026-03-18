#!/bin/bash
#SBATCH --time=00-08:00:00              # Job time allocation
#SBATCH --gres=gpu:1                    # Request GPU(s)
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
        --base_model "runs/251001-060742_bs16_ns74976_lr1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3" \
        --dataset ../generate_density_datasets/data_lists/perturbed_0.05-0.10_mix_train.json \
        --testset ../generate_density_datasets/data_lists/perturbed_0.05-0.10_val.json \
        --runs_base_dir runs_ft_perturbed \
        --num_epochs 400 \
        --test_interval 8 \
        --batch_size 8 \
        --lr 8e-4 \
        --lr_warm  8000 \
        --lr_decay 350e3 \
        --finetune_method "elora" \
        --elora_rank 16 \

        # --finetune_method "elora" \
        # --elora_rank 24 \
        # --weight_decay 1e-4

        # --dataset ../generate_density_datasets/data_lists/perturbed_0.05-0.15_mix_train.json \
        # --testset ../generate_density_datasets/data_lists/perturbed_0.05-0.15_val.json \
        # --dataset ../generate_density_datasets/data_lists/perturbed_0.05-0.10_mix_train.json \
        # --testset ../generate_density_datasets/data_lists/perturbed_0.05-0.10_val.json \
        # --dataset ../generate_density_datasets/data_lists/perturbed_0.05_mix_train.json \
        # --testset ../generate_density_datasets/data_lists/perturbed_0.05_val.json \
        # --dataset ../generate_density_datasets/data_lists/P_mix_train.json \
        # --testset ../generate_density_datasets/data_lists/P_only_val.json \
        # --dataset ../generate_density_datasets/data_lists/ccsd-cid_train.json \
        # --testset ../generate_density_datasets/data_lists/ccsd-cid_val.json \
        # --include_elements 15 \
