#!/bin/bash
#SBATCH --time=00-02:00:00      # Job time allocation
#SBATCH --gres=gpu:1            # Request GPU(s)
#SBATCH -p gpu-h100-80g,gpu-a100-80g    # Request specific GPU partitions
#SBATCH --mem=16G               # Memory
#SBATCH -c 8                    # Number of cores
#SBATCH -J e3nn_test_density    # Job name
#SBATCH -o logs/test_%j.log     # Output file

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
    --testset ../generate_density_datasets/data_list_ccsd-cid_test.json \
    --test_samples 1000 \
    --run_dir "runs/250929-123912_bs32_ns9372_lr1.5e-03-8000-3.5e+05_irreps128-128-128-128x6_corr3" \
    --num_proc_test 8
    # --include_elements "15"
