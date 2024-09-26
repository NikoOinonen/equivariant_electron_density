#!/bin/bash
#SBATCH --time=01-00:00:00      # Job time allocation
#SBATCH --gres=gpu:4            # Request GPU(s)
#SBATCH -p gpu-h100-80g,gpu-a100-80g,gpu-v100-32g    # Request specific GPU partitions
#SBATCH --mem=64G               # Memory
#SBATCH -c 8                    # Number of cores
#SBATCH -J e3nn_train_density   # Job name
#SBATCH -o train_%j.log         # Output file
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

num_gpus=$(echo "$SLURM_JOB_GPUS" | sed -e $'s/,/\\\n/g' | wc -l)
echo "Number of GPUs: $num_gpus"

lr=3e-2
lr_warmup=8000
lr_decay=10000
batch_average=2

# Run script
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --max_restarts 0 \
    train_density.py \
        --dataset ../generate_density_datasets/dataset_train.pickle \
        --testset ../generate_density_datasets/dataset_val.pickle \
        --epochs 20 \
        --batch_average $batch_average \
        --learning_rate $lr \
        --lr_warmup_batches $lr_warmup \
        --lr_decay_batches $lr_decay \
        --ldep true \
        --run_comment "gpu${num_gpus}_avg${batch_average}_lr${lr}_warmup${lr_warmup}_decay${lr_decay}"
