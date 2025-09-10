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

lr=1.5e-3
lr_warmup=8000
lr_decay=350e3
batch_average=2
irreps_hidden="128-128-128-128"
num_layers=6
correlation_order=3
free_density_input="True"
exclude_elements=""
split=1000
epochs=3000
test_epochs=75

batch_size=$(( num_gpus * batch_average ))
comment="bs${batch_size}_lr${lr}-${lr_warmup}-${lr_decay}_irreps${irreps_hidden}x${num_layers}_corr${correlation_order}_split${split}"
if [ "$free_density_input" != "" ]; then
    comment="${comment}_density_input"
fi
if [ "$exclude_elements" != "" ]; then
    comment="${comment}_exclude${exclude_elements}"
fi

# Run script
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --max_restarts 0 \
    train_density.py \
        --dataset ../generate_density_datasets/dataset_train.pickle \
        --testset ../generate_density_datasets/dataset_val.pickle \
        --epochs $epochs \
        --test_epochs $test_epochs \
        --train_split $split \
        --batch_average $batch_average \
        --learning_rate $lr \
        --lr_warmup_batches $lr_warmup \
        --lr_decay_batches $lr_decay \
        --irreps_hidden $irreps_hidden \
        --num_layers $num_layers \
        --correlation_order $correlation_order \
        --free_density_input "$free_density_input" \
        --exclude_elements "$exclude_elements" \
        --ldep true \
        --run_comment $comment \
