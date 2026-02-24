#!/usr/bin/bash
#SBATCH -J transformer
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 6:00:00
#SBATCH --switches=1
#SBATCH -o log/%A/log.out
#SBATCH -e log/%A/err.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export LOGLEVEL=INFO
JOB_ID=$SLURM_JOB_ID

# Ensure logging directory exists
LOG_DIR="log/$JOB_ID"
CONFIG_DIR="$LOG_DIR/config"
mkdir -p "$CONFIG_DIR"

# Check if all configuration files exist before proceeding
for config_file in "$@"; do
    if [ ! -f "$config_file" ]; then
        echo "Error: Required configuration file '$config_file' not found. Exiting."
        exit 1
    fi
done

# Copy and rename configuration files
count=1
CONFIG_LIST=()

for config_file in "$@"; do
    base_name=$(basename "$config_file")
    new_name="${count}_${base_name}"
    cp "$config_file" "$CONFIG_DIR/$new_name"
    CONFIG_LIST+=("$CONFIG_DIR/$new_name")
    ((count++))
done

# Convert config list to a space-separated string
CONFIG_ARGS=$(printf "%s " "${CONFIG_LIST[@]}")

# Launch the job
JOB_ID=$JOB_ID srun uv run torchrun \
    --standalone \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    -m src.decent_train --cfg-list $CONFIG_ARGS

