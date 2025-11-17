#!/usr/bin/bash

### Task name
#SBATCH --account=sds_baek_energetic
#SBATCH --job-name=train_poseidon

### Output file
#SBATCH --output=results/slrm_logs/train_poseidon_%j.out


### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=34

### How much memory in total (MB)
#SBATCH --mem=150G


### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
##SBATCH --gres=gpu:a6000:2
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80gb

### Partition
#SBATCH --partition=gpu

#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# debug=true
sim_name="poseidon_02"
# Set up paths
base_dir="/scratch/zsa8rk/poseidon"
python_exec="${base_dir}/scOT/train.py"
checkpoint_path="${base_dir}/results"
data_dir="/scratch/zsa8rk/datasets"
config_file="${base_dir}/configs/run.yaml"

export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# finetune:
# path="/home/flwi01/coding/poseidon/results/poseidon_test00/Large-Physics-Foundation-Model/poseidon_test00/checkpoint-200"
resume_training=true


accelerate_args="
--config_file $base_dir/configs/accel_config.yaml"


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################

exec_args="--config $config_file \
    --wandb_run_name  $sim_name \
    --wandb_project_name Large-Physics-Foundation-Model \
    --checkpoint_path  $checkpoint_path \
    --data_path  $data_dir"

if [ -n "$path" ]; then
    exec_args="$exec_args --finetune_from $path"
fi
if [ "$resume_training" = true ]; then
    exec_args="$exec_args --resume_training"
fi

# Capture Python output and errors in a variable and run the script
echo "Starting training"
accelerate launch $accelerate_args $python_exec $exec_args
