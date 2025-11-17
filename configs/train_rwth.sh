#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=train_poseidon

### Output file
#SBATCH --output=results/slrm_logs/train_poseidon_%j.out

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --cpus-per-task=48
##SBATCH --exclusive

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=01:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:2


#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

# export CUDA_VISIBLE_DEVICES=1,3

######################################################################################
############################# Set paths ##############################################
######################################################################################
# debug mode
# debug=true
sim_name="poseidon_test_restart02"
# Set up paths
base_dir="/hpcwork/rwth1802/coding/poseidon"
python_exec="${base_dir}/scOT/train.py"
checkpoint_path="${base_dir}/results"
data_dir="/hpcwork/rwth1802/coding/General-Physics-Transformer/data/datasets"
config_file="${base_dir}/configs/run.yaml"

export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# finetune:
# path="/hpcwork/rwth1802/coding/poseidon/results/Large-Physics-Foundation-Model/poseidon_test_restart01/checkpoint-10"
# Comment out path when resuming training - checkpoint will be auto-detected
resume_training=true

accelerate_args="--config_file ./configs/accel_config.yaml \
--num_cpu_threads_per_process 23"


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
