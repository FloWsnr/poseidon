#!/usr/bin/bash

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
sim_name="poseidon_test03"
# Set up paths
python_bin="/home/flwi01/miniforge3/envs/gphyt/bin/python"
base_dir="/home/flwi01/coding/poseidon"
python_exec="${base_dir}/scOT/train.py"
checkpoint_path="${base_dir}/results"
data_dir="/home/flwi01/coding/gphyt_datasets"
config_file="${base_dir}/configs/run.yaml"

export OMP_NUM_THREADS=1 # (num cpu - num_workers) / num_gpus

# finetune:
# path="/home/flwi01/coding/poseidon/results/poseidon_test00/Large-Physics-Foundation-Model/poseidon_test00/checkpoint-200"
resume_training=false


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
