#!/bin/bash

# 1. 6D pose estimator
cd 6D_pose_estimator

run_class() {
    local class=$1
    local cuda_device=$2
    local display_position=$3

    CUDA_VISIBLE_DEVICES=${cuda_device} python3 scripts/inference_multi.py \
        --category ${class} \
        --ckpt /home/xingrui/vqa/pretrain_6d_pose/output/experiments/0407_superclevr_${class}/ckpts/saved_model_15000.pth \
        --dataset_path /home/xingrui/publish/superclevr_3D_questions/output/ver_texture_new \
        --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
        --px_sample 31 \
        --py_sample 31 \
        --save_results output/json/ver_texture_new \
        --display_position ${display_position}

    echo "Finished evaluation for $class on CUDA device $cuda_device"
}

# List of names
classes=("aeroplane" "bus" "bicycle" "car" "motorbike")

# Loop through names and assign CUDA device index
cuda_device=1
display_position=0
for class in "${classes[@]}"; do
    # run_class $class $cuda_device $display_position &
    # run_class $class $cuda_device $display_position
    run_class $class 3 $display_position
    # Increment CUDA device index
    ((cuda_device++))
    ((display_position++))
    
    # If CUDA device index exceeds 4, reset it to 1
    if [ $cuda_device -gt 3 ]; then
        cuda_device=1
    fi
done




# 2. 3D-NMS
