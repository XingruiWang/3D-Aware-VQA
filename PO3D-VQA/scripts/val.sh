#!/bin/bash

# 1.1 6D pose estimator
# cd 6D_pose_estimator

# run_class() {
#     local class=$1
#     local cuda_device=$2
#     local display_position=$3

#     CUDA_VISIBLE_DEVICES=${cuda_device} python3 scripts/inference_multi.py \
#         --category ${class} \
#         --ckpt /home/xingrui/vqa/pretrain_6d_pose/output/experiments/0407_superclevr_${class}/ckpts/saved_model_15000.pth \
#         --dataset_path /home/xingrui/publish/superclevr_3D_questions/output/ver_texture_new \
#         --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
#         --px_sample 31 \
#         --py_sample 31 \
#         --save_results output/json/ver_texture_new \
#         --display_position ${display_position}

#     echo "Finished evaluation for $class on CUDA device $cuda_device"
# }

# # List of names
# classes=("aeroplane" "bus" "bicycle" "car" "motorbike")

# # Loop through names and assign CUDA device index
# cuda_device=1
# display_position=0
# for class in "${classes[@]}"; do
#     # run_class $class $cuda_device $display_position &
#     # run_class $class $cuda_device $display_position
#     run_class $class 3 $display_position
#     # Increment CUDA device index
#     ((cuda_device++))
#     ((display_position++))
    
#     # If CUDA device index exceeds 4, reset it to 1
#     if [ $cuda_device -gt 3 ]; then
#         cuda_device=1
#     fi
# done


# # 1.2 Process
# python process/post_process_new.py


# # 1.3 re-render and calculate pose / occlusion / part
# python process/mesh_projection.py



# 1.4 attributes 
cd attr_net

CUDA_VISIBLE_DEVICES=2 python tools/run_test_prob.py \
    --load_path /home/xingrui/vqa/superclevr-NSVQA/data/attr_net/outputs/trained_model/checkpoint_best.pt \
    --img_dir /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/images \
    --pred_bbox /home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/pred_prob.json \
    --output_file data/scene_pred-nemo.json \
    --dataset superclevr \
    --type object

# cd attr_net

# CUDA_VISIBLE_DEVICES=3 python tools/run_test_prob.py \
#     --load_path /home/xingrui/vqa/superclevr-NSVQA-baseline/data/pnsvqa/attr_net/outputs/trained_model/checkpoint_best.pt \
#     --img_dir /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/images \
#     --pred_bbox /home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/anno_prob_part_test.json \
#     --output_file ../../data/parts-nemo/reason/scene_pred-nemo-pred-large.json \
#     --dataset superclevr \
#     --type part

# Reasoning
cd ${BASE}/reason

python tools/preprocess_questions_superclevr.py \
    --input_questions_json /home/xingrui/publish/superclevr_3D_questions/output/superclevr_questions_depth.json \
    --output_h5_file ../data/superclevr_z_direction/preprocess/SuperCLEVR_questions.h5 \
    --output_vocab_json ../data/superclevr_z_direction/preprocess/SuperCLEVR_vocab.json

python tools/run_test_prob.py \
    --superclevr_question_path ../data/superclevr_z_direction/preprocess/SuperCLEVR_questions.h5 \
    --superclevr_scene_path /home/xingrui/vqa/superclevr-NSVQA/data/superclevr_z_direction/reason/scene_pred-nemo.json \
    --superclevr_vocab_path ../data/superclevr_z_direction/preprocess/SuperCLEVR_vocab.json \
    --superclevr_gt_question_path /home/xingrui/publish/superclevr_3D_questions/output/superclevr_questions_depth.json \
    --save_result_path ../data/superclevr_z_direction/reason/results-superclevr.json \
    --length 100 \
    --dataset superclevr \
    --prob \
