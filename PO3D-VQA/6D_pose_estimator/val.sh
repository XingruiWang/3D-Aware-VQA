
class=motorbike

# CUDA_VISIBLE_DEVICES=0 python3 scripts/inference_multi.py \
#     --category ${class} --test_index 1 \
#     --ckpt /home/xingrui/vqa/pretrain_6d_pose/output/experiments/0407_superclevr_${class}/ckpts/saved_model_15000.pth \
#     --dataset_path /home/xingrui/publish/superclevr_3D_questions/output/only_plane_in_sky_2 \
#     --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
#     --px_sample 31 \
#     --py_sample 31 \
#     --save_results output/json/z_direction/${class}

CUDA_VISIBLE_DEVICES=1 python3 scripts/inference_single.py \
    --category ${class} --test_index 31880 \
    --ckpt /home/xingrui/vqa/pretrain_6d_pose/output/experiments/0407_superclevr_${class}/ckpts/saved_model_15000.pth \
    --dataset_path /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new \
    --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
    --px_sample 31 \
    --py_sample 31 \
    --down_sample_rate 8 \
    --save_results output/json/debug/${class}


# the paper version
# CUDA_VISIBLE_DEVICES=1 python3 scripts/inference_multi.py \
#     --category ${class} --test_index 1 \
#     --ckpt /home/xingrui/vqa/pretrain_6d_pose/output/experiments/0407_superclevr_${class}/ckpts/saved_model_15000.pth \
#     --dataset_path /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new \
#     --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
#     --px_sample 31 \
#     --py_sample 31 \
#     --save_results output/json/0511/${class}


# CUDA_VISIBLE_DEVICES=1 python3 scripts/inference_single.py \
#     --category car --test_index 1 \
#     --ckpt experiments/oct09_superclevr_car/ckpts/saved_model_15000.pth \
#     --dataset_path /mnt/data0/xingrui/ccvl17/ver_texture\
#     --mesh_path ~/vqa/nemo_superclevr_copy/CAD_cate \
#     --px_sample 31 \
#     --py_sample 31 \
#     --save_results output
