CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py \
    --exp_name oct07_superclevr_bicycle \
    --category bicycle \
    --iterations 15000 \
    --save_itr 3000 \
    --update_lr_itr 10000 \
    --batch_size 12 \
#     > train0.txt 2>&1 &

# exit

# CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py \
#     --exp_name oct09_superclevr_car \
#     --category car \
#     --iterations 15000 \
#     --save_itr 3000 \
#     --update_lr_itr 10000 \
#     --batch_size 12 \
#     > train0.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py \
#     --exp_name oct09_superclevr_aeroplane \
#     --category aeroplane \
#     --iterations 15000 \
#     --save_itr 3000 \
#     --update_lr_itr 10000 \
#     --batch_size 12 \
#     > train1.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python3 scripts/train.py \
#     --exp_name oct09_superclevr_bus \
#     --category bus \
#     --iterations 15000 \
#     --save_itr 3000 \
#     --update_lr_itr 10000 \
#     --batch_size 12 \
#     > train2.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python3 scripts/train.py \
#     --exp_name oct09_superclevr_motorbike \
#     --category motorbike \
#     --iterations 15000 \
#     --save_itr 3000 \
#     --update_lr_itr 10000 \
#     --batch_size 12 \
#     > train3.txt 2>&1 &

# exit
