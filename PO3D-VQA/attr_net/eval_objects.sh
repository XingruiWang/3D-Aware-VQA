CUDA_VISIBLE_DEVICES=3 python tools/run_eval.py \
    --run_dir ../../data/attr_net/outputs/eval_model/object \
    --obj_ann_path /mnt/data0/xingrui/superclevr_anno/superclevr_aligned.json \
    --img_dir /home/xingrui/vqa/NSCL_super/data/superclevr/200k/images \
    --scene_path /home/xingrui/vqa/NSCL_super/data/superclevr/200k/scenes.json \
    --dataset superclevr \
    --type object \
    --load_path ../../data/attr_net/outputs/trained_model/object/checkpoint_best.pt