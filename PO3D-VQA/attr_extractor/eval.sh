BASE=/home/xingrui/vqa/ns-vqa
DATASET=ver_mask
TRAIN_DATASET=ver_texture
TYPE=objects

cd ${BASE}/scene_parse/attr_net


CUDA_VISIBLE_DEVICES=1 python tools/run_eval.py \
    --run_dir /home/xingrui/vqa/ns-vqa/data/${DATASET}/attr_net/outputs/eval_model \
    --obj_ann_path /home/xingrui/vqa/ns-vqa/data/${DATASET}/attr_net/superclevr_anno.json \
    --img_dir /mnt/data0/xingrui/ccvl17/${DATASET}/images \
    --scene_path /mnt/data0/xingrui/ccvl17/${DATASET}/superCLEVR_scenes.json \
    --dataset superclevr \
    --load_path ../../data/${TRAIN_DATASET}/attr_net/outputs/trained_model/objects/checkpoint_best.pt \
    --type objects \
    --multiple_dataset

