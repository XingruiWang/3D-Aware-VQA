DATASET=ver_mask

CUDA_VISIBLE_DEVICES=2 python tools/run_test.py \
    --load_path ../../data/attr_net/outputs/trained_model/object/checkpoint_best.pt \
    --img_dir /mnt/data0/xingrui/ccvl17/${DATASET}/images \
    --pred_bbox /home/xingrui/vqa/ns-vqa/data/${DATASET}/detection/objects/superclevr_objects_test.json
    --output_file ../../data/${DATASET}/reason/scene_pred.json