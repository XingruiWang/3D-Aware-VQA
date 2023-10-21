type=object

python tools/run_train.py \
    --run_dir ../data/attr_net/outputs/trained_model/${type} \
    --obj_ann_path /home/xingrui/vqa/nemo_superclevr_copy/output/json/0405/anno_prob.json \
    --img_dir /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/images \
    --scene_path /home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/superCLEVR_scenes_210k_occlusion.json \
    --dataset superclevr \
    --type ${type} \
    --display_every 1 \
    