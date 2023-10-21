'''
@Xingrui
Run test 
'''
import os
import sys
sys.path.append('./')

from options import get_options
from datasets import get_dataloader
from model import get_model
from trainer import get_trainer
import torch
import json
import cv2
import ipdb
from tqdm import tqdm
from enum import Enum
import numpy as np
import pandas as pd
import copy
import pycocotools.mask as mask_util
from scipy.special import softmax


SUPERCLEVR_COLORS =  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SUPERCLEVR_MATERIALS = ['rubber', 'metal']
SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']
SUPERCLEVR_PARTNAMES = ['left_mirror', 'fender_front', 'footrest', 'wheel_front_right', 'crank_arm_left', 'wheel_front_left', 'bumper', 'headlight', 'door_front_left', 'wing', 'front_left_wheel', 'side_stand', 'footrest_left_s', 'tailplane_left', 'wheel_front', 'mirror', 'right_head_light', 'back_left_door', 'left_tail_light', 'head_light_right', 'gas_tank', 'front_bumper', 'tailplane', 'taillight_center', 'back_bumper', 'headlight_right', 'panel', 'front_right_door', 'door_mid_left', 'hood', 'door_left_s', 'front_right_wheel', 'wing_left', 'head_light_left', 'back_right_door', 'tail_light_right', 'seat', 'taillight', 'door_front_right', 'trunk', 'back_left_wheel', 'exhaust_right_s', 'cover', 'brake_system', 'wing_right', 'pedal_left', 'rearlight', 'headlight_left', 'right_tail_light', 'engine_left', 'crank_arm', 'fender_back', 'engine', 'fender', 'door_back_right', 'wheel_back_left', 'back_license_plate', 'cover_front', 'headlight_center', 'engine_right', 'roof', 'left_head_light', 'taillight_right', 'fin', 'saddle', 'mirror_right', 'door', 'bumper_front', 'door_mid_right', 'head_light', 'bumper_back', 'wheel_back_right', 'footrest_right_s', 'drive_chain', 'license_plate_back', 'tail_light', 'pedal', 'windscreen', 'license_plate', 'exhaust_left_s', 'handle_left', 'handle', 'back_right_wheel', 'right_mirror', 'wheel', 'fork', 'taillight_left', 'handle_right', 'front_left_door', 'carrier', 'license_plate_front', 'crank_arm_right', 'wheel_back', 'cover_back', 'propeller', 'exhaust', 'tail_light_left', 'mirror_left', 'pedal_right', 'tailplane_right', 'door_right_s', 'front_license_plate']
SUPERCLEVR_SIZES = ['large', 'small']

CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']

def transform(img):
    img = img.astype(np.float64)
    img = cv2.resize(img, (224, 224),
                    interpolation = cv2.INTER_CUBIC)
    img = img[:, :, ::-1].copy()

    img /= 255.0
    img -= [0.485, 0.456, 0.406]
    img /= [0.229, 0.224, 0.225]

    img = img.transpose((2, 0, 1))[None, :, :, :]
    return torch.from_numpy(img)


def get_objects_img(file_name, img_dir, objects_anno):
    '''
    output list of objects image, transformed to tensor
    '''
    file_name = file_name.split('/')[-1]
    image = cv2.imread(os.path.join(img_dir, file_name))
    objects_list = []
    for o in objects_anno:
        object_box = o['bbox']
        box = [int(object_box[0]), int(object_box[1]), int(object_box[2]), int(object_box[3])]
        object_img = image[box[1]:box[3], box[0]: box[2], :]
        object_img = transform(object_img)
        objects_list.append(object_img)
    return objects_list

def get_objects_img_seg(file_name, img_dir, objects_anno):
    '''
    output list of objects image, transformed to tensor
    '''
    file_name = file_name.split('/')[-1]
    image = cv2.imread(os.path.join(img_dir, file_name))
    image = transform(image)
    objects_list = []
    for o in objects_anno:
        object_mask = copy.deepcopy(o['mask'])
        object_mask['counts'] = eval(object_mask['counts'])
        object_mask = np.array(mask_util.decode(object_mask), dtype=np.float32)
        object_mask = cv2.resize(object_mask, (224, 224),
                        interpolation = cv2.INTER_NEAREST)[:, :, None]
        object_mask = torch.from_numpy(object_mask.transpose((2, 0, 1)))

        object_img = torch.cat([image, object_mask.unsqueeze(0)], dim=1)

        objects_list.append(object_img)
    return objects_list

def get_label(y, bs, dataset):
    shape, color, material, size = y
    color_id = torch.argmax(color.data, dim=1)
    shape_id = torch.argmax(shape.data, dim=1)
    material_id = torch.argmax(material.data, dim=1)
    size_id = torch.argmax(size.data, dim=1)

    results = []
    if dataset == 'superclevr':
        for i in range(bs):
            results.append({
                'shape': SUPERCLEVR_SHAPES[shape_id[i]],
                'color': SUPERCLEVR_COLORS[color_id[i]],
                'material': SUPERCLEVR_MATERIALS[material_id[i]],
                'size': SUPERCLEVR_SIZES[size_id[i]]
            })
    elif dataset == 'clevr':
        for i in range(bs):
            results.append({
                'shape': CLEVR_SHAPES[shape_id[i]],
                'color': CLEVR_COLORS[color_id[i]],
                'material': CLEVR_MATERIALS[material_id[i]],
                'size': CLEVR_SIZES[size_id[i]]
            })            
    else:
        print("error")
    return results

def pred_batch(objects, attr_model, file_name, objects_ann, dataset):
    bs = 4
    L = len(objects)
    i = 0
    j = i + bs
    results = []
    # for i, img in enumerate(objects):
    while i < L:
        j = min(j, L)
        objects_batch = torch.cat(objects[i:j], dim = 0)

        y_batch = attr_model(objects_batch.cuda())
        label_batch = get_label(y_batch, j - i, dataset) # shape, color, material, size
        
        for ii, label in enumerate(label_batch):
            object_result = label
            object_result['image_filename'] = os.path.join(img_dir, file_name.split('/')[-1])
            object_result['bbox'] = objects_ann[i + ii]['bbox']
            # object_result['mask'] = objects_ann[i + ii]['mask']
            object_result['id'] = i + ii
            object_result['score'] = objects_ann[i + ii]['score']
            object_result['pose_degree'] = objects_ann[i + ii]['pose_degree']
            object_result['class'] = objects_ann[i + ii]['class']
            results.append(object_result)
        i += bs
        j += bs
    return results


def main(pred_bbox, img_dir, attr_model, output_file, dataset='superclevr'):
    attr_model.eval()
    pred_dict = {}
    with open(pred_bbox) as f:
        pred_bbox_dict = json.load(f)

    for i in tqdm(pred_bbox_dict):

        # file_name, objects{bbox}
        objects_anno = pred_bbox_dict[i]
        file_name = objects_anno[0]['image_filename']

        # objects = get_objects_img_seg(file_name, img_dir, objects_anno)
        objects = get_objects_img(file_name, img_dir, objects_anno)

        results = pred_batch(objects, attr_model, file_name, objects_anno, dataset)

        pred_dict[i] = results
        print(file_name)
        import ipdb
        ipdb.set_trace()
        # print(results)

    with open(output_file, 'w') as f:
        json.dump(pred_dict, f, indent = 2)
    return pred_dict



if __name__ == '__main__':
    # pred_bbox='/mnt/data0/xingrui/superclevr_anno/superclevr_pred.json'
    # pred_bbox = '/home/xingrui/vqa/ns-vqa/data/ver_mask/detection/objects/superclevr_objects_test.json'

    # pred_bbox='/mnt/data0/xingrui/superclevr_anno/superclevr_pred_sample.json'
    opt = get_options('test')
    # img_dir = '/mnt/data0/xingrui/ccvl17/ver_mask/images/'
    img_dir = opt.img_dir
    # output_file = '../../data/ver_mask/reason/scene_pred.json'
    output_file = opt.output_file
    pred_bbox = opt.pred_bbox
    

    model = get_model(opt)
    model = model.double().cuda()

    ckpt = torch.load(opt.load_path)
    state_dict = ckpt['model_state']
    model.load_state_dict(state_dict)

    main(pred_bbox, img_dir, model, output_file, opt.dataset)
