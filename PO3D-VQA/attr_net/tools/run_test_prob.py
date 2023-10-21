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
import torch.nn as nn
import json
import cv2
import ipdb
from tqdm import tqdm
from enum import Enum
import numpy as np
import pandas as pd
import math
from scipy.special import softmax


SUPERCLEVR_COLORS =  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SUPERCLEVR_MATERIALS = ['rubber', 'metal']
SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']
SUPERCLEVR_PARTNAMES = ['left_mirror', 'fender_front', 'footrest', 'wheel_front_right', 'crank_arm_left', 'wheel_front_left', 'bumper', 'headlight', 'door_front_left', 'wing', 'front_left_wheel', 'side_stand', 'footrest_left_s', 'tailplane_left', 'wheel_front', 'mirror', 'right_head_light', 'back_left_door', 'left_tail_light', 'head_light_right', 'gas_tank', 'front_bumper', 'tailplane', 'taillight_center', 'back_bumper', 'headlight_right', 'panel', 'front_right_door', 'door_mid_left', 'hood', 'door_left_s', 'front_right_wheel', 'wing_left', 'head_light_left', 'back_right_door', 'tail_light_right', 'seat', 'taillight', 'door_front_right', 'trunk', 'back_left_wheel', 'exhaust_right_s', 'cover', 'brake_system', 'wing_right', 'pedal_left', 'rearlight', 'headlight_left', 'right_tail_light', 'engine_left', 'crank_arm', 'fender_back', 'engine', 'fender', 'door_back_right', 'wheel_back_left', 'back_license_plate', 'cover_front', 'headlight_center', 'engine_right', 'roof', 'left_head_light', 'taillight_right', 'fin', 'saddle', 'mirror_right', 'door', 'bumper_front', 'door_mid_right', 'head_light', 'bumper_back', 'wheel_back_right', 'footrest_right_s', 'drive_chain', 'license_plate_back', 'tail_light', 'pedal', 'windscreen', 'license_plate', 'exhaust_left_s', 'handle_left', 'handle', 'back_right_wheel', 'right_mirror', 'wheel', 'fork', 'taillight_left', 'handle_right', 'front_left_door', 'carrier', 'license_plate_front', 'crank_arm_right', 'wheel_back', 'cover_back', 'propeller', 'exhaust', 'tail_light_left', 'mirror_left', 'pedal_right', 'tailplane_right', 'door_right_s', 'front_license_plate']
SUPERCLEVR_SIZES = ['large', 'small']

with open('/home/xingrui/vqa/super-clevr-gen/question_generation/metadata_part.json') as f:
    metadata = json.load(f)

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


def get_objects_img(img_dir, objects_anno):
    '''
    output list of objects image, transformed to tensor
    '''
    file_name = objects_anno[0]['image_filename']
    file_name = file_name.split('/')[-1]
    image = cv2.imread(os.path.join(img_dir, file_name))
    objects_list = []
    for o in objects_anno:
        object_box = o['bbox']
        box = [int(object_box[0]), int(object_box[1]), int(object_box[2]), int(object_box[3])]
        object_img = image[box[1]:box[3], box[0]: box[2], :]
        
        cv2.imwrite('debug.png', object_img)
        object_img = transform(object_img)
        objects_list.append(object_img)
    return objects_list

def get_parts_img(img_dir, objects_anno):
    '''
    output list of parts image, transformed to tensor
    '''
    file_name = objects_anno[0]['image_filename']
    file_name = file_name.split('/')[-1]
    image = cv2.imread(os.path.join(img_dir, file_name))
    parts_list = []
    obj_index = []
    parts_name = []
    for i, o in enumerate(objects_anno):
        for p, parts in o['parts'].items():
            obj_index.append(i)
            parts_box = parts['bbox']
            x1 = max(0, (parts_box[1]-3))
            y1 = max(0, (parts_box[0]-3))
            x2 = min(image.shape[0], (parts_box[1]+parts_box[3]+3))
            y2 = min(image.shape[1], (parts_box[0] + parts_box[2]+3))
            part_img = image[x1:x2, y1:y2 , :]
            part_img = transform(part_img)
            parts_list.append(part_img)
            parts_name.append(p)
            
    return parts_list, obj_index, parts_name


def get_prob(y, bs):

    shape, color, material, size = y

    color_prob = nn.Softmax(dim=1)(color).cpu().numpy()
    shape_prob= nn.Softmax(dim=1)(shape).cpu().numpy()
    material_prob = nn.Softmax(dim=1)(material).cpu().numpy()
    size_prob = nn.Softmax(dim=1)(size).cpu().numpy()

    results = []
    for i in range(bs):
        results.append({
            'shape': list(shape_prob[i]),
            'color': list(color_prob[i]),
            'material': list(material_prob[i]),
            # 'size': list(size_prob[i])
            'size': [0, 1]
        })
    return results


def direction_probabilities(degree):
    assert 0 <= degree <= 360, "Degree must be between 0 and 360"

    # Normalize the degree to be within [0, 360)
    degree %= 360

    # Define angles for each direction
    back = 0
    right = 90
    front = 180
    left = 270

    # Calculate the absolute differences between the input degree and each direction
    
    diffs = np.abs(np.array([front, right, back, left]) - degree)
    # Calculate the smallest angle differences by considering the circular nature of degrees
    angle_diffs = np.minimum(diffs, 360 - diffs) / 180 * np.pi
    # direction_probs = angle_diffs / 180
    # Negate the angle differences and apply softmax to get the probabilities
    direction_probs = softmax(-angle_diffs.reshape(1, -1)).flatten()
    return direction_probs.tolist()

def post_process(nemo_detection, prob):
    super_class = nemo_detection['class']
    child_classes = metadata['types']['Shape'][super_class]
    shape_prob = prob['shape']

    is_match = False
    for c in child_classes:
        c_idx = SUPERCLEVR_SHAPES.index(c)
        p = shape_prob[c_idx]
        if p > 0.3:
            is_match = True
            break
    if is_match:
        if nemo_detection['class'] == 'bicycle':
            nemo_detection['pose_degree'] += 90
        nemo_detection['pose_degree'] = nemo_detection['pose_degree'] % 360
        nemo_detection['pose'] = direction_probabilities(nemo_detection['pose_degree'])
        return nemo_detection
    else:
        return None

def pred_batch(objects, attr_model, objects_ann, dataset):
    bs = 16
    L = len(objects)
    i = 0
    j = i + bs
    results = []

    image_filename = objects_ann[0]['image_filename']

    # for i, img in enumerate(objects):
    while i < L:
        j = min(j, L)
        objects_batch = torch.cat(objects[i:j], dim = 0)

        with torch.no_grad():
            y_batch = attr_model(objects_batch.cuda())
            
        prob_batch = get_prob(y_batch, j - i) # shape, color, material, size
        
        for ii, prob in enumerate(prob_batch):
            object_result = prob
            nemo_output = post_process(objects_ann[i + ii], prob)

            if nemo_output is None:
                continue

            object_result['image_filename'] = os.path.join(img_dir, image_filename.split('/')[-1])
            object_result['bbox'] = nemo_output['bbox']
            object_result['id'] = i + ii
            object_result['score'] = nemo_output['score']
            object_result['pose_degree'] =nemo_output['pose_degree']
            object_result['pose'] = nemo_output['pose']
            object_result['class'] = objects_ann[i + ii]['class']
            object_result['azimuth'] = objects_ann[i + ii]['azimuth']
            object_result['elevation'] = objects_ann[i + ii]['elevation']
            object_result['theta'] = objects_ann[i + ii]['theta']
            object_result['distance'] = objects_ann[i + ii]['distance']
            object_result['principal'] = objects_ann[i + ii]['principal']

            y = object_result['principal'][1]
            d = object_result['distance']

            if -y / 250 * 45 + 63 > d and object_result['class'] == 'aeroplane':
                object_result['height'] = 0
            else:
                object_result['height'] = 3
            results.append(object_result)
        i += bs
        j += bs
    return results


def pred_batch_parts(parts, attr_model, objects_ann, dataset, obj_idx):
    bs = 16
    L = len(parts)
    i = 0
    j = i + bs
    results = []

    image_filename = objects_ann[0]['image_filename']

    # for i, img in enumerate(objects):
    while i < L:
        j = min(j, L)
        objects_batch = torch.cat(parts[i:j], dim = 0)

        with torch.no_grad():
            y_batch = attr_model(objects_batch.cuda())
            
        prob_batch = get_prob(y_batch, j - i) # shape, color, material, size
        
        for ii, prob in enumerate(prob_batch):
        #     object_result = prob
        #     # nemo_output = post_process(objects_ann[i + ii], prob)

        #     object_result['image_filename'] = os.path.join(img_dir, image_filename.split('/')[-1])
        #     object_result['bbox'] = nemo_output['bbox']
        #     object_result['id'] = i + ii
        #     object_result['score'] = nemo_output['score']
        #     object_result['pose_degree'] =nemo_output['pose_degree']
        #     object_result['pose'] = nemo_output['pose']
        #     object_result['class'] = objects_ann[i + ii]['class']

            results.append(prob)
        i += bs
        j += bs
    return results

def main(pred_bbox, img_dir, attr_model, output_file, dataset='superclevr'):
    attr_model.eval()
    pred_dict = {}
    with open(pred_bbox) as f:
        pred_bbox_dict = json.load(f)

    for i in tqdm(pred_bbox_dict):
        # if int(i) < 31810:
        #     continue
        # file_name, objects{bbox}
        objects_anno = pred_bbox_dict[i]
        print(i, len(pred_bbox_dict[i]))
        # file_name = pred_bbox_dict[i]['file_name']

        objects = get_objects_img(img_dir, objects_anno)

        results = pred_batch(objects, attr_model, objects_anno, dataset)

        pred_dict[i] = results

        # print(results)
    parrent_path = '/'.join(output_file.split("/")[:-1])
    os.makedirs(parrent_path, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(pred_dict, f, indent = 2)
    return pred_dict

def main_parts(pred_bbox, img_dir, attr_model, output_file, dataset='superclevr'):
    attr_model.eval()
    pred_dict = {}
    with open(pred_bbox) as f:
        pred_bbox_dict = json.load(f)

    for i in tqdm(pred_bbox_dict):
        # if int(i) < 31809:
        #     continue
        # if int(i) > 0:
        #     continue
        objects_anno = pred_bbox_dict[i]
        parts, obj_index, parts_name = get_parts_img(img_dir, objects_anno)
        results = pred_batch_parts(parts, attr_model, objects_anno, dataset, obj_index)

        # ith scene
        part_idx = 0
        for obj in pred_bbox_dict[i]:
            for p, parts in obj['parts'].items():
                # print(results[part_idx])
                obj['parts'][p] = {**results[part_idx], **parts}
                part_idx += 1

        # print(results)
    parrent_path = '/'.join(output_file.split("/")[:-1])
    os.makedirs(parrent_path, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(pred_bbox_dict, f, indent = 2)
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
    run_type = opt.type
    

    model = get_model(opt)
    model = model.double().cuda()

    ckpt = torch.load(opt.load_path)
    state_dict = ckpt['model_state']
    model.load_state_dict(state_dict)

    if 'part' in run_type:
        print("Run parts setting")
        main_parts(pred_bbox, img_dir, model, output_file)
    else:
        main(pred_bbox, img_dir, model, output_file)
