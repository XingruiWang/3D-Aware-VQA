import numpy
import json
from tqdm import tqdm
import random
import argparse


SUPERCLEVR_COLORS =  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SUPERCLEVR_MATERIALS = ['rubber', 'metal']
SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']
SUPERCLEVR_PARTNAMES = ['left_mirror', 'fender_front', 'footrest', 'wheel_front_right', 'crank_arm_left', 'wheel_front_left', 'bumper', 'headlight', 'door_front_left', 'wing', 'front_left_wheel', 'side_stand', 'footrest_left_s', 'tailplane_left', 'wheel_front', 'mirror', 'right_head_light', 'back_left_door', 'left_tail_light', 'head_light_right', 'gas_tank', 'front_bumper', 'tailplane', 'taillight_center', 'back_bumper', 'headlight_right', 'panel', 'front_right_door', 'door_mid_left', 'hood', 'door_left_s', 'front_right_wheel', 'wing_left', 'head_light_left', 'back_right_door', 'tail_light_right', 'seat', 'taillight', 'door_front_right', 'trunk', 'back_left_wheel', 'exhaust_right_s', 'cover', 'brake_system', 'wing_right', 'pedal_left', 'rearlight', 'headlight_left', 'right_tail_light', 'engine_left', 'crank_arm', 'fender_back', 'engine', 'fender', 'door_back_right', 'wheel_back_left', 'back_license_plate', 'cover_front', 'headlight_center', 'engine_right', 'roof', 'left_head_light', 'taillight_right', 'fin', 'saddle', 'mirror_right', 'door', 'bumper_front', 'door_mid_right', 'head_light', 'bumper_back', 'wheel_back_right', 'footrest_right_s', 'drive_chain', 'license_plate_back', 'tail_light', 'pedal', 'windscreen', 'license_plate', 'exhaust_left_s', 'handle_left', 'handle', 'back_right_wheel', 'right_mirror', 'wheel', 'fork', 'taillight_left', 'handle_right', 'front_left_door', 'carrier', 'license_plate_front', 'crank_arm_right', 'wheel_back', 'cover_back', 'propeller', 'exhaust', 'tail_light_left', 'mirror_left', 'pedal_right', 'tailplane_right', 'door_right_s', 'front_license_plate']
SUPERCLEVR_SIZES = ['large', 'small']
SUPERCLEVR_POSES= ['front_pose', 'right_pose', 'back_pose', 'left_pose']

def onehot_index(word, dictionary):
    if word in dictionary:
        onehot = [0]*len(dictionary)
        onehot[dictionary.index(word)] = 1
        return onehot
    else:
        return "The word is not in the dictionary"


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--prediction_path', required=True)
parser.add_argument('--scene_path', required=True)
parser.add_argument('--output_path', required=True)


# pred_bbox_file = '/mnt/data0/xingrui/superclevr_anno/superclevr_pred.json'
# scene_anno_file = '/mnt/data0/xingrui/superclevr_anno/superclevr_anno.json'
args = parser.parse_args()
pred_bbox_file = args.prediction_path
scene_anno_file = args.scene_path

def convert_bbox(bbox, format):
    if format == 'XYWH':
        x1, y1, w, h = bbox
        return {'x1': x1, 'x2': x1 + w, 'y1': y1, 'y2': y1 + h}
    elif format == 'XYXY':
        x1, y1, x2, y2 = bbox
        return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}        
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = convert_bbox(bb1, 'XYXY')
    bb2 = convert_bbox(bb2, 'XYWH')
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

with open(pred_bbox_file) as f:
    pred_bbox = json.load(f)

with open(scene_anno_file) as f:
    scene_anno = json.load(f)
# print(scene_anno.keys())

scene_anno_pred = {}


# for image_id in tqdm(random.sample(scene_anno.keys(), 1)):
for image_id in tqdm(scene_anno):
    if image_id not in pred_bbox: continue
    scene_anno_pred[image_id] = []
    # pred_scene = pred_bbox[image_id]['objects']
    pred_scene = pred_bbox[image_id]

    for o in pred_scene:
        bbox = o['bbox']
        score = o['score']
        class_id = o['class']

        max_iou = 0.2
        max_id = -1
        for i, gt_object in enumerate(scene_anno[image_id]):
            iou = get_iou(bbox, gt_object['bbox'])
            if iou > max_iou:
                max_id = i
                max_iou = iou
        if max_id > 0:
            aligned_object = scene_anno[image_id][max_id]
            scene_anno_pred[image_id].append({'image_filename': aligned_object['image_filename'],
                                                'shape': onehot_index(aligned_object['shape'], SUPERCLEVR_SHAPES+SUPERCLEVR_PARTNAMES),
                                                'material': onehot_index(aligned_object['material'], SUPERCLEVR_MATERIALS),
                                                'size': onehot_index(aligned_object['size'], SUPERCLEVR_SIZES),
                                                'color': onehot_index(aligned_object['color'], SUPERCLEVR_COLORS),   
                                                'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                                                'pred_class': class_id,
                                                'score': score
                                            })
            # scene_anno_pred[image_id].append({'image_filename': aligned_object['image_filename'],
            #                                     'shape': aligned_object['shape'],
            #                                     'color': aligned_object['color'],
            #                                     'material': aligned_object['material'],
            #                                     'size': aligned_object['size'],
            #                                     'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
            #                                     'pred_class': class_id,
            #                                     'score': score
            #                                 })

                                    
    # print('pred', image_id, scene_anno_pred[image_id])

# output_file = '/mnt/data0/xingrui/superclevr_anno/superclevr_aligned.json'
output_file = args.output_path
with open(output_file, 'w') as f:
    json.dump(scene_anno_pred, f)
    print("Writing to file:", '/mnt/data0/xingrui/superclevr_anno/superclevr_aligned.json')


        # print(o['objects'])
#     image_filename = scene['image_filename']
#     objects = scene['objects']
#     obj_mask_box = scene['obj_mask_box']

#     pred_bbox_i = pred_bbox[image_id]

