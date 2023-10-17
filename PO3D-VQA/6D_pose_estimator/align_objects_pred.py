import cv2
import numpy as np
import os
import json
import ipdb
from tqdm import tqdm
import ipdb
from math import radians, degrees
from scipy.special import softmax

SUPERCLEVR_COLORS =  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SUPERCLEVR_MATERIALS = ['rubber', 'metal']
SUPERCLEVR_SIZES = ['large', 'small']
SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']

def one_hot(value, categories):
    index = categories.index(value)
    one_hot_vector = np.zeros(len(categories), dtype=int)
    one_hot_vector[index] = 1
    return one_hot_vector.tolist()

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

# def process_pose(rotation):
#     if isinstance(rotation, str):
#         return rotation
#     if rotation >= 0 and rotation < 45:
#         pose = 'back_pose'            
#     elif rotation > 45 and rotation <= 135:
#         pose = 'right_pose' 
#     elif rotation > 135 and rotation <= 225:
#         pose = 'front_pose'
#     elif rotation > 225 and rotation <= 315:
#         pose = 'left_pose'        
#     elif rotation > 315:
#         pose = 'back_pose'   
#     else:
#         pose = 'None'       
#     return pose

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

def main():

    output_dict = {}
    # scene_dir = '/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/scenes'
    # prediction_path = "/home/xingrui/vqa/nemo_superclevr_copy/output/json/0405/superCLEVR_new_prediciton_nemo_all.json"
    # prediction_path = "/home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/superCLEVR_new_prediciton_nemo_all_0424.json"
    prediction_path = "/home/xingrui/vqa/nemo_superclevr_copy/output/json/z_direction/superCLEVR_new_prediciton_nemo.json"

    with open(prediction_path) as f:
        prediction = json.load(f)

    # for i in tqdm(range(25310, 32809)):
    for i in tqdm(range(0, 100)):
        if str(i) not in prediction:
            continue

        output_dict[str(i)] = []


        prediction_scene = prediction[str(i)]['objects'] # dict of prediction, have bbox, class. score
        img_file = prediction[str(i)]['file_name']

        for object_id, o in enumerate(prediction_scene):

            output_object = {
                "image_filename": img_file.split("/")[-1],
                "id": object_id,
            }
            output_object = {**output_object, **o}

            output_object['pose_degree'] = np.degrees(output_object['azimuth'])
            
            # if output_object['class'] == 'bicycle':
            #     output_object['pose_degree'] += 90

            # output_object['pose_degree'] = output_object['pose_degree'] % 360
            # output_object['pose'] = direction_probabilities(output_object['pose_degree'])

            output_dict[str(i)].append(output_object)
    # with open("/home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/pred_prob.json", "w") as f:
    with open("/home/xingrui/vqa/nemo_superclevr_copy/output/json/z_direction/pred_prob.json", "w") as f:
        json.dump(output_dict, f)    

if __name__ == '__main__':
    main()


