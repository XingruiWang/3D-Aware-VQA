import cv2
import numpy as np
import os
import json
import ipdb
from tqdm import tqdm

classes = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike']

colors = np.array([
    (0, 0, 0),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (255, 0, 0),   # red
    (0, 255, 0),   # green
    (0, 0, 255),   # blue
    (255, 255, 0), # yellow
    (255, 0, 255), # magenta
    (0, 255, 255), # cyan
    (255, 128, 0), # orange
    (128, 0, 255), # purple
    (255, 255, 255) # white
], dtype=np.uint8)

def get_overlap(seg1, seg2):
    intersect_area = np.sum(seg1 * seg2)
    area1 = np.sum(seg1)
    area2 = np.sum(seg2)

    if min(area1, area2) < 1:
        return 1.0
    return intersect_area / min(area1, area2)


def filter_object(class_objects, img, vis=True):
    M = 100
    if vis:
        img_name = img.split('/')[-1]
        img = cv2.imread(img)
    all_object=[]
    for c in classes:
        for o in class_objects[c]:
            if o['corr2d_max'] > M and o['score'] > 15 and o['elevation'] > 0.2 and o['elevation'] < 0.95 and np.sum(o['seg']) > 0:
                all_object.append(o)
    is_object = [obj['score_map'] for obj in all_object]
    is_object = np.stack(is_object, axis=2)
    max_score = np.max(is_object, axis = 2)
    mask_all = np.zeros_like(img)

    for obj in all_object:
        obj['seg'] = (max_score == obj['score_map']) * obj['seg'] 

    all_object = [obj for obj in all_object if np.sum(obj['seg'])>500]
    
    for i, o in enumerate(all_object):

        mask = o["seg"][:, :]
        mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)[:, :, None]

        np.random.seed(i)
        color = np.random.randint(0, 256, size=3)
        # Apply the mask to the image
        for c in range(3):
            mask_all[:, :, c] = np.where(mask[:, :, 0] == 1, mask_all[:, :, c] * 0.5 + color[c] * 0.5, mask_all[:, :, c])


        # Find the min and max indices along the x and y axes for the non-zero elements in the mask
        indices = np.where(mask == 1)
        y_indices, x_indices = indices[0], indices[1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Draw the bounding boxes on the original image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    img = img*0.5+np.where(mask_all > 0, mask_all*0.6+10, img*0.5)
    cv2.imwrite(f'output/json/ver_texture_new/detection/{img_name}', img)

    return all_object

def extract_bboxes(mask):

    m = cv2.resize(mask, dsize=( 640, 480))
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]

    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:

        x1, x2= horizontal_indicies[[0, -1]]

        y1, y2  = vertical_indicies[[0, -1]]

        return [int(x1), int(y1), int(x2), int(y2)]
    else:
        return [0, 0, 0, 0]

def main():

    output = 'output/json/ver_texture_new/'
    os.makedirs(output+'detection', exist_ok=True)
    img_dir = '/home/xingrui/publish/superclevr_3D_questions/output/ver_texture_new/images'
    output_dict = {}


    for i in tqdm(range(0, 100)):
        img_name = 'superCLEVR_new_{:06d}'.format(i)
        if not os.path.exists(os.path.join(output+'bicycle', img_name, img_name+"_results.json")):
            print("No predict file")
            continue
        class_objects = {}
        for c in classes:
            class_objects[c] = []
            try:
                with open(os.path.join(output+c, img_name, img_name+"_results.json")) as f:
                    results_dict=json.load(f)
                seg = cv2.imread(os.path.join(output+c, img_name, "segmentation.png"))[:, :, ::-1]
                object_score = cv2.imread(os.path.join(output+c, img_name, "obj_score.png"), cv2.IMREAD_GRAYSCALE)
                object_score = cv2.resize(object_score, dsize=(seg.shape[1], seg.shape[0]), interpolation=cv2.INTER_LINEAR)
            except:
                continue
            # cv2.imwrite('test-mask-all.png'.format(i), seg*50)
            for o, obj in enumerate(results_dict):
                obj['class'] = c
                object_mask = np.zeros((seg.shape[0], seg.shape[1]))
                object_mask[np.where(np.all(seg==colors[o+1], axis=-1))]= 1
                obj['seg'] = object_mask
                obj['score_map'] = object_score * object_mask

                # cv2.imwrite('test-mask{}.jpg'.format(i), object_mask*255)
                class_objects[c].append(obj)
        object_list = filter_object(class_objects, os.path.join(img_dir, img_name+'.png'))

        for o in object_list:
            
            bbox = extract_bboxes(o['seg'])
            # print(np.sum(o['seg']), bbox)
            del o['seg']
            del o['score_map']
            o['bbox'] = bbox

        output_dict[i] = {'objects':object_list, 'file_name':os.path.join(img_dir, img_name+'.png')}
        
    save_dir = '/'.join(output.split('/')[:])
    # os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "superCLEVR_new_prediciton_nemo.json"), 'w') as f:
        json.dump(output_dict, f)
        print("Save to", os.path.join(save_dir, "superCLEVR_new_prediciton_nemo.json"))

if __name__ == '__main__':
    main()


