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
    if vis:
        img_name = img.split('/')[-1]
        img = cv2.imread(img)
    unique_object=[]
    for c in classes:
        if c == 'aeroplane':
            for o in class_objects[c]:
                if o['corr2d_max'] > 170 and o['score'] > 10 and o['elevation'] > 0.3 and o['elevation'] < 0.95 and np.sum(o['seg']) > 0:
                    unique_object.append(o)
                # vis
                # mask = np.resize(o['seg'], dsize=(img.shape[0], img.shape[1]))[:, :, None]
                # cv2.imwrite('test.jpg', img*mask)
                # print(o['score'], o['corr2d_max'], o['keypoint_max'])
        else:
            for o in class_objects[c]:
                # print(c)
                if o['corr2d_max'] > 170 and o['score'] > 10 and o['elevation'] > 0.3 and o['elevation'] < 0.95 and np.sum(o['seg']) > 0:
                    delete_idx = []
                    has_overlap = False
                    for i in range(len(unique_object)):
                        overlap = get_overlap(o['seg'], unique_object[i]['seg'])
                        if overlap > 0.75:
                            has_overlap = True
                            # if unique_object[i]['score'] < o['score']:
                            if (unique_object[i]['score'] + 30) < o['score'] and unique_object[i]['score'] < 50:
                                delete_idx.append(i)
                    delect_offset = 0
                    for i in delete_idx:
                        del unique_object[i - delect_offset]
                        delect_offset += 1

                    if not has_overlap or len(delete_idx) > 0 or unique_object[i]['score'] > 50:

                        unique_object.append(o)
                    # vis
                    # print(o['class'] )
                    # mask = cv2.resize(o['seg'], dsize=( img.shape[1], img.shape[0]))
                    # cv2.imwrite('test.jpg', img*mask)
                    # print(o['score'], o['corr2d_max'], o['keypoint_ratio'], o['keypoint_avg'])
                    # ipdb.set_trace()
    #vis

    for o in unique_object:
        # print(o['class'])
        # print(o['score'], o['corr2d_max'], o['keypoint_ratio'], o['keypoint_avg'])
        # mask_all = mask_all * 0.9 + o['seg']* 0.9 - o['seg'] * mask_all * 0.8
        mask = o["seg"]
        name = o["class"]
        mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)[:, :, None]

        # Find the min and max indices along the x and y axes for the non-zero elements in the mask
        indices = np.where(mask == 1)
        y_indices, x_indices = indices[0], indices[1]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # Draw the bounding boxes on the original image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(img, name, (x_min, y_min - 5), font, font_scale, (0, 255, 0), font_thickness)

    cv2.imwrite(f'/home/xingrui/vqa/nemo_superclevr_copy/output/json/0405/detection_2/{img_name}', img)

    return unique_object

def extract_bboxes(mask):

    m = cv2.resize(mask, dsize=( 640, 480))

    # m = mask[:, :, 0]
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]

    vertical_indicies = np.where(np.any(m, axis=1))[0]
    # print(horizontal_indicies, vertical_indicies)
    if horizontal_indicies.shape[0]:

        x1, x2= horizontal_indicies[[0, -1]]

        y1, y2  = vertical_indicies[[0, -1]]

        return [int(x1), int(y1), int(x2), int(y2)]
    else:
        return [0, 0, 0, 0]

def main():
    # output = 'output/json/ver_mask/ver_mask_'
    output = '/home/xingrui/vqa/nemo_superclevr_copy/output/json/0405/'
    img_dir = '/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/images'
    output_dict = {}

    for i in tqdm(range(25310, 32810)):
    # for i in tqdm(range(31816, 31817)):
        img_name = 'superCLEVR_new_{:06d}'.format(i)
        if not os.path.exists(os.path.join(output+'bicycle', img_name, img_name+"_results.json")):
            continue
        class_objects = {}
        for c in classes:
            class_objects[c] = []
            try:
                with open(os.path.join(output+c, img_name, img_name+"_results.json")) as f:
                    results_dict=json.load(f)
                seg = cv2.imread(os.path.join(output+c, img_name, "segmentation.png"))[:, :, ::-1]
            except:
                continue
            # cv2.imwrite('test-mask-all.png'.format(i), seg*50)
            for o, obj in enumerate(results_dict):
                obj['class'] = c
                object_mask = np.zeros((seg.shape[0], seg.shape[1]))
                object_mask[np.where(np.all(seg==colors[o+1], axis=-1))]= 1
                obj['seg'] = object_mask
                # cv2.imwrite('test-mask{}.jpg'.format(i), object_mask*255)
                class_objects[c].append(obj)
        object_list = filter_object(class_objects, os.path.join(img_dir, img_name+'.png'))

        for o in object_list:
            
            bbox = extract_bboxes(o['seg'])
            # print(np.sum(o['seg']), bbox)
            del o['seg']
            o['bbox'] = bbox

        output_dict[i] = {'objects':object_list, 'file_name':os.path.join(img_dir, img_name+'.png')}
        
    save_dir = '/'.join(output.split('/')[:-1])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "superCLEVR_new_prediciton_nemo_all.json"), 'w') as f:
        json.dump(output_dict, f)
        print("Save to", os.path.join(save_dir, "superCLEVR_new_prediciton_nemo_all.json"))

if __name__ == '__main__':
    main()


