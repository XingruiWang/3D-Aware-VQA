import math
import os

import BboxTools as bbt
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_anno(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ == 'class':
            out.append(record['objects'][0, 0]['class'][0, idx])
        elif key_ == 'difficult':
            out.append(record['objects'][0, 0]['difficult'][0, idx])
        elif key_ == 'height':
            out.append(record['imgsize'][0, 0][0][1])
        elif key_ == 'width':
            out.append(record['imgsize'][0, 0][0][0])
        elif key_ == 'bbox':
            out.append(record['objects'][0, 0]['bbox'][0, idx][0])
        elif key_ == 'cad_index':
            if len(record['objects'][0, 0]['cad_index'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['cad_index'][0, idx][0, 0])
        elif key_ == 'principal':
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(np.array([None, None]))
            else:
                px = record['objects'][0, 0]['viewpoint'][0, idx]['px'][0, 0][0, 0]
                py = record['objects'][0, 0]['viewpoint'][0, idx]['py'][0, 0][0, 0]
                out.append(np.array([px, py]))
        elif key_ in ['theta', 'azimuth', 'elevation']:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0] * math.pi / 180)
        else:
            if len(record['objects'][0, 0]['viewpoint'][0, idx][0]) == 0:
                out.append(None)
            else:
                out.append(record['objects'][0, 0]['viewpoint'][0, idx][key_][0, 0][0, 0])

    if len(out) == 1:
        return out[0]

    return tuple(out)


class PASCAL3DPTest(Dataset):
    def __init__(self, img_path, anno_path, list_file, category, image_h=320, image_w=448, crop_object=False, transform=None):
        super().__init__()
        self.img_path = img_path
        self.anno_path = anno_path
        self.list_file = list_file
        self.category = category
        self.image_h = image_h
        self.image_w = image_w
        self.crop_object = crop_object
        self.transform = transform

        self.file_list = [l.strip() for l in open(self.list_file).readlines()]
        self.file_list = sorted(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        fname = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, fname+'.JPEG'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        record = sio.loadmat(os.path.join(self.anno_path, fname.split('.')[0]+'.mat'))['record']

        if self.crop_object:
            resize_rate = float(200 * get_anno(record, 'distance') / 1000)
        else:
            resize_rate = float(min(self.image_h / img.shape[0], self.image_w / img.shape[1]))

        bbox = get_anno(record, 'bbox', idx=0)
        box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
        box_ori = box.copy()
        box_ori = box_ori.set_boundary(img.shape[0:2])
        box *= resize_rate

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
        img = cv2.resize(img, dsize=dsize)

        if self.crop_object:
            center = get_anno(record, 'principal', idx=0)
            center = [int(center[1]*resize_rate), int(center[0]*resize_rate)]
        else:
            center = (img.shape[0] // 2, img.shape[1] // 2)
        out_shape = [self.image_h, self.image_w]

        box1 = bbt.box_by_shape(out_shape, center)
        if out_shape[0] // 2 - center[0] > 0 or out_shape[1] // 2 - center[1] > 0 or out_shape[0] // 2 + center[0] - \
                img.shape[0] > 0 or out_shape[1] // 2 + center[1] - img.shape[1] > 0:
            if len(img.shape) == 2:
                padding = (
                (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)))
            else:
                padding = (
                (max(out_shape[0] // 2 - center[0], 0), max(out_shape[0] // 2 + center[0] - img.shape[0], 0)),
                (max(out_shape[1] // 2 - center[1], 0), max(out_shape[1] // 2 + center[1] - img.shape[1], 0)),
                (0, 0))
            img = np.pad(img, padding, mode='constant')
            box = box.shift([padding[0][0], padding[1][0]])
            box1 = box1.shift([padding[0][0], padding[1][0]])
        img_box = box1.set_boundary(img.shape[0:2])
        box_in_cropped = img_box.box_in_box(box)

        img_cropped = img_box.apply(img)
        proj_foo = bbt.projection_function_by_boxes(box_ori, box_in_cropped, compose=False)

        principal = get_anno(record, 'principal', idx=0)
        principal[0] = proj_foo[1](principal[0])
        principal[1] = proj_foo[0](principal[1])

        azimuth = get_anno(record, 'azimuth', idx=0)
        elevation = get_anno(record, 'elevation', idx=0)
        theta = get_anno(record, 'theta', idx=0)
        distance_orig = get_anno(record, 'distance', idx=0)
        distance = distance_orig / resize_rate
        fine_cad_idx = get_anno(record, 'cad_index', idx=0)

        (y1, y2), (x1, x2) = box_in_cropped.bbox
        sample = {
            'img': img_cropped,
            'img_name': fname,
            'azimuth': azimuth,
            'elevation': elevation,
            'theta': theta,
            'distance_orig': distance_orig,
            'distance': distance,
            'fine_cad_idx': fine_cad_idx,
            'resize_rate': resize_rate,
            'principal': principal,
            'bbox': [x1, y1, x2, y2]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def debug(self, item, fname='debug.png'):
        sample = self.__getitem__(item)
        print(sample['img_name'])
        img = sample['img']
        p1, p2 = sample['principal']
        img = cv2.circle(img, (int(p1), int(p2)), 2, (0, 255, 0), -1)
        [x1, y1, x2, y2] = sample['bbox']
        img = cv2.line(img, (int(x1), int(y1)), (int(x1), int(y2)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x2), int(y2)), (int(x1), int(y2)), (0, 255, 0), 2)
        img = cv2.line(img, (int(x2), int(y2)), (int(x2), int(y1)), (0, 255, 0), 2)
        Image.fromarray(img).save(fname)
