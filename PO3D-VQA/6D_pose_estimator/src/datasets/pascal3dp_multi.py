import math
import os

import BboxTools as bbt
import cv2
import numpy as np
import os
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def get_num_objs(record):
    objects = record['objects'][0, 0]
    return len(objects['bbox'][0])


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


class PASCAL3DPMulti(Dataset):
    def __init__(self, img_path, anno_path, list_file, image_h, image_w, ext='.JPEG', transform=None, category=None,
                 enable_cache=False):
        self.img_path = img_path
        self.anno_path = anno_path
        self.list_file = list_file
        self.image_h = image_h
        self.image_w = image_w
        self.ext = ext
        self.transform = transform
        self.category = category
        self.enable_cache = enable_cache
        self.cache_img = dict()
        self.cache_record = dict()

        self.file_list = [l.strip() for l in open(list_file).readlines()]
        self.file_list = [x.split(' ')[0] for x in self.file_list]
        self.file_list = [x.split('.')[0] for x in self.file_list]

        img_files = os.listdir(img_path)
        self.file_list = sorted([x for x in self.file_list if x + self.ext in img_files])

    def __getitem__(self, item):
        name_img = self.file_list[item]
        if name_img in self.cache_img:
            img = self.cache_img[name_img]
            record = self.cache_record[name_img]
        else:
            img = Image.open(os.path.join(self.img_path, name_img + self.ext))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img)
            record = sio.loadmat(os.path.join(self.anno_path, name_img + '.mat'))['record']
            if self.enable_cache:
                self.cache_img[name_img] = img
                self.cache_record[name_img] = record

        resize_rate = float(min(self.image_h / img.shape[0], self.image_w / img.shape[1]))

        num_objs = get_num_objs(record)
        box_list = []
        box_ori_list = []
        target_idxs = []
        for i in range(num_objs):
            try:
                if self.category is not None and self.category != get_anno(record, 'class', idx=i):
                    continue
                if get_anno(record, 'distance', idx=i) is None:
                    continue
                bbox = get_anno(record, 'bbox', idx=i)
                box = bbt.from_numpy(bbox, sorts=('x0', 'y0', 'x1', 'y1'))
                box_ori = box.copy()
                box_ori = box_ori.set_boundary(img.shape[0:2])
                box *= resize_rate
                box_list.append(box)
                box_ori_list.append(box_ori)
                target_idxs.append(i)
            except:
                continue

        dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
        img = cv2.resize(img, dsize=dsize)

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
            for i in range(len(box_list)):
                box_list[i] = box_list[i].shift([padding[0][0], padding[1][0]])
            box1 = box1.shift([padding[0][0], padding[1][0]])
        img_box = box1.set_boundary(img.shape[0:2])
        box_in_cropped_list = [img_box.box_in_box(b) for b in box_list]

        img_cropped = img_box.apply(img)
        proj_foo_list = [bbt.projection_function_by_boxes(b_ori, b_in_cropped, compose=False) for b_ori, b_in_cropped in
                         zip(box_ori_list, box_in_cropped_list)]

        principal_list = []
        for i, idx in enumerate(target_idxs):
            principal = get_anno(record, 'principal', idx=idx)
            if principal[0] is not None:
                principal[0] = proj_foo_list[i][1](principal[0])
                principal[1] = proj_foo_list[i][0](principal[1])
            principal_list.append(principal)

        difficult_list = np.array([get_anno(record, 'difficult', idx=i) for i in target_idxs])
        category_list = np.array([get_anno(record, 'class', idx=i) for i in target_idxs])
        azimuth_list = np.array([get_anno(record, 'azimuth', idx=i) for i in target_idxs])
        elevation_list = np.array([get_anno(record, 'elevation', idx=i) for i in target_idxs])
        theta_list = np.array([get_anno(record, 'theta', idx=i) for i in target_idxs])
        distance_orig_list = np.array([get_anno(record, 'distance', idx=i) for i in target_idxs])
        distance_list = np.array([d if d is None else d / resize_rate for d in distance_orig_list])
        fine_cad_idx_list = np.array([get_anno(record, 'cad_index', idx=i) for i in target_idxs])

        sample = {
            'img': img_cropped,
            'img_name': name_img,
            'azimuth': azimuth_list,
            'elevation': elevation_list,
            'theta': theta_list,
            'distance_orig': distance_orig_list,
            'distance': distance_list,
            'fine_cad_idx': fine_cad_idx_list,
            'resize_rate': resize_rate,
            'principal': principal_list,
            'bbox': [b.bbox for b in box_in_cropped_list],
            'num_objs': num_objs,
            'category': category_list,
            'difficult': difficult_list
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.file_list)

    def get_image_size(self):
        name_img = self.file_list[0]
        img = Image.open(os.path.join(self.img_path, name_img))
        return np.array(img).shape[0:2]

    def debug(self, item):
        sample = self.__getitem__(item)

        print(sample['img_name'])
        img = sample['img']
        principal = sample['principal']
        bbox = sample['bbox']
        print(principal)
        print(bbox)

        # for i in range(sample['num_objs']):
        for i in range(len(sample['principal'])):
            p = principal[i]
            if p[0] is not None:
                img = cv2.circle(img, (int(p[0]), int(p[1])), 2, color=(255, 0, 0), thickness=2)
            (y1, y2), (x1, x2) = bbox[i]
            img = cv2.line(img, (int(x1), int(y1)), (int(x1), int(y2)), (0, 255, 0), 2)
            img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y1)), (0, 255, 0), 2)
            img = cv2.line(img, (int(x2), int(y2)), (int(x1), int(y2)), (0, 255, 0), 2)
            img = cv2.line(img, (int(x2), int(y2)), (int(x2), int(y1)), (0, 255, 0), 2)

        Image.fromarray(img).save(f'debug.png')
