import json
import os

import BboxTools as bbt
import cv2
import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import subcate_to_cate


class SuperCLEVRTest(Dataset):
    def __init__(self, dataset_path, prefix, category=None, subcategory=None, transform=None, partial = 1.0):
        super().__init__()
        self.img_path = os.path.join(dataset_path, 'images')
        self.scene_path = os.path.join(dataset_path, 'scenes')
        self.question_path = os.path.join(dataset_path, 'questions/superclevr_questions.json')
        self.prefix = prefix
        self.category = category
        self.subcategory = subcategory
        self.transform = transform

        assert (category is None and subcategory is not None) or (subcategory is None and category is not None)
        self.partial = partial
        self.prepare()
        # self.prepare_with_question()
    
    def prepare_with_question(self):
        with open(self.question_path) as f:
            questions = json.load(f)['questions'][-10000:]
            # questions = json.load(f)['questions'][:]
        self.file_list = []
        for q in questions:
            img_name = q['image_filename'].split('.')[0]
            if len(self.file_list) == 0 or self.file_list[-1] != img_name:
                self.file_list.append(img_name)
        self.file_list = sorted(self.file_list)


    def prepare(self):
        self.file_list = sorted([x.split('.')[0] for x in os.listdir(self.img_path) if x.startswith(self.prefix) and x.endswith('.png')])
        if self.partial < 1.0:
            self.file_list = self.file_list[-int(len(self.file_list)*self.partial):]
        self.file_list = self.file_list[-1000:]

    def __getitem__(self, item):
        img_name = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, img_name+'.png'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        with open(os.path.join(self.scene_path, img_name+'.json')) as f:
            scene = json.load(f)
        # mw = np.array(scene['matrix_world'])
        # mw_inv = np.array(scene['matrix_world_inverted'])
        # proj_mat = np.array(scene['projection_matrix'])
        # cam_loc = np.array(scene['camera_location'])

        objects = scene['objects']
        anno = []
        for obj in objects:
            if self.category:
                if subcate_to_cate[obj['shape']] != self.category:
                    continue
            if self.subcategory:
                if obj['shape'] != self.subcategory:
                    continue
            anno.append({
                'location': obj['3d_coords'],
                # 'size_r': obj['size_r'],
                'pixel_coords': obj['pixel_coords'],
                'shape': obj['shape'],
                # 'theta': obj['theta'],
                'color': obj['color'],
                'category': subcate_to_cate[obj['shape']],
                'subcategory': obj['shape']
            })

        sample = {
            'img_name': img_name,
            'img': img,
            # 'mw': mw,
            # 'mw_inv': mw_inv,
            # 'proj_mat': proj_mat,
            # 'cam_loc': cam_loc,
            'objects': anno
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem_by_id__(self, idx):
        item = 0
        for img_name in self.file_list:
            if img_name == f'superCLEVR_new_{idx:06d}':
                break
            item += 1
        # print(item, len(self.file_list), self.file_list)
        img_name = self.file_list[item]
        img = Image.open(os.path.join(self.img_path, img_name+'.png'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        with open(os.path.join(self.scene_path, img_name+'.json')) as f:
            scene = json.load(f)
        # mw = np.array(scene['matrix_world'])
        # mw_inv = np.array(scene['matrix_world_inverted'])
        # proj_mat = np.array(scene['projection_matrix'])
        # cam_loc = np.array(scene['camera_location'])

        objects = scene['objects']
        anno = []
        for obj in objects:
            if self.category:
                if subcate_to_cate[obj['shape']] != self.category:
                    continue
            if self.subcategory:
                if obj['shape'] != self.subcategory:
                    continue
            anno.append({
                'location': obj['3d_coords'],
                # 'size_r': obj['size_r'],
                'pixel_coords': obj['pixel_coords'],
                'shape': obj['shape'],
                # 'theta': obj['theta'],
                'color': obj['color'],
                'category': subcate_to_cate[obj['shape']],
                'subcategory': obj['shape']
            })

        sample = {
            'img_name': img_name,
            'img': img,
            # 'mw': mw,
            # 'mw_inv': mw_inv,
            # 'proj_mat': proj_mat,
            # 'cam_loc': cam_loc,
            'objects': anno
        }

        if self.transform:
            sample = self.transform(sample)
        
        return sample