import BboxTools as bbt
import cv2
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

colors = np.array([(205, 92, 92), (255, 160, 122), (255, 0, 0), (255, 192, 203),
                   (255, 105, 180), (255, 20, 147), (255, 69, 0), (255, 165, 0),
                   (255, 215, 0), (255, 255, 0), (255, 218, 185), (238, 232, 170),
                   (189, 183, 107), (230, 230, 250), (216, 191, 216), (238, 130, 238),
                   (255, 0, 255), (102, 51, 153), (75, 0, 130), (123, 104, 238),
                   (127, 255, 0), (50, 205, 50), (0, 250, 154), (60, 179, 113),
                   (154, 205, 50), (102, 205, 170), (32, 178, 170), (0, 255, 255),
                   (175, 238, 238), (127, 255, 212), (70, 130, 180), (176, 196, 222),
                   (135, 206, 250), (30, 144, 255)], dtype=np.uint8)


class SuperCLEVRTrain(Dataset):
    def __init__(self, img_path, anno_path, prefix, mask_path=None, category=None, subcategory=None, transform=None, enable_cache=True, max_num_obj=8, partial=1.0):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.anno_path = anno_path
        self.prefix = prefix
        self.category = category
        self.subcategory = subcategory
        self.transform = transform
        self.enable_cache = enable_cache
        self.max_num_obj = max_num_obj
        self.partial = partial

        assert (category is None and subcategory is not None) or (category is not None and subcategory is None)

        self.cache_img = dict()
        self.cache_anno = dict()

        self.prepare()
    
    def prepare(self):
        file_list = sorted([x.split('.')[0] for x in os.listdir(self.img_path) if x.startswith(self.prefix) and x.endswith('.png')])
        if self.partial < 1.0:
            file_list = file_list[:int(len(file_list)*self.partial)]
        file_list_cate = []
        for f in tqdm(file_list):
            anno = dict(np.load(os.path.join(self.anno_path, f+'.npz'), allow_pickle=True))
            if self.category:
                cate_list = [obj['category'] for obj in anno['objects']]
                if self.category in cate_list:
                    file_list_cate.append(f)
            if self.subcategory:
                cate_list = [obj['sub_category'] for obj in anno['objects']]
                if self.subcategory in cate_list:
                    file_list_cate.append(f)
        self.file_list = file_list_cate
    
    def __getitem__(self, item):
        img_name = self.file_list[item]

        if img_name in self.cache_img:
            img = self.cache_img[img_name]
            anno = self.cache_anno[img_name]
        else:
            img = Image.open(os.path.join(self.img_path, img_name+'.png'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            anno = dict(np.load(os.path.join(self.anno_path, img_name+'.npz'), allow_pickle=True))
            if self.enable_cache:
                self.cache_img[img_name] = img
                self.cache_anno[img_name] = anno
        
        if self.mask_path:
            m = np.load(os.path.join(self.mask_path, img_name+'.npy'))
        
        w, h = img.size
        
        objects = anno['objects']

        """
        all_kp, all_kpvis, obj_mask, all_distance = None, None, np.zeros((h, w), dtype=np.uint8), np.zeros((self.max_num_obj,), dtype=np.float32)
        count_obj = 0
        for obj in objects:
            if obj['category'] != self.category:
                continue
            if all_kp is None:
                all_kp, all_kpvis = np.zeros((self.max_num_obj, len(obj['kp']), 2), dtype=np.int32), np.zeros((self.max_num_obj, len(obj['kp'])), dtype=np.uint8)
            all_kp[count_obj] = obj['kp']
            all_kpvis[count_obj] = obj['kpvis']
            obj_mask = obj_mask | obj['obj_mask']
            all_distance[count_obj] = obj['distance']
            count_obj += 1
        """

        if self.category:
            objects = [obj for obj in objects if obj['category'] == self.category]
        if self.subcategory:
            objects = [obj for obj in objects if obj['sub_category'] == self.subcategory]
        obj = objects[np.random.randint(0, len(objects))]
        all_kp = obj['kp']
        all_kpvis = obj['kpvis']
        all_distance = obj['distance']
        count_obj = 1

        obj_mask = np.zeros((h, w), dtype=np.uint8)
        for obj in objects:
            if self.category:
                if obj['category'] != self.category:
                    continue
            if self.subcategory:
                if obj['sub_category'] != self.subcategory:
                    continue
            obj_mask = obj_mask | obj['obj_mask']
        
        sample = {'img_name': img_name, 'img': img, 'kp': all_kp,
                  'kpvis': all_kpvis, 'obj_mask': obj_mask,
                  'distance': all_distance, 'num_objs': count_obj}
        
        if self.mask_path:
            sample['mask_names'] = anno['all_mask_names']
            sample['masks'] = m

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.file_list)
    
    def debug(self, item):
        transform = self.transform
        self.transform = None

        sample = self.__getitem__(item)
        
        print('kp', sample['kp'].shape)
        print('kpvis', sample['kpvis'].shape)
        print('obj_mask', sample['obj_mask'].shape)
        print('distance', sample['distance'].shape, sample['distance'])
        print('num_objs', sample['num_objs'])
        if 'mask_names' in sample and 'masks' in sample:
            print('mask_names', len(sample['mask_names']))
            print('masks', sample['masks'].shape)

        if 'masks' in sample:
            car_cates = ['truck', 'suv', 'minivan', 'sedan', 'wagon']
            car_parts = ["back_bumper", "back_left_door", "back_left_wheel", "back_left_window", "back_right_door", "back_right_wheel", "back_right_window", "back_windshield", "front_bumper", "front_left_door", "front_left_wheel", "front_left_window", "front_right_door", "front_right_wheel", "front_right_window", "front_windshield", "hood", "roof", "trunk"]
            mask_img = np.zeros((sample['masks'].shape[1], sample['masks'].shape[2], 3), dtype=np.uint8)
            count_color = 0
            for i, mask_name in enumerate(sample['mask_names']):
                obj_name, part_name = mask_name.split('..')
                obj_cate = obj_name.split('_')[0]
                if '.' in part_name:
                    part_name = part_name.split('.')[0]
                print(obj_cate, part_name)
                if obj_cate not in car_cates:
                    continue
                if part_name not in car_parts:
                    continue
                mask_img[sample['masks'][i] == 1] = colors[count_color]
                count_color += 1
            Image.fromarray(mask_img).save(f'debug_parts.png')

        img = np.array(sample['img'])
        kp = sample['kp']
        kpvis = sample['kpvis']
        for i in range(len(kp)):
            if kpvis[i] > 0:
                img = cv2.circle(img, (kp[i, 0], kp[i, 1]), 1, (0, 255, 0), -1)
                pass
        Image.fromarray(img).save('debug_img.png')
        Image.fromarray(sample['obj_mask']*255).save('debug_mask.png')

        self.transform = transform
