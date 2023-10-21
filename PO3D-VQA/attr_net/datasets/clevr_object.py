
import os
import json

import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
import pycocotools.mask as mask_util

import time
import copy


CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']

def _preprocess_rle(rle):
    """Turn ASCII string into rle bytes"""
    rle['counts'] = rle['counts'].encode('ASCII')
    return rle

def seg2bbox(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""

    mask = np.array(mask_util.decode(masks), dtype=np.float32)

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return int(inds.min()), int(inds.max())


    flat_mask = mask.sum(axis=0)
    x0, x1 = get_bounds(flat_mask)
    flat_mask = mask.sum(axis=1)
    y0, y1 = get_bounds(flat_mask)
    # boxes = np.array([x0, y0, x1, y1])
    boxes = [x0, y0, x1, y1]

    return boxes

class ClevrObjectDataset(Dataset):
    def __init__(self, img_dir, obj_ann_path, scene_path, split = 'train', 
                border_noise = True, trim = 1.0, bbox_mode = 'XYXY', aug_level = "light"):
        '''
        type: objects or parts
        split: train or val
        self.anns: all annotated obejct & part, same format as prediction output
        self.object_list: List of object, contain (image_file, attributes, bbox), used for training / test attr net
        '''
        if split != "train":
            self.border_noise = False
        self.trim = trim
        self.obj_ann_path = obj_ann_path
        self.level = aug_level
        print(obj_ann_path)
        if not obj_ann_path or not os.path.exists(obj_ann_path):
            print("Generate annotation file from scenes")
            self.anns = self.anno_object(scene_path, save = obj_ann_path)
        else:        
            print("File exist. Load from given file: ", obj_ann_path)
            with open(obj_ann_path) as f:   
                self.anns = json.load(f)

        self.bbox_mode = bbox_mode

        self.img_dir = img_dir
        print("----------------", split)
        self.object_list = self.filter_object(split)
        # transform_list = [transforms.ToPILImage(),
        #                   transforms.Resize((224, 224)),
        #                   transforms.ToTensor(),
        #                   transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
        # self._transform = transforms.Compose(transform_list)

        self.shape2id = {c: i for i, c in enumerate(CLEVR_SHAPES)}
        self.size2id = {c: i for i, c in enumerate(CLEVR_SIZES)}
        self.color2id = {c: i for i, c in enumerate(CLEVR_COLORS)}
        self.materail2id = {c: i for i, c in enumerate(CLEVR_MATERIALS)}

        self.border_noise = border_noise

        
    def get_box(self, seed, box, level = "hard"):
        np.random.seed(seed)
        if level == "hard":
            e_w = int((box[2] -  box[0]) * 0.4)
            e_h = int((box[3] -  box[1]) * 0.4)

            e_w = min(e_w, 60)
            e_h = min(e_h, 60)

            b_w = np.random.randint(-e_w * 0.3, e_w, size=2)
            b_h = np.random.randint(-e_h * 0.3, e_h, size=2)

            a = max(0, box[0] - b_w[0])
            b = max(0, box[1] - b_h[0])
            c = box[2] + b_w[1]
            d = box[3] + b_h[1]


        elif level == 'light':
            epsilon =  int((box[2] -  box[0] + box[3] -  box[1]) * 0.05)
            epsilon = max(epsilon, 6)            
            border = np.random.randint(0, epsilon, size=4)

            a = max(0, box[0] - border[0])
            b = max(0, box[1] - border[1])
            c = box[2] + border[2]
            d = box[3] + border[3]

        # if a >= c:
        #     a = box[0]
        #     c = box[2]

        # if b >= d:
        #     b = box[1]
        #     d = box[3]            
        return [a, b, c, d]

    def filter_object(self, split):
        print(split, "======================")
        object_list = []
        all_id = list(self.anns.keys())

        if split == 'train':
            all_id = all_id[:int(len(all_id) * 0.8 * self.trim)]
            # all_id = all_id[:32]
        elif split == 'val':
            all_id = all_id[int(len(all_id) * 0.8):]
            # all_id = all_id[:32]
            # all_id = all_id[int(len(all_id) * 0.8):(int(len(all_id) * 0.8) + int(len(all_id) * 0.2 * self.trim))]
            # all_id = all_id[int(len(all_id) * 0.5):int(len(all_id) * 0.5) + 32]
        elif split == 'test':
            # all_id = all_id[:32]
            all_id = all_id[int(len(all_id) * 0.8):]   



        for i in all_id:
            for o in self.anns[i]:
                mask = copy.deepcopy(o['mask'])
                mask['counts'] = eval(mask['counts'])
                object_list.append({'image_filename': os.path.join(self.img_dir, o['image_filename']),
                                    'bbox': o['bbox'],
                                    'shape': o['shape'],
                                    'color': o['color'],
                                    'material': o['material'],
                                    'size': o['size'],
                                    'mask': mask,
                                    })

        return object_list

    def __len__(self):
        print()
        return len(self.object_list)

    def transform(self, img):

        img = img.astype(np.float)
        img = cv2.resize(img, (224, 224),
                        interpolation = cv2.INTER_CUBIC)
        img = img[:, :, ::-1].copy()

        img /= 255.0
        img -= [0.485, 0.456, 0.406]
        img /= [0.229, 0.224, 0.225]

        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        object = self.object_list[idx]
        if 'mask' in object:
            return self.__getitem_seg__(idx)
        else:
            return self.__getitem_bbox__(idx)

    def __getitem_seg__(self, idx):
        object = self.object_list[idx]
        img_name = object['image_filename']
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        mask = np.array(mask_util.decode(object['mask']), dtype=np.float32)
        mask = cv2.resize(mask, (224, 224),
                        interpolation = cv2.INTER_NEAREST)[:, :, None]
        # img_view = mask * img * 0.8 +  (1-mask) * img * 0.2
        # cv2.imwrite('test_{}_{}_{}_{}_{}.png'.format(idx, object['shape'], object['color'], object['material'], object['size']), img_view )   
        img = self.transform(img)

        mask = torch.from_numpy(mask.transpose((2, 0, 1)))
        
        object_img = torch.cat([img, mask], dim=0)

        if object_img is None:
            print(box)
        # cv2.imwrite('output/test_{}.png'.format(idx), object_img)
        
        shape = torch.tensor(self.shape2id[object['shape']]).long()
        color = torch.tensor(self.color2id[object['color']]).long()
        material = torch.tensor(self.materail2id[object['material']]).long()
        size = torch.tensor(self.size2id[object['size']]).long()
        return object_img, (shape, color, material, size)   

    def __getitem_bbox__(self, idx):
        object = self.object_list[idx]
        img_name = object['image_filename']
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)

        object_box = object['bbox']

        if self.bbox_mode == 'XYWH':
            box = [int(object_box[0]), int(object_box[1]), int(object_box[2] + object_box[0]), int(object_box[1] + object_box[3])]
        elif self.bbox_mode == 'XYXY':
            box = [int(object_box[0]), int(object_box[1]), int(object_box[2]), int(object_box[3])]
        if self.border_noise:
            box = self.get_box(idx+100, box, level = self.level)

        object_img = img[box[1]:box[3], box[0]: box[2], :]
        # cv2.imwrite('test_{}_{}_{}_{}_{}.png'.format(idx, object['shape'], object['color'], object['material'], object['size']), object_img)


        if object_img is None:
            print(box)
        # cv2.imwrite('output/test_{}.png'.format(idx), object_img)
        
        object_img = self.transform(object_img)

        shape = torch.tensor(self.shape2id[object['shape']]).long()
        color = torch.tensor(self.color2id[object['color']]).long()
        material = torch.tensor(self.materail2id[object['material']]).long()
        size = torch.tensor(self.size2id[object['size']]).long()

        return object_img, (shape, color, material, size)


    def anno_object(self, scene_path, idx_range = [0, 0.5], save = None):
        if idx_range[0] == None:
            idx_range[0] = 0

        ann = {}
        with open(scene_path, 'r') as scene_file:
            scenes = json.load(scene_file)

        scenes = scenes['scenes']

        print("Annotating scenes ...")
        for scene in tqdm(scenes[:]):
            image_id = scene['image_index']
            image_filename = scene['image_filename']
            objects = scene['objects']

            ann[image_id] = []

            for i, o in enumerate(objects):
                anno_object = {}
                anno_object['image_filename'] = image_filename
                anno_object['id'] = i
                anno_object['shape'] = o['shape']
                anno_object['color'] = o['color']
                anno_object['material'] = o['material']
                anno_object['size'] = o['size']
                anno_object['mask'] = copy.deepcopy(o['mask'])

                mask = _preprocess_rle(o['mask'])
                bbox = seg2bbox(mask)
                anno_object['bbox'] = bbox

                ann[image_id].append(anno_object)
        if save:
            with open(save, 'w') as save_file:
                print(ann)
                json.dump(ann, save_file, indent = 2)

        return ann


if __name__ == '__main__':
    # img = cv2.imread("/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/images/superCLEVR_new_000000.png", cv2.IMREAD_COLOR)
    # object_box = [262, 120, 82, 65]
    # box = [object_box[0], object_box[1], object_box[2] + object_box[0], object_box[1] + object_box[3]]

    # object_img = img[box[1]:box[3], box[0]: box[2], :]
    # object_img = self._transform(object_img)
    # # cv2.imwrite("test.png", img)


    dataset = SuperClevrObjectDataset('/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/images', 
                                    None,
                                    '/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/scenes.json')

    for x, label in dataset:

        print(label)

    def get_dataset():
        # def __init__(self, img_dir, obj_ann_path, scene_path, type = 'part', split = 'train', 
        # os.path.join('/mnt/data0/xingrui/superclevr_anno/', 'superclevr_anno.json')
        # /mnt/data0/xingrui/superclevr_anno/superclevr_anno.json
        
        # dataset = SuperClevrObjectDataset('/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/images', 
        #                                 None,
        #                                 '/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/scenes.json')
        ds = ClevrObjectDataset('/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/images', 
                                    '/mnt/data0/xingrui/superclevr_anno/superclevr_anno.json',
                                    '/home/xingrui/vqa/NSCL_super/data/superclevr/0k55k/scenes.json', 
                                    'part',
                                    'train')
        return ds


    def get_dataloader(opt, split):
        ds = get_dataset(type = opt.type)
        loader = DataLoader(ds, 32, num_workers=4, shuffle=True)
        return loader

