import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.spatial import distance_matrix
import torch
from torchvision import transforms


def hflip(sample, mapping):
    sample['img'] = transforms.functional.hflip(sample['img'])
    sample['obj_mask'] = np.fliplr(sample['obj_mask']).copy()
    sample['kp'][:, 1] = sample['img'].size[0] - sample['kp'][:, 1] - 1
    sample['kp'] = sample['kp'][mapping]
    sample['iskpvisible'] = sample['iskpvisible'][mapping]
    return sample


class RandomHorizontalFlip(object):
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.prepare_mapping()
        self.trans = transforms.RandomApply([lambda x, y=self.mapping: hflip(x, y)], p=0.5)
    
    def prepare_mapping(self):
        xvert, _ = load_off(self.mesh_path)
        xvert_prime = xvert.copy()
        xvert_prime[:, 0] = -xvert[:, 0]
        dist_mat = distance_matrix(xvert, xvert_prime)
        self.mapping = np.argmin(dist_mat, axis=1)
    
    def __call__(self, sample):
        sample = self.trans(sample)
        return sample


class ToTensor(object):
    def __init__(self):
        self.trans = transforms.ToTensor()
    
    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        # sample['kp'] = torch.tensor(sample['kp'], dtype=torch.int32)
        # sample['kpvis'] = torch.tensor(sample['kpvis'], dtype=troch.float32)
        # sample['distance'] = torch.tensor(sample['distance'], dtype=torch.float32)
        # sample['obj_mask'] = torch.tensor(sample['obj_mask'], dtype=torch.int32)
        return sample


class Normalize(object):
    def __init__(self):
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        sample['img'] = self.trans(sample['img'])
        return sample
