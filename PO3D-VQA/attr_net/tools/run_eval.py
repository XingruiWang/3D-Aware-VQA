'''
@Xingrui
Run evaludation 
'''
import os
import sys
sys.path.append('./')

from options import get_options
from datasets import get_dataloader
from model import get_model
from trainer import get_trainer
import torch

from tqdm import tqdm
from enum import Enum
import numpy as np
import pandas as pd

SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']
SUPERCLEVR_PARTNAMES = ['left_mirror', 'fender_front', 'footrest', 'wheel_front_right', 'crank_arm_left', 'wheel_front_left', 'bumper', 'headlight', 'door_front_left', 'wing', 'front_left_wheel', 'side_stand', 'footrest_left_s', 'tailplane_left', 'wheel_front', 'mirror', 'right_head_light', 'back_left_door', 'left_tail_light', 'head_light_right', 'gas_tank', 'front_bumper', 'tailplane', 'taillight_center', 'back_bumper', 'headlight_right', 'panel', 'front_right_door', 'door_mid_left', 'hood', 'door_left_s', 'front_right_wheel', 'wing_left', 'head_light_left', 'back_right_door', 'tail_light_right', 'seat', 'taillight', 'door_front_right', 'trunk', 'back_left_wheel', 'exhaust_right_s', 'cover', 'brake_system', 'wing_right', 'pedal_left', 'rearlight', 'headlight_left', 'right_tail_light', 'engine_left', 'crank_arm', 'fender_back', 'engine', 'fender', 'door_back_right', 'wheel_back_left', 'back_license_plate', 'cover_front', 'headlight_center', 'engine_right', 'roof', 'left_head_light', 'taillight_right', 'fin', 'saddle', 'mirror_right', 'door', 'bumper_front', 'door_mid_right', 'head_light', 'bumper_back', 'wheel_back_right', 'footrest_right_s', 'drive_chain', 'license_plate_back', 'tail_light', 'pedal', 'windscreen', 'license_plate', 'exhaust_left_s', 'handle_left', 'handle', 'back_right_wheel', 'right_mirror', 'wheel', 'fork', 'taillight_left', 'handle_right', 'front_left_door', 'carrier', 'license_plate_front', 'crank_arm_right', 'wheel_back', 'cover_back', 'propeller', 'exhaust', 'tail_light_left', 'mirror_left', 'pedal_right', 'tailplane_right', 'door_right_s', 'front_license_plate']

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class PartAcc():
    def __init__(self):
        self.keys = []
        self.data = {}
        self.cm_shape = np.zeros((129,129))

        self.id2shape = {i : n for i, n in enumerate(SUPERCLEVR_SHAPES + SUPERCLEVR_PARTNAMES)}
    
    def add(self, name):
        name = int(name)
        if name in self.keys:
            print("Already exists")
            return
        else:
            self.keys.append(name)
            self.data[name] = {
                'shape_acc': AverageMeter('shape acc'),
                'color_acc': AverageMeter('color acc'),
                'material_acc': AverageMeter('material acc'),
                'size_acc': AverageMeter('size acc'),
                'n': 0
            }
    
    def get(self, name):
        name = int(name)
        return self.data[name]

    def get_str(self, name):
        name = int(name)
        data = self.data[name]
        output_str = "Part: {}, shape = {}, color = {}, material = {}, size = {}, count = {}".format(data['shape_acc'],
                                                                            data['color_acc'], 
                                                                            data['material_acc'], 
                                                                            data['size_acc'], 
                                                                            data['n'])
        return output_str                                                           

    def accuracy(self, y, target):
        bs = y[0].size(0)
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target

        shape_acc = sum(torch.argmax(shape.data, dim=1) == shape_gt.cuda()) / bs
        color_acc = sum(torch.argmax(color.data, dim=1) == color_gt.cuda()) / bs
        material_acc = sum(torch.argmax(material.data, dim=1) == material_gt.cuda()) / bs
        size_acc = sum(torch.argmax(size.data, dim=1) == size_gt.cuda()) / bs
        
        return (shape_acc, color_acc, material_acc, size_acc)

    def update(self, name, shape_acc, color_acc, material_acc, size_acc):
        name = int(name)
        self.data[name]['shape_acc'].update(int(shape_acc))
        self.data[name]['color_acc'].update(int(color_acc))
        self.data[name]['material_acc'].update(int(material_acc))
        self.data[name]['size_acc'].update(int(size_acc))
        self.data[name]['n'] += 1



    def batch_update(self, y, target):
        bs = target[0].size(0)
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target
        
        shape_acc = torch.argmax(shape.data, dim=1) == shape_gt.cuda()
        color_acc = torch.argmax(color.data, dim=1) == color_gt.cuda()
        material_acc = torch.argmax(material.data, dim=1) == material_gt.cuda()
        size_acc = torch.argmax(size.data, dim=1) == size_gt.cuda()

        for b in range(bs):
            name = int(shape_gt[b])
            if name not in self.keys:
                self.add(name)
            self.update(name, shape_acc[b], color_acc[b], material_acc[b], size_acc[b])
            self.cm_shape[int(torch.argmax(shape.data, dim=1)[b]), int(shape_gt[b])] += 1

        
    def write(self, path):
        output = "Name,Shape,Color,Material,Size,Count\n"

        for n in self.data:
            o = "{},{},{},{},{},{}\n".format(self.id2shape[n], 
                                            self.data[n]['shape_acc'].avg, 
                                            self.data[n]['color_acc'].avg,
                                            self.data[n]['material_acc'].avg,
                                            self.data[n]['size_acc'].avg,
                                            self.data[n]['n']
                                            )
            output += o
        with open(path, 'w') as f:
            f.write(output)

        column = []
        for i in range(129):
            column.append(self.id2shape[i])
            if i in self.data and self.data[i]['n'] > 0:
                self.cm_shape[:, i] /= self.data[i]['n']
        self.cm_shape = pd.DataFrame(self.cm_shape, columns = column)
        self.cm_shape.to_csv('../../data/attr_net/outputs/eval_model/confusion_matrix.csv')

        


def evaluate(model, val_loader):
    cm_shape = np.zeros((129, 129))
    # cm_color = np.zeros((8, 8))
    # cm_material = np.zeros((2, 2))
    # cm_size = np.zeros((2, 2))

    model.eval()

    losses = AverageMeter('train_losses')
    shape_acc = AverageMeter('shape acc')
    color_acc = AverageMeter('color acc')
    material_acc = AverageMeter('material acc')
    size_acc = AverageMeter('size acc')

    part_acc = PartAcc()


    val_loop = tqdm(val_loader)
    for x, label in val_loop:
        x = x.cuda()
        y = model(x)
        loss_output = model.loss(y, label)
        accs = model.accuracy(y, label)


        losses.update(loss_output.item())

        shape_acc.update(accs[0].item())
        color_acc.update(accs[1].item())
        material_acc.update(accs[2].item())
        size_acc.update(accs[3].item())

        # part_acc.batch_update(y, label)
        val_loop.set_postfix({'loss': losses.avg, 
                                'shape acc': shape_acc.avg,
                                'color acc': color_acc.avg,
                                'material acc': material_acc.avg,
                                'size acc': size_acc.avg,
                                })

    
    # return loss / t if t != 0 else 0
    # part_acc.write('../../data/attr_net/outputs/eval_model/part_acc.csv')
    return (shape_acc.avg, color_acc.avg , material_acc.avg, size_acc.avg)



opt = get_options('val')

# train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')

model = get_model(opt)
model = model.double().cuda()

ckpt = torch.load(opt.load_path)
state_dict = ckpt['model_state']
model.load_state_dict(state_dict)

shape_acc, color_acc , material_acc, size_acc = evaluate(model, val_loader)

Avg_acc = sum([shape_acc, color_acc , material_acc, size_acc]) / 4

print("Accuracy = {}\nAccuracy/shape = {}\nAccuracy/color={}\nAccuracy/material = {}\nAccuracy/size = {}"
    .format(Avg_acc, shape_acc, color_acc , material_acc, size_acc))