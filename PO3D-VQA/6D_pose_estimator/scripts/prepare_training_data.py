import _init_paths

import argparse
import concurrent.futures
import json
import math
import os
import random
import shutil

import cv2
import multiprocessing
import numpy as np
from PIL import Image
import scipy.io as sio
from scipy.spatial import ConvexHull
from tqdm import tqdm

from src.models.calculate_point_direction import cal_point_weight, direction_calculator
from src.models.mesh_memory_map import MeshConverter
from src.utils import str2bool, subcate_to_cate, load_off, MESH_FACE_BREAKS_1000


def parse_args():
    parser = argparse.ArgumentParser('Create SuperCLEVR 6D pose training set')
    parser.add_argument('--dataset_name', type=str, default='superclevr')
    parser.add_argument('--categories', type=str, nargs='+', default=['aeroplane', 'bus', 'car', 'bicycle', 'motorbike'])
    parser.add_argument('--workers', type=int, default=8)

    # Data args
    parser.add_argument('--image_h', type=int, default=480)
    parser.add_argument('--image_w', type=int, default=640)
    parser.add_argument('--save_path', type=str, default='/home/wufeim/nemo_superclevr/superclevr')
    parser.add_argument('--mesh_path', type=str, default='/home/wufeim/nemo_superclevr/CAD_cate')
    parser.add_argument('--dataset_path', type=str, default='/home/wufeim/nemo_superclevr/superclevr/output/superclevr_nemo')
    parser.add_argument('--filename_prefix', type=str, default='superCLEVR')
    parser.add_argument('--splits', type=str, nargs='+', default=['train'])

    parser.add_argument('--part_mask', type=str2bool, default=False)

    return parser.parse_args()


def srgb_to_linear(x, mod=0.1):
    if x <=0.04045 :
        y = x / 12.92
    else:
        y = ((x + 0.055) / 1.055) ** 2.4
    if mod is not None:
        y = round(y / mod) * mod
    return y


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def get_face_centers(xvert):
    xmin, xmax = np.min(xvert[:, 0]), np.max(xvert[:, 0])
    ymin, ymax = np.min(xvert[:, 1]), np.max(xvert[:, 1])
    zmin, zmax = np.min(xvert[:, 2]), np.max(xvert[:, 2])
    centers = np.array([
        [(xmin+xmax)*0.5, (ymin+ymax)*0.5, zmin],
        [(xmin+xmax)*0.5, (ymin+ymax)*0.5, zmax],
        [(xmin+xmax)*0.5, ymin, (zmin+zmax)*0.5],
        [(xmin+xmax)*0.5, ymax, (zmin+zmax)*0.5],
        [xmin, (ymin+ymax)*0.5, (zmin+zmax)*0.5],
        [xmax, (ymin+ymax)*0.5, (zmin+zmax)*0.5],
    ])
    return centers


def get_nocs_colors(xvert):
    xvert = xvert.copy()
    xvert[:, 0] -= (np.min(xvert[:, 0]) + np.max(xvert[:, 0]))/2.0
    xvert[:, 1] -= (np.min(xvert[:, 1]) + np.max(xvert[:, 1]))/2.0
    xvert[:, 2] -= (np.min(xvert[:, 2]) + np.max(xvert[:, 2]))/2.0
    xvert[:, 0] /= np.max(np.abs(xvert[:, 0])) * 2.0
    xvert[:, 1] /= np.max(np.abs(xvert[:, 1])) * 2.0
    xvert[:, 2] /= np.max(np.abs(xvert[:, 2])) * 2.0
    xvert += 0.5
    xvert = np.rint(np.clip(xvert * 255.0, 0, 255)).astype(np.uint8)
    return xvert


def transform(xvert, theta, scale, loc):
    rotate_mat = get_rot_z(theta / 180. * math.pi)
    xvert = (rotate_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert * scale + loc
    return xvert


def dist(loc, cam_loc):
    return np.sqrt(np.sum((np.array(loc)-np.array(cam_loc))**2))


def get_visibility(face_centers, cam_loc, cate):
    vis = [0, 1, 0, 0, 0, 0]

    dist_list = [dist(c, cam_loc) for c in face_centers]
    if dist_list[2] <= dist_list[3]:
        vis[2] = 1
    else:
        vis[3] = 1
    if dist_list[4] <= dist_list[5]:
        vis[4] = 1
    else:
        vis[5] = 1
    
    breaks = [0] + MESH_FACE_BREAKS_1000[cate]
    kpvis = np.zeros((breaks[-1],), dtype=np.uint8)
    for i in range(6):
        if vis[i] == 1:
            kpvis[breaks[i]:breaks[i+1]] = 1
    return kpvis


def prepare_object(obj_dict, cam_loc, occ, proj_mat, mw_inv, args):
    theta = obj_dict['rotation']
    subcate = obj_dict['shape']
    cate = subcate_to_cate[subcate]
    scale = obj_dict['size_r']
    x, y = obj_dict['3d_coords'][:2]
    loc = obj_dict['location']

    # save elevation
    elevation = np.arcsin(cam_loc[2] / dist(loc, cam_loc))

    xvert, xface = load_off(os.path.join(args.mesh_path, cate, '01.off'))
    face_centers = get_face_centers(xvert)
    colors = get_nocs_colors(xvert)

    xvert = transform(xvert, theta, scale, loc)
    pts_3d = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    face_centers = transform(face_centers, theta, scale, loc)
    kpvis = get_visibility(face_centers, cam_loc, cate)

    P = np.dot(proj_mat, mw_inv)
    pts_2d = np.dot(P, pts_3d)
    pts_2d[0, :] = pts_2d[0, :] / pts_2d[3, :] * (args.image_w//2) + (args.image_w//2)
    pts_2d[1, :] = -pts_2d[1, :] / pts_2d[3, :] * (args.image_h//2) + (args.image_h//2)
    pts_2d = np.rint(pts_2d.transpose((1, 0))[:, :2]).astype(np.int32)

    kpvis = np.logical_and(kpvis, np.all(pts_2d >= np.zeros_like(pts_2d), axis=1))
    kpvis = np.logical_and(kpvis, np.all(pts_2d < np.array([args.image_w, args.image_h]), axis=1))
    pts_2d = np.max([np.zeros_like(pts_2d), pts_2d], axis=0)
    pts_2d = np.min([np.ones_like(pts_2d) * (np.array([args.image_w, args.image_h]) - 1), pts_2d], axis=0)

    for i, p in enumerate(pts_2d):
        if kpvis[i] > 0 and occ[p[1], p[0]] == 1:
            kpvis[i] = 0
    
    try:
        hull = ConvexHull(pts_2d).vertices
    except:
        return None, None

    obj_mask = np.zeros((args.image_h, args.image_w), dtype=np.uint8)
    obj_mask = cv2.fillPoly(obj_mask, [pts_2d[hull].reshape(-1, 1, 2)], 1) & (1-occ)
    occ = cv2.fillPoly(occ, [pts_2d[hull].reshape(-1, 1, 2)], 1)

    return {
        'category': cate,
        'sub_category': subcate,
        'theta': theta,
        'scale': scale,
        'loc': loc,
        'kp': pts_2d,
        'kpvis': kpvis,
        'obj_mask': obj_mask,
        'distance': dist(loc, cam_loc),
        'elevation': elevation
    }, occ


def process_sample(fname, image_path, scene_path, mask_path, args):
    with open(scene_path) as f:
        scene = json.load(f)
    
    mw = np.array(scene['matrix_world'])
    mw_inv = np.array(scene['matrix_world_inverted'])
    proj_mat = np.array(scene['projection_matrix'])
    cam_loc = np.array(scene['camera_location'])

    objects = scene['objects']
    dist_list = [dist(obj['location'], cam_loc) for obj in objects]
    objects = [obj for _, obj in sorted(list(zip(dist_list, objects)), key=lambda x: x[0])]

    occ = np.zeros((args.image_h, args.image_w), dtype=np.uint8)
    all_objects = []
    for i in range(len(objects)):
        obj, occ = prepare_object(objects[i], cam_loc, occ, proj_mat, mw_inv, args)
        if obj is None or occ is None:
            return None
        all_objects.append(obj)
    
    if mask_path is not None:
        mask = np.array(Image.open(mask_path))
        s = set()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                s.add((mask[i][j][0], mask[i][j][1], mask[i][j][2]))

        color_to_map = {}
        for color in s:
            m = np.all(mask[:, :, :3] == np.array(color).astype(np.uint8), axis=-1)
            color_key = (srgb_to_linear(color[0]/255.0), srgb_to_linear(color[1]/255.0), srgb_to_linear(color[2]/255.0))
            color_to_map[color_key] = m
        
        part_colors = scene['part_colors']
        all_mask_names, all_masks = [], []
        for p_name, p_color in part_colors.items():
            obj_name, part_name = p_name.split('..')
            p_color = tuple(p_color)
            if p_color == (0.1, 0.1, 0.1):
                print(p_name, p_color)
            if p_color not in color_to_map:
                continue
            else:
                all_masks.append(color_to_map[p_color])
                all_mask_names.append(p_name)
        all_masks = np.stack(all_masks, axis=0)

    if mask_path is not None:
        return {
            'img_name': fname,
            'matrix_world': mw,
            'matrix_world_inv': mw_inv,
            'projection_matrix': proj_mat,
            'camera_location': cam_loc,
            'obj_mask': occ,
            'objects': all_objects,
            'all_mask_names': all_mask_names
        }, all_masks
    else:
        return {
            'img_name': fname,
            'matrix_world': mw,
            'matrix_world_inv': mw_inv,
            'projection_matrix': proj_mat,
            'camera_location': cam_loc,
            'obj_mask': occ,
            'objects': all_objects
        }


def worker(split, save_path, args, start_idx=0, end_idx=500):
    print(f'Start preparing images for {split} from {start_idx} to {end_idx}')
    out_str = '\n'
    save_img_path = os.path.join(save_path, split, 'images')
    save_mask_path = os.path.join(save_path, split, 'masks')
    save_anno_path = os.path.join(save_path, split, 'annotations')

    os.makedirs(save_img_path, exist_ok=True)
    os.makedirs(save_mask_path, exist_ok=True)
    os.makedirs(save_anno_path, exist_ok=True)

    img_path = os.path.join(args.dataset_path, 'images')
    mask_path = os.path.join(args.dataset_path, 'masks')
    scene_path = os.path.join(args.dataset_path, 'scenes')

    fnames = [f'{args.filename_prefix}_{split}_{i:06d}' for i in range(start_idx, end_idx)]
    error_cases = []
    for fname in fnames:
        if args.part_mask:
            save_dict, masks = process_sample(fname, os.path.join(img_path, fname+'.png'), os.path.join(scene_path, fname+'.json'), os.path.join(mask_path, fname+'.png'), args)
        else:
            save_dict = process_sample(fname, os.path.join(img_path, fname+'.png'), os.path.join(scene_path, fname+'.json'), None, args)
        if save_dict is None:
            error_cases.append(fname)
            continue

        shutil.copyfile(os.path.join(img_path, fname+'.png'), os.path.join(save_img_path, fname+'.png'))
        np.savez(os.path.join(save_anno_path, fname), **save_dict)
        if args.part_mask:
            np.save(os.path.join(save_mask_path, fname+'.npy'), masks)
    
    if len(error_cases) > 0:
        print('Error cases:')
        print(error_cases)


def main():
    args = parse_args()
    print(args)

    save_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)

    objs = []

    for i in range(4):
        objs.append(threadpool_executor.submit(worker, 'new', save_path, args, start_idx=100*i, end_idx=100*i+100))

    for obj in tqdm(objs):
        obj.result()


def test():
    args = parse_args()
    print(args)

    save_path = os.path.join(args.save_path, args.dataset_name)
    os.makedirs(save_path, exist_ok=True)

    worker('new', save_path, args, start_idx=0, end_idx=100)


if __name__ == '__main__':
    main()
