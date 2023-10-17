import _init_paths

import argparse
import math
import os
import pickle
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, camera_position_from_spherical_angles
from scipy.optimize import least_squares
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

from src.datasets import SuperCLEVRTest
from src.models import NetE2E, NearestMemoryManager, mask_remove_near, MeshInterpolateModule
from src.optim import pre_compute_kp_coords
from src.utils import str2bool, MESH_FACE_BREAKS_1000, load_off, normalize_features, center_crop_fun, plot_score_map, \
    keypoint_score, plot_mesh, plot_loss_landscape, create_cuboid_pts, pose_error, plot_multi_mesh, notation_blender_to_pyt3d, \
    add_3d

from solve_pose_multi import solve_pose_multi_obj, save_segmentation_maps
import time
import gc

def parse_args():
    parser = argparse.ArgumentParser(description='Inference 6D pose estimation on SuperCLEVR')

    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--test_index', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='/home/wufeim/nemo_superclevr/experiments/oct09_superclevr_car/ckpts/saved_model_15000.pth')
    parser.add_argument('--save_results', type=str, default='/home/wufeim/nemo_superclevr/vis_outputs')
    parser.add_argument('--metrics',type=str, nargs='+', default=['pose_error'])
    parser.add_argument('--thr', type=float, default=20.0)
    parser.add_argument('--pre_filter', type=str2bool, default=False)
    parser.add_argument('--display_position', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')

    

    # Data args
    parser.add_argument('--image_h', type=int, default=480)
    parser.add_argument('--image_w', type=int, default=640)
    parser.add_argument('--mesh_path', type=str, default='/home/wufeim/nemo_superclevr/CAD_cate')
    parser.add_argument('--dataset_path', type=str, default='/home/wufeim/nemo_superclevr/superclevr/superclevr_val/new')
    parser.add_argument('--prefix', type=str, default='superCLEVR')
    parser.add_argument('--split', type=str, default='new')

    # Model args
    parser.add_argument('--backbone', type=str, default='resnetext')
    parser.add_argument('--d_feature', type=int, default=128)
    parser.add_argument('--local_size', type=int, default=1)
    parser.add_argument('--separate_bank', type=str2bool, default=False)
    parser.add_argument('--max_group', type=int, default=512)
    parser.add_argument('--num_noise', type=int, default=0)
    parser.add_argument('--adj_momentum', type=float, default=0.0)

    # Render args
    parser.add_argument('--down_sample_rate', type=int, default=8)
    parser.add_argument('--blur_radius', type=float, default=0.0)
    parser.add_argument('--num_faces', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=0.01)
    parser.add_argument('--mode', type=str, default='bilinear')

    # Optimization args
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--azimuth_sample', type=int, default=12)
    parser.add_argument('--elevation_sample', type=int, default=7)
    parser.add_argument('--theta_sample', type=int, default=3)
    parser.add_argument('--distance_sample', type=int, default=6)
    parser.add_argument('--px_sample', type=int, default=21)
    parser.add_argument('--py_sample', type=int, default=21)
    parser.add_argument('--loss_type', type=str, default='with_clutter')
    parser.add_argument('--adam_beta_0', type=float, default=0.4)
    parser.add_argument('--adam_beta_1', type=float, default=0.6)

    args = parser.parse_args()

    args.mesh_path = os.path.join(args.mesh_path, args.category)
    args.save_results = args.save_results + f'/{args.category}'

    return args

def compute_kpt_per_object(seg_map, kps):

    b, h, w = seg_map.shape
    kps_list = []
    kps_avg_list = []
    kps = np.max(kps, axis=2)
    for i in range(b):
        if np.sum(seg_map[i] == 1) == 0:
            kps_list.append(0)
            kps_avg_list.append(0)
            continue
        object_mask = (seg_map[i] == 1) * (kps > 0)
        kps_obj = kps[object_mask]
        avg = np.sum(kps_obj) / np.sum(object_mask)
        kps_score = np.sum(kps_obj > 0.5) / np.sum(object_mask)
        kps_list.append(kps_score)
        kps_avg_list.append(avg)
        # print(avg, np.sum(object_mask), kps_score)
    return kps_list, kps_avg_list

def keypoint_score2(feature_map, memory, nocs, clutter_score=None, device='cuda:0'):
    if not torch.is_tensor(feature_map):
        feature_map = torch.tensor(feature_map, device=device).unsqueeze(0) # (1, C, H, W)
    if not torch.is_tensor(memory):
        memory = torch.tensor(memory, device=device) # (nkpt, C)
    
    nkpt, c = memory.size()
    feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = memory.view(nkpt, c, 1, 1)

    kpt_map = torch.sum(feature_map * memory, dim=1) # (nkpt, H, W)
    kpt_map, idx = torch.max(kpt_map, dim=0)

    nocs_map = nocs[idx, :].view(kpt_map.shape[0], kpt_map.shape[1], 3).to(device)

    nocs_map = nocs_map * kpt_map.unsqueeze(2)

    if clutter_score is not None:
        nocs_map[kpt_map < clutter_score] = 0.0

    return nocs_map.detach().cpu().numpy()


def get_nocs_features(xvert):
    xvert = xvert.clone()
    xvert[:, 0] -= (torch.min(xvert[:, 0]) + torch.max(xvert[:, 0]))/2.0
    xvert[:, 1] -= (torch.min(xvert[:, 1]) + torch.max(xvert[:, 1]))/2.0
    xvert[:, 2] -= (torch.min(xvert[:, 2]) + torch.max(xvert[:, 2]))/2.0
    xvert[:, 0] /= torch.max(torch.abs(xvert[:, 0])) * 2.0
    xvert[:, 1] /= torch.max(torch.abs(xvert[:, 1])) * 2.0
    xvert[:, 2] /= torch.max(torch.abs(xvert[:, 2])) * 2.0
    xvert += 0.5
    return xvert


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def transform(xvert, theta, scale, loc):
    rotate_mat = get_rot_z(theta / 180. * math.pi)
    xvert = (rotate_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert * scale + loc
    return xvert


def project(params, pts, pts2d):
    azimuth, elevation, theta, distance, px, py = params

    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    azimuth = -azimuth
    elevation = - (math.pi / 2 - elevation)

    Rz = np.array([
        [math.cos(azimuth), -math.sin(azimuth), 0],
        [math.sin(azimuth), math.cos(azimuth), 0],
        [0, 0, 1],
    ])  # rotation by azimuth
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(elevation), -math.sin(elevation)],
        [0, math.sin(elevation), math.cos(elevation)],
    ])  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    P = np.array([[3000, 0, 0],
                  [0, 3000, 0],
                  [0, 0, -1]]).dot(R[:3, :4])

    x3d = pts
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T

    x2d = np.dot(P, x3d_)  # 3x4 * 4x8 = 3x8
    x2d[0, :] = x2d[0, :] / x2d[2, :]
    x2d[1, :] = x2d[1, :] / x2d[2, :]
    x2d = x2d[0:2, :]

    R2d = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]])
    x2d = np.dot(R2d, x2d).T

    x2d[:, 1] *= -1

    x2d[:, 0] += px*448
    x2d[:, 1] += py*320

    # x2d = x2d - (x2d[8] - pts2d[8])
    # return x2d

    return x2d


def fun(params, pts, pts2d):
    x2d = project(params, pts, pts2d)
    return (x2d - pts2d).ravel()


"""
def notation_blender_to_pyt3d(mesh_path, sample):
    # xvert, _ = load_off(mesh_path)
    xvert_orig = create_cuboid_pts(mesh_path)
    xvert = xvert_orig * sample['size_r']

    rot_mat = get_rot_z(sample['theta'] / 180. * math.pi)
    xvert = (rot_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert + sample['location']
    xvert = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    xvert = (sample['mw_inv'] @ xvert)[:3].transpose((1, 0))
    xvert = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    pts_2d = sample['proj_mat'] @ xvert
    pts_2d[0, :] = pts_2d[0, :] / pts_2d[3, :] * 224 + 224
    pts_2d[1, :] = -pts_2d[1, :] / pts_2d[3, :] * 160 + 160
    pts_2d = pts_2d[:2, :].T

    azim_sample = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    elev_sample = [-np.pi/6, 0.0, np.pi/6, np.pi/3]
    theta_sample = [-np.pi/6, 0.0, np.pi/6]
    dist_sample = [4, 8, 12, 16, 20]

    center = np.sum(pts_2d, axis=0)
    min_x, min_cost = None, 1e8
    for azim in azim_sample:
            for elev in elev_sample:
                for theta in theta_sample:
                    for dist in dist_sample:
                        x0 = np.array([azim, elev, theta, dist, center[0]/448, center[1]/320])
                        res = least_squares(fun, x0, x_scale=[2*np.pi, 2*np.pi, 2*np.pi, 20, 1.0, 1.0], ftol=1e-3, method='trf', args=(xvert_orig, pts_2d))
                        if min_cost > res.cost:
                            min_cost = res.cost
                            min_x = res.x
    return min_x, pts_2d, project(min_x, xvert_orig, pts_2d)
"""


def plot_corners(img, kps, c=(255, 0, 0), text=False):
    cv2.circle(img, (int(kps[0]), int(kps[1])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[2]), int(kps[3])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[4]), int(kps[5])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[6]), int(kps[7])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[8]), int(kps[9])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[10]), int(kps[11])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[12]), int(kps[13])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[14]), int(kps[15])), 2, color=c, thickness=2)
    cv2.circle(img, (int(kps[16]), int(kps[17])), 2, color=c, thickness=2)
    if text:
        cv2.putText(img, str(0), (int(kps[0]), int(kps[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(1), (int(kps[2]), int(kps[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(2), (int(kps[4]), int(kps[5])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(3), (int(kps[6]), int(kps[7])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(4), (int(kps[8]), int(kps[9])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(5), (int(kps[10]), int(kps[11])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(6), (int(kps[12]), int(kps[13])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(7), (int(kps[14]), int(kps[15])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
        cv2.putText(img, str(8), (int(kps[16]), int(kps[17])), cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2, cv2.LINE_AA)
    return img


def plot_lines(img, kps, c=(255, 0, 0)):
    pts = np.reshape(kps, (9, 2))
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [3, 7]]
    for i, j in lines:
        img = cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])), (int(pts[j, 0]), int(pts[j, 1])), c, 1)
    return img


def main():
    args = parse_args()
    os.makedirs(args.save_results, exist_ok=True)

    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = SuperCLEVRTest(
        dataset_path=args.dataset_path,
        prefix=f'{args.prefix}_{args.split}',
        category=args.category,
        transform=None
    )
    print(f'found {len(dataset)} samples')
    sample = dataset[0]

    net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size], output_dimension=args.d_feature,
                 reduce_function=None, n_noise_points=args.num_noise, pretrain=True, noise_on_mask=True)
    print(f'num params {sum(p.numel() for p in net.net.parameters())}')
    net = nn.DataParallel(net).cuda().train()
    # checkpoint = torch.load(args.ckpt, map_location='cpu')
    checkpoint = torch.load(args.ckpt)
    net.load_state_dict(checkpoint['state'])

    if isinstance(checkpoint['memory'], torch.Tensor):
        checkpoint['memory'] = [checkpoint['memory']]
    
    xvert, xface = load_off(os.path.join(args.mesh_path, '01.off'), to_torch=True)
    nocs_features = get_nocs_features(xvert).cuda()
    n = int(xvert.shape[0])
    memory_bank = NearestMemoryManager(inputSize=args.d_feature, outputSize=n+args.num_noise*args.max_group, K=1, num_noise=args.num_noise,
                                       num_pos=n, momentum=args.adj_momentum)
    memory_bank = memory_bank.cuda()
    with torch.no_grad():
        memory_bank.memory.copy_(checkpoint['memory'][0][0:memory_bank.memory.shape[0]])
    memory = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].detach().cpu().numpy()
    clutter = checkpoint['memory'][0][memory_bank.memory.shape[0]::].detach().cpu().numpy()  # (2560, 128)
    feature_bank = torch.from_numpy(memory)
    clutter_bank = torch.from_numpy(clutter)
    clutter_bank = clutter_bank.cuda()
    clutter_bank = normalize_features(torch.mean(clutter_bank, dim=0)).unsqueeze(0)  # (1, 128)
    kp_features = checkpoint['memory'][0][0:memory_bank.memory.shape[0]].to(args.device)
    clutter_bank = [clutter_bank]

    render_image_size = max(args.image_h, args.image_w) // args.down_sample_rate
    map_shape = (args.image_h//args.down_sample_rate, args.image_w//args.down_sample_rate)
    cameras = PerspectiveCameras(focal_length=12.0, device=args.device)
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=args.blur_radius,
        faces_per_pixel=args.num_faces,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    inter_module = MeshInterpolateModule(xvert, xface, feature_bank, rasterizer, post_process=center_crop_fun(map_shape, (render_image_size, ) * 2))
    inter_module = inter_module.cuda()

    poses, kp_coords, kp_vis = pre_compute_kp_coords(os.path.join(args.mesh_path, '01.off'),
                                                     mesh_face_breaks=MESH_FACE_BREAKS_1000[args.category],
                                                     azimuth_samples=np.linspace(0, np.pi*2, args.azimuth_sample, endpoint=False),
                                                     elevation_samples=np.linspace(-np.pi/2, np.pi/2, args.elevation_sample),
                                                     theta_samples=np.linspace(-np.pi/3, np.pi/3, args.theta_sample),
                                                     distance_samples=np.linspace(5, 30, args.distance_sample, endpoint=True))
    poses = poses.reshape(args.azimuth_sample, args.elevation_sample, args.theta_sample, args.distance_sample, 4)
    
    object_score_dict = {}
    for i in tqdm(range(len(dataset)), desc = args.category, position = args.display_position):
        sample = dataset[i]
        # gt_poses = [notation_blender_to_pyt3d(os.path.join(args.mesh_path, '01.off'), s, sample, args.image_h, args.image_w)[0] for s in sample['objects'] if s['category'] == args.category]
        # for g in gt_poses:
            # g['used'] = False

        img_tensor = norm(to_tensor(sample['img'])).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(args.device)
            feature_map = net.module.forward_test(img_tensor)
        
        if args.pre_filter:
            with torch.no_grad():
                nkpt, c = kp_features.size()
                b, c, hm_h, hm_w = feature_map.size()
                kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * kp_features.view(nkpt, c, 1, 1), dim=1)
                kpt_score_map, _ = torch.max(kpt_score_map, dim=0, keepdim=True)  # (1, h, w)
                kpt_score_map = F.unfold(kpt_score_map.unsqueeze(0), kernel_size=(3, 3), padding=1).squeeze().sum(dim=0)  # (h*w, )

                px_samples = torch.from_numpy(np.linspace(0, args.image_w, args.px_sample, endpoint=True)).to(args.device)
                py_samples = torch.from_numpy(np.linspace(0, args.image_h, args.py_sample, endpoint=True)).to(args.device)
                xv, yv = torch.meshgrid(px_samples, py_samples)
                principal_samples = torch.stack([xv, yv], dim=2).reshape(-1, 2)
                principal_samples = torch.round(principal_samples/args.down_sample_rate)
                principal_samples[:, 0] = torch.clamp(principal_samples[:, 0], min=0, max=hm_w-1)
                principal_samples[:, 1] = torch.clamp(principal_samples[:, 1], min=0, max=hm_h-1)
                ind = principal_samples[:, 1] * hm_w + principal_samples[:, 0]

                corr = torch.take_along_dim(kpt_score_map, ind.long())

                ind = corr.argsort(descending=True)[:100]
                xv = xv.reshape(-1)[ind].detach().cpu().numpy()
                yv = yv.reshape(-1)[ind].detach().cpu().numpy()
            pred = solve_pose_multi_obj(
                feature_map, inter_module, kp_features, clutter_bank, poses, kp_coords, kp_vis,
                epochs=args.epochs,
                lr=args.lr,
                adam_beta_0=args.adam_beta_0,
                adam_beta_1=args.adam_beta_1,
                mode=args.mode,
                loss_type=args.loss_type,
                device=args.device,
                px_samples=px_samples,
                py_samples=py_samples,
                clutter_img_path=None,
                object_img_path=None,
                blur_radius=args.blur_radius,
                verbose=True,
                down_sample_rate=args.down_sample_rate,
                hierarchical=0,
                xv=xv,
                yv=yv
            )
        else:
            px_samples = np.linspace(0, args.image_w, args.px_sample, endpoint=True)
            py_samples = np.linspace(0, args.image_h, args.py_sample, endpoint=True)

            pred = solve_pose_multi_obj(
                feature_map, inter_module, kp_features, clutter_bank, poses, kp_coords, kp_vis,
                epochs=args.epochs,
                lr=args.lr,
                adam_beta_0=args.adam_beta_0,
                adam_beta_1=args.adam_beta_1,
                mode=args.mode,
                loss_type=args.loss_type,
                device=args.device,
                px_samples=px_samples,
                py_samples=py_samples,
                clutter_img_path=None,
                object_img_path=None,
                blur_radius=args.blur_radius,
                verbose=True,
                down_sample_rate=args.down_sample_rate,
                hierarchical=0,
            )
        clutter_score = None
        for cb in clutter_bank:
            cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
            if clutter_score is None:
                clutter_score = cs
            else:
                clutter_score = torch.max(clutter_score, cs)
        os.makedirs(os.path.join(args.save_results, sample['img_name']), exist_ok=True)

        with open(os.path.join(args.save_results, '{}/{}_results.json'.format(sample['img_name'], sample['img_name'])), 'w') as f:
            # pred = [p for p in pred['final'] if p['score']>=args.thr]
            if not 'seg_maps' in pred:
                continue
            seg_union = np.zeros((pred['seg_maps'].shape[1], pred['seg_maps'].shape[2]))
            for i in range(pred['seg_maps'].shape[0]):
                object_mask = pred['seg_maps'][i]==1 
                seg_union[object_mask] = i+1
            # print(seg_union)
            cv2.imwrite(os.path.join(args.save_results, '{}/{}.png'.format(sample['img_name'], sample['img_name'])), seg_union )
            if len(pred['final']) > 0:

                kpt_score = keypoint_score2(feature_map, memory, nocs_features, clutter_score)
                kps_score, kps_avg_list = compute_kpt_per_object(pred['seg_maps'], kpt_score)
                
                for i, p in enumerate(pred['final']):
                    p['corr2d_max'] = float(p['corr2d_max'])
                    p['keypoint_ratio'] = kps_score[i]
                    p['keypoint_avg'] = kps_avg_list[i]

                # kps_score[i], pred['final'][i]['corr2d_max'], pred['final'][i]['score'], args.category
            json.dump(pred['final'], f)
        # ax = sns.heatmap(np.transpose(pred['corr2d']), square=True, xticklabels=False, yticklabels=False, cbar=False)
        # ax.figure.tight_layout()
        # ax.figure.savefig(os.path.join(args.save_results, f'corr2d.png'), bbox_inches='tight', pad_inches=0.0)
        # plot the corr2d map
        corr_name = os.path.join(args.save_results, sample['img_name'],f'corr2d.png')
        plot_score_map(np.transpose(pred['corr2d']), corr_name)

        # plt.imshow(np.transpose(pred['corr2d']), interpolation='nearest')
        # plt.tight_layout()
        # print(os.path.join(args.save_results, sample['img_name'],f'corr2d.png'))
        # plt.savefig(os.path.join(args.save_results, sample['img_name'],f'corr2d.png'))
        plt_end = time.time()
        # # print('pred', len(pred['final']))
        # # print('sample', sample['num_objs'])

        save_segmentation_maps(pred['seg_maps'], os.path.join(args.save_results, sample['img_name'],f'segmentation.png'))
        seg_end = time.time()
        # sample['img'] = np.array(sample['img'])
        # Image.fromarray(sample['img']).save(os.path.join(args.save_results, f'input.png'))

        # # save clutter score map
        # clutter_score = None
        # for cb in clutter_bank:
        #     cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        #     if clutter_score is None:
        #         clutter_score = cs
        #     else:
        #         clutter_score = torch.max(clutter_score, cs)
        # plot_score_map(clutter_score.detach().cpu().numpy(), os.path.join(args.save_results, f'clutter_score.png'))

        kpt_score = keypoint_score(feature_map, memory)
        plot_score_map(kpt_score.detach().cpu().numpy(), os.path.join(args.save_results, sample['img_name'], f'kp_score_agnostic.png'))
        # # save kp score map
        # kpt_score = keypoint_score2(feature_map, memory, nocs_features, clutter_score)
        # kpt_score = np.rint(np.clip(kpt_score * 255.0, 0, 255)).astype(np.uint8)
        # kpt_score = cv2.resize(kpt_score, (560, 400), interpolation=cv2.INTER_NEAREST)
        # Image.fromarray(kpt_score).save(os.path.join(args.save_results, f'kp_score.png'))

        # # visualize predicted pose
        plot_pred = [p for p in pred['final'] if p['score']>=30 and p['corr2d_max'] > 150]

        # for p in plot_pred:
        #     print(p['principal'], p['distance'])
        # print()

        scores = [p['score'] for p in plot_pred]
        img = plot_multi_mesh(np.asarray(sample['img']), os.path.join(args.mesh_path, '01.off'), plot_pred, down_sample_rate=args.down_sample_rate)

        text_size, _ = cv2.getTextSize('0.1Nonefp', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        line_height = text_size[1] + 2
        for p in plot_pred:
            l = p['score']
            center = (int(p['principal'][0]), int(p['principal'][1]))
            img = cv2.putText(img, f'{l:.2f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            center = (int(p['principal'][0]), int(p['principal'][1]+line_height))
            # if p['err'] is None:
            #     img = cv2.putText(img, 'fp', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # elif p['err'] <= np.pi/18:
            #     img = cv2.putText(img, f'{p["err"]:.2f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # elif p['err'] <= np.pi/6:
            #     img = cv2.putText(img, f'{p["err"]:.2f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # else:
            #     img = cv2.putText(img, f'{p["err"]:.2f}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        object_score = np.zeros_like(pred['object_score_maps'][0])
        for p in pred['object_score_maps']:
            object_score = np.maximum(p, object_score)
        plot_score_map(object_score, os.path.join(args.save_results, sample['img_name'], f'obj_score.png'), vmin=0.0, vmax=1.0, color=False)

        # object_score_dict[i] = object_score

        Image.fromarray(img).save(os.path.join(args.save_results, sample['img_name'], f'pred.png'))
if __name__ == '__main__':
    main()
