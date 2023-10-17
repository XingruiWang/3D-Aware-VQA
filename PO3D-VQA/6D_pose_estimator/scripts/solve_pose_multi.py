import copy
import time

import cv2
import numpy as np
from PIL import Image
from pytorch3d.renderer import OpenGLPerspectiveCameras, RasterizationSettings, MeshRasterizer, camera_position_from_spherical_angles
import seaborn as sns
from skimage.feature import peak_local_max
import torch

from src.utils import camera_position_to_spherical_angle
from src.optim import loss_func_type_a, loss_func_type_b, loss_func_type_c, loss_func_type_d
from src.utils import flow_warp

from tqdm import tqdm
import ipdb
# colors = np.array([
#     (0, 0, 0),
#     (31, 119, 180),
#     (255, 127, 14),
#     (44, 160, 44),
#     (214, 39, 40),
#     (148, 103, 189),
#     (140, 86, 75),
#     (227, 119, 194),
#     (127, 127, 127),
#     (188, 189, 34),
#     (23, 190, 207)
# ], dtype=np.uint8)

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

def eval_loss(pred, feature_map, inter_module, clutter_bank, use_z=False, down_sample_rate=8,
              loss_type='with_clutter', mode='bilinear', blur_radius=0.0, device='cuda:0'):
    if loss_type == 'without_clutter':
        loss_func = loss_func_type_a
    elif loss_type == 'with_clutter':
        loss_func = loss_func_type_b
    elif loss_type == 'z_map':
        loss_func = loss_func_type_c
    elif loss_type == 'softmax':
        loss_func = loss_func_type_d
    else:
        raise ValueError('Unknown loss function type')
    use_z = (loss_type == 'z_map')

    b, c, hm_h, hm_w = feature_map.size()

    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)

    C = camera_position_from_spherical_angles(pred['distance'], pred['elevation'], pred['azimuth'], degrees=False, device=device)
    C = torch.nn.Parameter(C, requires_grad=True)
    theta = torch.tensor(pred['theta'], dtype=torch.float32).to(device)
    theta = torch.nn.Parameter(theta, requires_grad=True)
    max_principal = pred['principal']
    flow = torch.tensor([-(max_principal[0]-hm_w*down_sample_rate/2)/down_sample_rate / 10.0, -(max_principal[1]-hm_h*down_sample_rate/2)/down_sample_rate / 10.0], dtype=torch.float32).to(device)
    flow = torch.nn.Parameter(flow, requires_grad=True)

    if use_z:
        z = torch.nn.Parameter(0.5 * torch.ones((predicted_map.size(0), 1, predicted_map.size(1), predicted_map.size(2)),
                                                dtype=torch.float32, device=predicted_map.device), requires_grad=True)

    projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
    flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
    projected_map = flow_warp(projected_map.unsqueeze(0), flow_map * 10.0)[0]
    object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)

    if use_z:
        loss = loss_func(object_score, clutter_score, z, device=device)
    else:
        loss = loss_func(object_score, clutter_score, device=device)
    return loss.item()


def maxima(arr, d=1):
    coordinates = peak_local_max(arr, min_distance=d, exclude_border=False)
    return coordinates


def get_corr(px_samples, py_samples, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w):
    xv, yv = np.meshgrid(px_samples, py_samples, indexing='ij')
    principal_samples = np.stack([xv, yv], axis=2).reshape(-1, 2)
    principal_samples = np.repeat(np.expand_dims(principal_samples, axis=1), kp_coords.shape[1], axis=1)
    kp_coords_curr = np.repeat(np.expand_dims(kp_coords, axis=1), len(principal_samples), axis=1)  # (num_poses, num_px x num_py, 1024, 2)
    kp_coords_curr += principal_samples
    kp_coords_curr = kp_coords_curr.reshape(-1, kp_coords_curr.shape[2], 2)  # (num_samples, 1024, 2)
    kp_coords_curr[:, :, 0] = np.rint(kp_coords_curr[:, :, 0]/down_sample_rate)
    kp_coords_curr[:, :, 1] = np.rint(kp_coords_curr[:, :, 1]/down_sample_rate)

    kp_vis_curr = np.repeat(np.expand_dims(kp_vis, axis=1), len(principal_samples), axis=1)  # (num_poses, num_px x num_py, 1024)
    kp_vis_curr = kp_vis_curr.reshape(-1, kp_vis_curr.shape[2])
    kp_vis_curr[kp_coords_curr[:, :, 0] < 0] = 0
    kp_vis_curr[kp_coords_curr[:, :, 0] >= hm_w-1] = 0
    kp_vis_curr[kp_coords_curr[:, :, 1] < 0] = 0
    kp_vis_curr[kp_coords_curr[:, :, 1] >= hm_h-1] = 0

    kp_coords_curr[:, :, 0] = np.clip(kp_coords_curr[:, :, 0], 0, hm_w-1)
    kp_coords_curr[:, :, 1] = np.clip(kp_coords_curr[:, :, 1], 0, hm_h-1)
    kp_coords_curr = (kp_coords_curr[:, :, 1:2] * hm_w + kp_coords_curr[:, :, 0:1]).astype(np.int32)  # (num_samples, 1024, 1)


    corr = np.take_along_axis(np.expand_dims(kpt_score_map, axis=0), kp_coords_curr, axis=2)[:, :, 0]
    corr = np.sum(corr * kp_vis_curr, axis=1)
    # corr = np.mean(corr * kp_vis, axis=1)

    return corr


def get_corr_pytorch(px_samples, py_samples, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w, device):
    # l = len(px_samples) // 5
    if len(px_samples) == 31:
        l, n = 3, 11
    elif len(px_samples) == 21:
        l, n = 7, 3

    all_corr = []
    for i in range(n):
        if i != n-1:
            px_s, py_s = torch.from_numpy(px_samples[i*l:i*l+l]).to(device), torch.from_numpy(py_samples).to(device)
        else:
            px_s, py_s = torch.from_numpy(px_samples[i*l:]).to(device), torch.from_numpy(py_samples).to(device)
        kpc = torch.from_numpy(kp_coords).to(device)
        kpv = torch.from_numpy(kp_vis).to(device)
        kps = torch.from_numpy(kpt_score_map).to(device)

        xv, yv = torch.meshgrid(px_s, py_s)
        principal_samples = torch.stack([xv, yv], dim=2).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)

        kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
        kpc += principal_samples
        kpc = kpc.reshape(-1, kpc.shape[2], 2)
        kpc = torch.round(kpc/down_sample_rate)

        kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
        kpv = kpv.reshape(-1, kpv.shape[2])
        kpv[kpc[:, :, 0] < 0] = 0
        kpv[kpc[:, :, 0] >= hm_w-1] = 0
        kpv[kpc[:, :, 1] < 0] = 0
        kpv[kpc[:, :, 1] >= hm_h-1] = 0

        kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w-1)
        kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h-1)
        kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()

        # device = kps.device
        # corr = np.take_along_axis(kps.unsqueeze(0).cpu().numpy(), kpc.cpu().numpy(), axis = 2)[:, :, 0]
        # corr = torch.from_numpy(corr).to(device)
        # ==============
        corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]

        corr = torch.sum(corr * kpv, dim=1)

        all_corr.append(corr.reshape(-1, len(px_s), len(py_s)))
    
    corr = torch.cat(all_corr, dim=-2).detach().cpu().numpy()
    return corr


def get_corr_pytorch_xv_yv(xv, yv, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w, device):
    # px_s, py_s = torch.from_numpy(px_samples).to(device), torch.from_numpy(py_samples).to(device)
    kpc = torch.from_numpy(kp_coords).to(device)
    kpv = torch.from_numpy(kp_vis).to(device)
    kps = torch.from_numpy(kpt_score_map).to(device)

    # xv, yv = torch.meshgrid(px_s, py_s)
    xv, yv = torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
    principal_samples = torch.stack([xv, yv], dim=1).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)

    kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
    kpc += principal_samples
    kpc = kpc.reshape(-1, kpc.shape[2], 2)
    kpc = torch.round(kpc/down_sample_rate)

    kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
    kpv = kpv.reshape(-1, kpv.shape[2])
    kpv[kpc[:, :, 0] < 0] = 0
    kpv[kpc[:, :, 0] >= hm_w-1] = 0
    kpv[kpc[:, :, 1] < 0] = 0
    kpv[kpc[:, :, 1] >= hm_h-1] = 0

    kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w-1)
    kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h-1)
    kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()


    corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]
    # device = kps.device
    # corr = np.take_along_axis(kps.unsqueeze(0).cpu().numpy(), kpc.cpu().numpy(), axis = 2)[:, :, 0]
    # corr = torch.from_numpy(corr).to(device)
    corr = torch.sum(corr * kpv, dim=1).detach().cpu().numpy()

    return corr


def solve_pose_multi_obj(feature_map, inter_module, kp_features, clutter_bank, poses,
                                kp_coords, kp_vis, epochs=300, lr=5e-2, adam_beta_0=0.4,
                                adam_beta_1=0.6, mode='bilinear', loss_type='with_clutter',
                                px_samples=None, py_samples=None, disable_p=False, device='cuda:0',
                                clutter_img_path=None, object_img_path=None, blur_radius=0.0,
                                verbose=False, down_sample_rate=8, hierarchical=0, xv=None, yv=None, fix_theta = True):
    """ Solve object pose with keypoint-based feature pre-rendering.
    Arguments:
    feature_map -- feature map of size (1, C=128, H/8, W/8)
    kp_features -- learned keypoint features of size (K=1024, C=128)
    kp_coords -- keypoint coordinates of size (P=432, K=1024, 2)
    kp_vis -- keypoint visibility of size (P=432, K=1024)
    """

    assert hierarchical == 0

    clutter_img_path = 'clutter.png' if clutter_img_path is None else clutter_img_path
    object_img_path = 'object.png' if object_img_path is None else object_img_path

    time1 = time.time()

    if loss_type == 'without_clutter':
        loss_func = loss_func_type_a
    elif loss_type == 'with_clutter':
        loss_func = loss_func_type_b
    elif loss_type == 'z_map':
        loss_func = loss_func_type_c
    elif loss_type == 'softmax':
        loss_func = loss_func_type_d
    else:
        raise ValueError('Unknown loss function type')
    use_z = (loss_type == 'z_map')

    # Clutter score I: activation score with the center of the clutter features
    # clutter_score = torch.nn.functional.conv2d(feature_map, clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
    # Clutter score II: activate score with the center of two clutter features
    clutter_score = None
    for cb in clutter_bank:
        cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
        if clutter_score is None:
            clutter_score = cs
        else:
            clutter_score = torch.max(clutter_score, cs)
        # print(torch.min(cs), torch.min(clutter_score))

    nkpt, c = kp_features.size()
    # feature_map = feature_map.expand(nkpt, -1, -1, -1)
    memory = kp_features.view(nkpt, c, 1, 1)
    b, c, hm_h, hm_w = feature_map.size()

    if hm_h >= 80 or hm_w >= 56:
        kpt_score_map = np.zeros((nkpt, hm_h, hm_w), dtype=np.float32)
        for i in range(nkpt):
            kpt_score_map[i] = torch.sum(feature_map[0] * memory[i], dim=0).detach().cpu().numpy()
    else:
        kpt_score_map = torch.sum(feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1) # (nkpt, H, W)
        kpt_score_map = kpt_score_map.detach().cpu().numpy()
    kpt_score_map = kpt_score_map.reshape(nkpt, -1)  # (nkpt, H x W)

    # clutter_score_np = clutter_score.detach().cpu().numpy()
    # kpt_score_map = np.maximum(kpt_score_map, clutter_score_np.reshape(-1)) - clutter_score_np.reshape(-1)

    time2 = time.time()

    # Instead of pre-rendering feature maps, we use sparse keypoint features for coarse detection
    if disable_p:
        px_samples = [hm_w*down_sample_rate/2]
        py_samples = [hm_h*down_sample_rate/2]
    else:
        px_samples = np.linspace(0, hm_w*down_sample_rate, 7, endpoint=True) if px_samples is None else px_samples
        py_samples = np.linspace(0, hm_h*down_sample_rate, 7, endpoint=True) if py_samples is None else py_samples
    # max_corr = -1e8
    # max_idx = -1
    # max_principal = None

    if xv is None or yv is None:
        xv, yv = np.meshgrid(px_samples, py_samples, indexing='ij')
        corr = get_corr_pytorch(px_samples, py_samples, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w, device)
        corr = corr.reshape(poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3], len(xv), len(yv))
        corr2d = corr.reshape(-1, len(xv), len(yv))
    else:
        corr = get_corr_pytorch_xv_yv(xv, yv, kpt_score_map, kp_coords, kp_vis, down_sample_rate, hm_h, hm_w, device)
        corr = corr.reshape(poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3], len(xv))
        corr2d = corr.reshape(-1, len(xv))

    corr2d_max = np.max(corr2d, axis=0)
    extrema_2d = maxima(corr2d_max, d=2)
    extrema = []
    for e in extrema_2d:
        c = corr2d[:, e[0], e[1]].reshape(poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3])
        e_azim, e_elev, e_the, e_dist = np.unravel_index(np.argmax(c, axis=None), c.shape)
        # print(corr2d_max[e[0], e[1]])
        if corr2d_max[e[0], e[1]] >= 80.0:
            p = poses[e_azim, e_elev, e_the, e_dist]
            extrema.append({
                'azimuth': p[0],
                'elevation': p[1],
                'theta': p[2],
                'distance': p[3],
                'px': px_samples[e[0]],
                'py': py_samples[e[1]],
                'principal': [px_samples[e[0]], py_samples[e[1]]],
                'corr2d_max': corr2d_max[e[0], e[1]]
            })
    pred = {'pre_render': extrema, 'corr2d': corr2d_max}
    if len(extrema) == 0:
        return {'final': []}

    time3 = time.time()

    refined = []
    object_score_list = []
    seg_map_list = []
    L1 = len(extrema)
    for i in range(len(extrema)):        
        azimuth_pre = extrema[i]['azimuth']
        elevation_pre = extrema[i]['elevation']

        if not fix_theta:
            theta_pre = extrema[i]['theta']
        else:
            theta_pre = 0.0
        distance_pre = extrema[i]['distance']
        px_pre = extrema[i]['px']
        py_pre = extrema[i]['py']
        corr2d_max = extrema[i]['corr2d_max']

        C = camera_position_from_spherical_angles(distance_pre, elevation_pre, azimuth_pre, degrees=False, device=device)
        C = torch.nn.Parameter(C, requires_grad=True)
        theta = torch.tensor(theta_pre, dtype=torch.float32).to(device)
        if not fix_theta:
            theta = torch.nn.Parameter(theta, requires_grad=True)
        else:
            theta = torch.nn.Parameter(theta, requires_grad=False)
        max_principal = [px_pre, py_pre]
        flow = torch.tensor([-(max_principal[0]-hm_w*down_sample_rate/2)/down_sample_rate / 10.0, -(max_principal[1]-hm_h*down_sample_rate/2)/down_sample_rate / 10.0], dtype=torch.float32).to(device)
        flow = torch.nn.Parameter(flow, requires_grad=True)

        if use_z:
            z = torch.nn.Parameter(0.5 * torch.ones((predicted_map.size(0), 1, predicted_map.size(1), predicted_map.size(2)),
                                                    dtype=torch.float32, device=predicted_map.device), requires_grad=True)
        if not fix_theta:
            param_list = [C, theta]
        else:
            param_list = [C]
        if not disable_p:
            param_list.append(flow)
        if use_z:
            param_list.append(z)
        optim = torch.optim.Adam(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
        # optim = torch.optim.Adagrad(params=param_list, lr=lr)
        # optim = torch.optim.AdamW(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)
        
        for epoch in range(epochs):
            projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
            flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
            projected_map = flow_warp(projected_map.unsqueeze(0), flow_map * 10.0)[0]
            object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)

            if use_z:
                loss = loss_func(object_score, clutter_score, z, device=device)
            else:
                loss = loss_func(object_score, clutter_score, device=device)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if (epoch + 1) % 100 == 0:
                scheduler.step(None)
            
            if epoch == 500:
                for g in optim.param_groups:
                    g['lr'] = lr / 5.0

        
        seg_map = ((object_score > clutter_score) * (object_score > 0.0)).detach().cpu().numpy().astype(np.uint8)
        seg_map_list.append(seg_map)
        object_score_list.append(object_score.squeeze().detach().cpu().numpy())
        
        distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
        theta_pred, distance_pred, elevation_pred, azimuth_pred = theta.item(), distance_pred.item(), elevation_pred.item(), azimuth_pred.item()
        px_pred, py_pred = -flow[0].item() * 10.0, -flow[1].item() * 10.0

        refined.append({
            'azimuth': azimuth_pred,
            'elevation': elevation_pred,
            'theta': theta_pred,
            'distance': distance_pred,
            'px': px_pred,
            'py': py_pred,
            'principal': [px_pred * down_sample_rate + hm_w*down_sample_rate / 2, py_pred * down_sample_rate + hm_h*down_sample_rate / 2],
            'corr2d_max': corr2d_max
        })

    
    pred['refined'] = refined

    
    object_score_maps = np.array(object_score_list)
    segmentation_maps = np.array(seg_map_list)
    object_idx_map, new_seg_maps = resolve_occ(segmentation_maps, object_score_maps)
    pred['object_idx_map'] = object_idx_map
    pred['seg_maps'] = new_seg_maps
    pred['object_score_maps'] = object_score_maps

    new_seg_maps = torch.from_numpy(new_seg_maps).float().to(device)
    final = []
    for i in range(len(extrema)):
        azimuth_pre = refined[i]['azimuth']
        elevation_pre = refined[i]['elevation']
        if not fix_theta:
            theta_pre = refined[i]['theta']
        else:
            theta_pre = 0.0
        distance_pre = refined[i]['distance']
        corr2d_max = refined[i]['corr2d_max']
        px_pre = -refined[i]['px'] * 0.1
        py_pre = -refined[i]['py'] * 0.1

        C = camera_position_from_spherical_angles(distance_pre, elevation_pre, azimuth_pre, degrees=False, device=device)
        C = torch.nn.Parameter(C, requires_grad=True)
        theta = torch.tensor(theta_pre, dtype=torch.float32).to(device)
        if not fix_theta:
            theta = torch.nn.Parameter(theta, requires_grad=True)
        else:
            theta = torch.nn.Parameter(theta, requires_grad=False)
        max_principal = [px_pre, py_pre]
        flow = torch.tensor([px_pre, py_pre], dtype=torch.float32).to(device)
        flow = torch.nn.Parameter(flow, requires_grad=True)

        if use_z:
            z = torch.nn.Parameter(0.5 * torch.ones((predicted_map.size(0), 1, predicted_map.size(1), predicted_map.size(2)),
                                                    dtype=torch.float32, device=predicted_map.device), requires_grad=True)

        if not fix_theta:
            param_list = [C, theta]
        else:
            param_list = [C]
        if not disable_p:
            param_list.append(flow)
        if use_z:
            param_list.append(z)
        optim = torch.optim.Adam(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
        # optim = torch.optim.Adagrad(params=param_list, lr=lr)
        # optim = torch.optim.AdamW(params=param_list, lr=lr, betas=(adam_beta_0, adam_beta_1))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.2)

        for epoch in range(epochs):
            projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
            flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
            projected_map = flow_warp(projected_map.unsqueeze(0), flow_map * 10.0)[0]
            object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0) * new_seg_maps[i]

            if use_z:
                loss = loss_func(object_score, clutter_score, z, device=device)
            else:
                loss = loss_func(object_score, clutter_score, device=device)

            loss.backward()
            optim.step()
            optim.zero_grad()

            if (epoch + 1) % 100 == 0:
                scheduler.step(None)
            
            if epoch == 500:
                for g in optim.param_groups:
                    g['lr'] = lr / 5.0
        
        seg_map = ((object_score > clutter_score) * (object_score > 0.0))
        score_map = object_score.squeeze()
        score = torch.sum(seg_map * score_map)
        
        distance_pred, elevation_pred, azimuth_pred = camera_position_to_spherical_angle(C)
        theta_pred, distance_pred, elevation_pred, azimuth_pred = theta.item(), distance_pred.item(), elevation_pred.item(), azimuth_pred.item()
        px_pred, py_pred = -flow[0].item() * 10.0, -flow[1].item() * 10.0
        # print(px_pred, py_pred)
        # print(px_pred * down_sample_rate + hm_w*down_sample_rate / 2, py_pred * down_sample_rate + hm_h*down_sample_rate / 2)

        final.append({
            'azimuth': azimuth_pred,
            'elevation': elevation_pred,
            'theta': theta_pred,
            'distance': distance_pred,
            'px': px_pred,
            'py': py_pred,
            'principal': [px_pred * down_sample_rate + hm_w*down_sample_rate / 2, py_pred * down_sample_rate + hm_h*down_sample_rate / 2],
            'loss': loss.item(),
            'score': score.item(),
            'corr2d_max': corr2d_max
        })

    # print("Prefilter:", time4_2 - time4_1, time4_4 - time4_3, time4 - time3, L1)
    # print("finetune:", time5_4 - time5_1, time5_3 - time5_2, L2)

    
    pred['final'] = final

    return pred


def resolve_occ(seg_maps, score_maps):
    seg_maps_copy = copy.deepcopy(seg_maps)

    """
    obj_idx_map = np.zeros((seg_maps.shape[1], seg_maps.shape[2]), dtype=np.int16)
    obj_idx_map[seg_maps[0] == 1] = 1
    curr_score_map = copy.deepcopy(score_maps[0])
    for i in range(1, seg_maps.shape[0]):
        overlap = (obj_idx_map > 0) * (seg_maps[i] == 1)
    """

    occ_reasoning_mat = np.zeros((seg_maps.shape[0], seg_maps.shape[0]), dtype=np.int16)
    for i in range(0, seg_maps.shape[0]):
        for j in range(0, seg_maps.shape[0]):
            if i >= j:
                continue
            overlap = (seg_maps[i] == 1) * (seg_maps[j] == 1)
            if np.sum(overlap) == 0:
                continue
            score_i = np.sum(overlap * score_maps[i]) + np.sum((seg_maps[i] - overlap) * score_maps[i]) * 0.5
            score_j = np.sum(overlap * score_maps[j]) + np.sum((seg_maps[j] - overlap) * score_maps[j]) * 0.5
            if score_i >= score_j:
                occ_reasoning_mat[i][j] = 1
                occ_reasoning_mat[j][i] = -1
                seg_maps_copy[j][overlap == 1] = 0
            else:
                occ_reasoning_mat[j][i] = 1
                occ_reasoning_mat[i][j] = -1
                seg_maps_copy[i][overlap == 1] = 0
    
    sm = seg_maps_copy[0]
    for i in range(1, seg_maps_copy.shape[0]):
        sm += seg_maps_copy[i]
    # print(np.sum(sm > 1))
    # print([np.sum(seg_maps_copy[i] * score_maps[i]) for i in range(seg_maps_copy.shape[0])])

    seg_maps_pad = np.concatenate([np.zeros((1, seg_maps_copy.shape[1], seg_maps_copy.shape[2]), dtype=seg_maps_copy.dtype), seg_maps_copy], axis=0)
    return np.argmax(seg_maps_pad, axis=0), seg_maps_copy


def get_segmentation_maps(object_score_maps):
    segmentation_maps = np.zeros_like(object_score_maps, dtype=np.uint8)
    for i in range(object_score_maps.shape[0]):
        segmentation_maps[i][object_score_maps[i] > 0.0] = 1
    return segmentation_maps


def save_segmentation_maps(segmentation_maps, path, down_sample_rate = 8):
    b, h, w = segmentation_maps.shape
    m = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(b):
        m[segmentation_maps[i] == 1] = colors[i+1]
    m = cv2.resize(m, (w*down_sample_rate, h*down_sample_rate), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(m).save(path)


def save_object_idx_map(m, dir, idx):
    m = colors[m]
    Image.fromarray(m).save(os.path.join(dir, f'reason_{idx}.png'))
