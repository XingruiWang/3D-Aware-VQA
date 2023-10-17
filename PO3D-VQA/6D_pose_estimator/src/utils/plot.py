import BboxTools as bbt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch3d.renderer import OpenGLPerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, PointLights, HardPhongShader, PerspectiveCameras
from pytorch3d.renderer import TexturesVertex as Textures
from pytorch3d.structures import Meshes
import seaborn as sns
import torch
import cv2

from .mesh import load_off, pre_process_mesh_pascal, campos_to_R_T, camera_position_from_spherical_angles
from ..optim import loss_func_type_a, loss_func_type_b, loss_func_type_c, loss_func_type_d
from .flow_warp import flow_warp


# def plot_score_map(score_map, fname, vmin=0.0, vmax=1.0, cbar=False):
#     # ax = sns.heatmap(score_map, square=True, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=cbar)
#     # ax.figure.tight_layout()
#     # ax.figure.savefig(fname, bbox_inches='tight', pad_inches=0.01)
#     plt.imshow(score_map, interpolation='nearest')
#     plt.tight_layout()
#     plt.savefig(fname)

def plot_score_map(score_map, fname, vmin=None, vmax=None, color=True):
    # ax = sns.heatmap(score_map, square=True, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=cbar)
    # ax.figure.tight_layout()
    # ax.figure.savefig(fname, bbox_inches='tight', pad_inches=0.01)
    if vmin is None:
        vmin = np.min(score_map)
    if vmax is None:
        vmax = np.max(score_map)

    normalized_data = np.clip(score_map, vmin, vmax)
    normalized_data = (normalized_data - vmin) / ( vmax- vmin)
    scaled_data = (normalized_data * 255).astype(np.uint8)
    # color_map = cv2.COLORMAP_JET  # You can use other color maps as well
    if color:
        color_map = cv2.COLORMAP_VIRIDIS
        heatmap = cv2.applyColorMap(scaled_data, color_map)
        cv2.imwrite(fname, heatmap)
    else:
        cv2.imwrite(fname, scaled_data)

def alpha_merge_imgs(im1, im2, alpha=1.0):
    mask = np.zeros(im1.shape[0:2], dtype=np.uint8)
    mask[np.sum(im2, axis=2) < 255*3] = int(255 * alpha)
    im2 = np.concatenate((im2, mask.reshape((*im1.shape[0:2], 1))), axis=2)
    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)
    im1.paste(im2, (0, 0), im2)
    return np.array(im1)


def fuse(img, img_list, alpha=0.8):
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    mask_count = np.zeros(img.shape[0:3], dtype=np.uint8)
    im2 = None
    for i in img_list:
        mask[np.sum(i, axis=2) == 255*3] = int(255 * alpha)
        mask_count[np.sum(i, axis=2) > 0, :] += 1
        if im2 is None:
            im2 = i.astype(np.int32)
        else:
            im2 += i.astype(np.int32)
    mask_count[mask_count == 0] = 1
    im2 = im2/mask_count
    im2 = np.concatenate((im2.astype(np.uint8), mask.reshape((*img.shape[0:2], 1))), axis=2)
    im2 = np.clip(np.rint(im2), 0, 255).astype(np.uint8)
    im2 = Image.fromarray(im2)
    img = Image.fromarray(img)
    img.paste(im2, (0, 0), im2)
    return np.array(img)


def plot_mesh(img, mesh_path, azimuth, elevation, theta, distance, principal, down_sample_rate=8, fuse=True, size=1.0, color=[1, 0.85, 0.85]):
    h, w, c = img.shape
    render_image_size = max(h, w)
    crop_size = (h, w)

    # cameras = OpenGLPerspectiveCameras(device='cuda:0', fov=12.0)
    cameras = PerspectiveCameras(focal_length=12.0, device='cuda:0')
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    raster_settings1 = RasterizationSettings(
        image_size=render_image_size // down_sample_rate,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings1
    )
    lights = PointLights(device='cuda:0', location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device='cuda:0', lights=lights, cameras=cameras)
    )

    x3d, xface = load_off(mesh_path)

    x3d = x3d * size
    verts = torch.from_numpy(x3d).to('cuda:0')
    verts = pre_process_mesh_pascal(verts)
    faces = torch.from_numpy(xface).to('cuda:0')
    # verts_rgb = torch.ones_like(verts)[None]
    verts_rgb = torch.ones_like(verts)[None] * torch.Tensor(color).view(1, 1, 3).to(verts.device)
    textures = Textures(verts_rgb.to('cuda:0'))
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device='cuda:0')
    R, T = campos_to_R_T(C, theta, device='cuda:0')
    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
    image = image[:, ..., :3]
    box_ = bbt.box_by_shape(crop_size, (render_image_size // 2,) * 2)
    bbox = box_.bbox
    image = image[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
    image = torch.squeeze(image).detach().cpu().numpy()
    image = np.array((image / image.max()) * 255).astype(np.uint8)

    # cy, cx = principal
    # dx = int(- cx + h/2)
    # dy = int(- cy + w/2)
    cx, cy = principal
    dx = int(-cx + w/2)
    dy = int(-cy + h/2)
    # image = np.roll(image, int(-dx), axis=0)
    # image = np.roll(image, int(-dy), axis=1)
    image_pad = np.pad(image, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode='edge')
    image = image_pad[dy+abs(dy):dy+abs(dy)+image.shape[0], dx+abs(dx):dx+abs(dx)+image.shape[1]]

    if fuse:
        a = 0.8
        mask = (image.sum(2) != 765)[:, :, np.newaxis]
        img = img * (1 - a * mask) + image * a * mask
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)
    else:
        return image

    if fuse:
        get_image = alpha_merge_imgs(img, image)
        # get_image = np.concatenate([raw_img, image], axis=1)
        return get_image
    else:
        return image


def plot_multi_mesh(img, mesh_path, pred, down_sample_rate=8, fuse=True, size=1.0, color=[1, 0.85, 0.85]):
    img_list = [plot_mesh(img, mesh_path, p['azimuth'], p['elevation'], p['theta'], p['distance'], p['principal'], down_sample_rate, fuse=False, size=size, color=color) for p in pred]
    if len(img_list) == 0:
        return img
    s = img_list[0]
    m = np.sum(img_list[0], axis=2) < 255*3
    for i in range(1, len(img_list)):
        mi = (np.sum(img_list[i], axis=2) < 255*3) & (~m)
        s[mi, :] = img_list[i][mi, :]
        m = m | mi
    if fuse:
        # get_image = alpha_merge_imgs(img, s)
        # return get_image
        a = 0.7
        mask = (s.sum(2) != 765)[:, :, np.newaxis]
        img = img * (1 - a * mask) + s * a * mask
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)
    else:
        return s


def plot_loss_landscape(center, feature_map, inter_module, clutter_bank, out_path, use_z=False,
                        down_sample_rate=8, loss_type='with_clutter', mode='bilinear',
                        blur_radius=0.0, device='cuda:0'):
    matplotlib.use('Agg')

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
    use_z = use_z or (loss_type == 'z_map')
    if use_z:
        loss_func = loss_func_type_c
    
    b, c, hm_h, hm_w = feature_map.size()

    azimuth_center = center['azimuth']
    elevation_center = center['elevation']
    theta_center = center['theta']
    distance_center = center['distance']
    px_center, py_center = center['principal']

    if isinstance(feature_map, torch.Tensor):
        predicted_map = feature_map.to(device).squeeze()
    else:
        predicted_map = torch.from_numpy(feature_map).to(device).squeeze()

    if isinstance(clutter_bank, list):
        clutter_score = None
        for cb in clutter_bank:
            cs = torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)
            if clutter_score is None:
                clutter_score = cs
            else:
                clutter_score = torch.max(clutter_score, cs)
    else:
        clutter_score = torch.nn.functional.conv2d(predicted_map.unsqueeze(0), clutter_bank.unsqueeze(2).unsqueeze(3)).squeeze(0).squeeze(0)

    azimuth_shifts = np.linspace(-3.14, 3.14, 121)
    elevation_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
    theta_shifts = np.linspace(-3.14 / 2, 3.14 / 2, 61)
    distance_shifts = np.linspace(-6, 6, 41)
    px_shifts = np.linspace(-40, 40, 81)
    py_shifts = np.linspace(-40, 40, 81)

    # Azimuth curve
    azim_curve = []
    for azim_s in azimuth_shifts:
        C = camera_position_from_spherical_angles(distance_center, elevation_center, azimuth_center+azim_s, degrees=False, device=device)
        theta = torch.tensor(theta_center, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        azim_curve.append(loss.item())
    
    # Elevation curve
    elev_curve = []
    for elev_s in elevation_shifts:
        C = camera_position_from_spherical_angles(distance_center, elevation_center+elev_s, azimuth_center, degrees=False, device=device)
        theta = torch.tensor(theta_center, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        elev_curve.append(loss.item())
    
    # Theta curve
    theta_curve = []
    for theta_s in theta_shifts:
        C = camera_position_from_spherical_angles(distance_center, elevation_center, azimuth_center, degrees=False, device=device)
        theta = torch.tensor(theta_center+theta_s, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        theta_curve.append(loss.item())
    
    # Distance curve
    dist_curve = []
    for dist_s in distance_shifts:
        C = camera_position_from_spherical_angles(distance_center+dist_s, elevation_center, azimuth_center, degrees=False, device=device)
        theta = torch.tensor(theta_center, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        dist_curve.append(loss.item())
    
    # Principal X curve
    px_curve = []
    for px_s in px_shifts:
        C = camera_position_from_spherical_angles(distance_center, elevation_center, azimuth_center, degrees=False, device=device)
        theta = torch.tensor(theta_center, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center+px_s-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        px_curve.append(loss.item())
    
    # Principal Y curve
    py_curve = []
    for py_s in py_shifts:
        C = camera_position_from_spherical_angles(distance_center, elevation_center, azimuth_center, degrees=False, device=device)
        theta = torch.tensor(theta_center, dtype=torch.float32).to(device)
        projected_map = inter_module(C, theta, mode=mode, blur_radius=blur_radius).squeeze()
        flow = torch.tensor([-(px_center-hm_w*down_sample_rate/2)/down_sample_rate, -(py_center+py_s-hm_h*down_sample_rate/2)/down_sample_rate], dtype=torch.float32).to(device)
        flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
        projected_map = flow_warp(projected_map.unsqueeze(0), flow_map)[0]
        object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)
        if use_z:
            loss = loss_func(object_score, clutter_score, object_score, device=device)
        else:
            loss = loss_func(object_score, clutter_score, device=device)
        py_curve.append(loss.item())
    
    plt.figure(figsize=(12, 6))
    plt.plot(azimuth_shifts, azim_curve, label='Azimuth')
    plt.plot(elevation_shifts, elev_curve, label='Elevation')
    plt.plot(theta_shifts, theta_curve, label='Theta')
    plt.plot(distance_shifts/2, dist_curve, label='Distance/2')
    plt.plot(px_shifts/10, px_curve, label='px/10')
    plt.plot(py_shifts/10, py_curve, label='py/10')
    plt.legend()
    plt.xlabel('shifts')
    plt.ylabel('loss')
    # plt.ylim((0.5, 1.0))
    plt.tight_layout()
    plt.savefig(out_path)
