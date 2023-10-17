import BboxTools as bbt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.renderer.mesh.rasterizer import Fragments
import pytorch3d.renderer.mesh.utils as utils
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
)
try:
    from pytorch3d.structures import Meshes, Textures
    use_textures = True
except:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer import TexturesVertex as Textures
    use_textures = False


def create_cuboid_pts(mesh_path):
    points, _ = load_off(mesh_path)
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    zmin, zmax = np.min(points[:, 2]), np.max(points[:, 2])
    pts = [[xmin, ymin, zmin],
           [xmin, ymax, zmin],
           [xmax, ymax, zmin],
           [xmax, ymin, zmin],
           [xmin, ymin, zmax],
           [xmin, ymax, zmax],
           [xmax, ymax, zmax],
           [xmax, ymin, zmax],
           [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]]  # 9x3
    return np.array(pts)


def create_keypoints_pts(mesh_path):
    points, _ = load_off(mesh_path)
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    zmin, zmax = np.min(points[:, 2]), np.max(points[:, 2])
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    zmid = (zmin + zmax) / 2.0
    pts = []
    for x in [xmin, xmid, xmax]:
        for y in [ymin, ymid, ymax]:
            for z in [zmin, zmid, zmax]:
                if x == xmid and y == ymid and z == zmid:
                    continue
                pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def camera_position_to_spherical_angle(camera_pose):
    distance_o = torch.sum(camera_pose ** 2, axis=1) ** .5
    azimuth_o = torch.atan(camera_pose[:, 0] / camera_pose[:, 2]) % np.pi + np.pi * (camera_pose[:, 0] < 0).type(camera_pose.dtype).to(camera_pose.device)
    elevation_o = torch.asin(camera_pose[:, 1] / distance_o)
    return distance_o, elevation_o, azimuth_o


def angel_gradient_modifier(base, grad_=None, alpha=(1.0, 1.0), center_=None):
    # alpha[0]: normal
    # alpha[1]: tangential
    if grad_ is None:
        grad_ = base.grad
        apply_to = True
    else:
        apply_to = False

    if center_ is not None:
        base_ = base.clone() - center_
    else:
        base_ = base

    with torch.no_grad():
        direction = base_ / torch.sum(base_ ** 2, dim=1) ** .5
        normal_vector = torch.sum(direction * grad_, dim=1, keepdim=True) * direction

        tangential_vector = grad_ - normal_vector
        out = normal_vector * alpha[0] + tangential_vector * alpha[1]

    if apply_to:
        base.grad = out

    return out


def decompose_pose(pose, sorts=('distance', 'elevation', 'azimuth', 'theta')):
    return pose[:, sorts.index('distance')], pose[:, sorts.index('elevation')], \
           pose[:, sorts.index('azimuth')], pose[:, sorts.index('theta')]


def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


def standard_loss_func_with_clutter(obj_s: torch.Tensor, clu_s: torch.Tensor):
    clu_s = torch.max(clu_s, dim=1)[0]
    return torch.ones(1, device=obj_s.device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))


def verts_proj(verts, azimuth, elevation, theta, distance, principal=None, M=3000, device='cpu'):
    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device=device)
    R, T = campos_to_R_T(C, theta, device=device)

    return verts_proj_matrix(verts, R, T, principal=principal, M=M)
    #
    # get = verts @ R + T.unsqueeze(1)
    # return principal - torch.cat([get[:, :, 1:2] / get[:, :, 2:3], get[:, :, 0:1] / get[:, :, 2:3]], dim=2) * M


def verts_proj_matrix(verts, R, T, principal, M=3000):
    if len(verts.shape) == 2:
        verts = verts.unsqueeze(0)

    get = verts @ R + T.unsqueeze(1)

    if principal is not None:
        if not isinstance(principal, torch.Tensor):
            principal = torch.Tensor([principal[0], principal[1]]).view(1, 1, 2).type(torch.float32).to(verts.device)
        return principal - torch.cat([get[:, :, 1:2] / get[:, :, 2:3], get[:, :, 0:1] / get[:, :, 2:3]], dim=2) * M
    else:
        return torch.cat([get[:, :, 1:2] / get[:, :, 2:3], get[:, :, 0:1] / get[:, :, 2:3]], dim=2) * M


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float or isinstance(theta, np.floating):
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * float(theta)
    elif isinstance(theta, np.ndarray):
        theta = torch.ones((1, 1, 1)).to(device_) * float(theta)
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if it is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = utils._clip_barycentric_coordinates(
            fragments.bary_coords
        )
        clipped_zbuf = utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments


def translation_matrix(dx, dy, device='cpu'):
    mat = [[1.0, 0.0, dx],
           [0.0, 1.0, dy],
           [0.0, 0.0, 1.0]]
    R = torch.tensor(mat, dtype=torch.float32, device=device)
    return R


def campos_to_R_T_det(campos, theta, dx, dy, device='cpu', at=((0, 0, 0),), up=((0, 1, 0), )):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    # translation = translation_matrix(dx, dy, device=device).unsqueeze(0)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    # R = torch.bmm(translation, R)
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]
    # T = T.T.unsqueeze(0)
    # T = torch.bmm(translation, T)[0].T  # (1, 3)
    return R, T


def campos_to_R_T(campos, theta, device='cpu', at=((0, 0, 0),), up=((0, 1, 0),)):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


# For meshes in PASCAL3D+
def pre_process_mesh_pascal(verts):
    verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    return verts


# Calculate interpolated maps -> [n, c, h, w]
# face_memory.shape: [n_face, 3, c]
def forward_interpolate(R, T, meshes, face_memory, rasterizer, blur_radius=0, mode='bilinear'):
    fragments = rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)

    # [n, h, w, 1, d]
    if mode == 'nearest':
        out_map = utils.interpolate_face_attributes(fragments.pix_to_face, set_bary_coords_to_nearest(fragments.bary_coords), face_memory)
    else:
        out_map = utils.interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_memory)
    
    if out_map.shape[3] > 1:
        pix_to_face, zbuf, bary_coords, dists = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords, fragments.dists

        sigma = blur_radius
        gamma = 0.004
        delta = (dists != -1) * 2.0 - 1.0
        D = torch.sigmoid(delta*dists**2/sigma)
        # print('D', torch.min(D), torch.max(D))
        exp_zbuf = torch.exp(zbuf.double()/gamma)
        # print('zbuf', torch.min(zbuf), torch.max(zbuf))
        # print('exp_zbuf', torch.min(exp_zbuf), torch.max(exp_zbuf))
        w = D.double() * exp_zbuf
        # print('w', torch.min(w), torch.max(w))
        w = w / torch.sum(w * (dists != -1), axis=3, keepdim=True)
        w[dists == -1] = 0.0
        # w2 = w1 * (dists != -1)

        # print('w', torch.min(w), torch.max(w))
        # print(torch.mean(w[dists != -1]), torch.mean(w[dists == -1]))
        # print(torch.sum(w))

        d = torch.sum(dists==-1, dim=3)

        # print('out_map', out_map.shape)
        out_map = torch.sum(out_map * w.unsqueeze(4).float(), axis=3)
        # print(out_map.shape)

    out_map = out_map.squeeze(dim=3).transpose(3, 2).transpose(2, 1)
    return out_map


def set_bary_coords_to_nearest(bary_coords_):
    ori_shape = bary_coords_.shape
    exr = bary_coords_ * (bary_coords_ < 0)
    bary_coords_ = bary_coords_.view(-1, bary_coords_.shape[-1])
    arg_max_idx = bary_coords_.argmax(1)
    # return torch.zeros_like(bary_coords_).scatter(1, arg_max_idx.unsqueeze(1), 1.0).view(*ori_shape) + exr
    nearest_target = torch.zeros_like(bary_coords_).scatter(1, arg_max_idx.unsqueeze(1), 1.0)
    softmax_coords = F.softmax(bary_coords_, dim=1)
    noise = (nearest_target - softmax_coords).detach()
    return (softmax_coords + noise).view(*ori_shape) + exr


def vertex_memory_to_face_memory(memory_bank, faces):
    return memory_bank[faces.type(torch.long)]


def center_crop_fun(out_shape, max_shape):
    box = bbt.box_by_shape(out_shape, (max_shape[0] // 2, max_shape[1] // 2), image_boundary=max_shape)
    return lambda x: box.apply(x)


def meshelize(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn
    print(base_idx, end=' ')

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn
    print(base_idx, end=' ')

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn
    print(base_idx, end=' ')

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn
    print(base_idx, end=' ')

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn
    print(base_idx, end=' ')

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn
    print(base_idx)

    return np.array(out_vertices), np.array(out_faces)
