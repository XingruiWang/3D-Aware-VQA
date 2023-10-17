import math

import numpy as np
from scipy.optimize import least_squares

from .mesh import create_cuboid_pts


def get_rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


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


def notation_blender_to_pyt3d(mesh_path, sample, anno, image_h, image_w):
    # xvert, _ = load_off(mesh_path)
    xvert_orig = create_cuboid_pts(mesh_path)
    xvert = xvert_orig * sample['size_r']

    rot_mat = get_rot_z(sample['theta'] / 180. * math.pi)
    xvert = (rot_mat @ xvert.transpose((1, 0))).transpose((1, 0))
    xvert = xvert + sample['location']
    xvert = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    xvert = (anno['mw_inv'] @ xvert)[:3].transpose((1, 0))
    xvert = np.concatenate([xvert.transpose((1, 0)), np.ones((1, len(xvert)), dtype=np.float32)], axis=0)
    pts_2d = anno['proj_mat'] @ xvert
    pts_2d[0, :] = pts_2d[0, :] / pts_2d[3, :] * (image_w//2) + (image_w//2)
    pts_2d[1, :] = -pts_2d[1, :] / pts_2d[3, :] * (image_h//2) + (image_h//2)
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
    
    pose = {
        'azimuth': min_x[0],
        'elevation': min_x[1],
        'theta': min_x[2],
        'distance': min_x[3],
        'principal': np.array([min_x[4]*448, min_x[5]*320])
    }
    return pose, pts_2d, project(min_x, xvert_orig, pts_2d)
