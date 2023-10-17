import math
import numpy as np

from ..models.calculate_occ import cal_occ_one_image
from ..utils import load_off


def pre_compute_kp_coords(mesh_path, mesh_face_breaks, azimuth_samples=None, elevation_samples=None, theta_samples=None,
                          distance_samples=None, viewport=3000):
    """ Calculate vertex visibility for cuboid models.
    """
    xvert, _ = load_off(mesh_path)

    xmin, xmax = np.min(xvert[:, 0]), np.max(xvert[:, 0])
    ymin, ymax = np.min(xvert[:, 1]), np.max(xvert[:, 1])
    zmin, zmax = np.min(xvert[:, 2]), np.max(xvert[:, 2])
    xmean = (xmin + xmax) / 2
    ymean = (ymin + ymax) / 2
    zmean = (zmin + zmax) / 2
    pts = [[xmean, ymean, zmin],
           [xmean, ymean, zmax],
           [xmean, ymin, zmean],
           [xmean, ymax, zmean],
           [xmin, ymean, zmean],
           [xmax, ymean, zmean]]

    azimuth_samples = np.linspace(0, np.pi*2, 12, endpoint=False) if azimuth_samples is None else azimuth_samples
    elevation_samples = np.linspace(-np.pi/6, np.pi/3, 4) if elevation_samples is None else elevation_samples
    theta_samples = np.linspace(-np.pi/6, np.pi/6, 3) if theta_samples is None else theta_samples
    # dist_samples = np.linspace(4, 30, 9, endpoint=True) if distance_samples is None else distance_samples
    dist_samples = np.linspace(4, 6, 3, endpoint=True) if distance_samples is None else distance_samples

    poses = np.zeros((len(azimuth_samples)*len(elevation_samples)*len(theta_samples)*len(dist_samples), 4), dtype=np.float32)
    num_vis_faces = []
    count = 0
    for azim_ in azimuth_samples:
        for elev_ in elevation_samples:
            for theta_ in theta_samples:
                for dist_ in dist_samples:
                    poses[count] = [azim_, elev_, theta_, dist_]
                    count += 1
                    if elev_ == 0:
                        if azim_ in [0, np.pi/2, np.pi, 3*np.pi/2]:
                            num_vis_faces.append(1)
                        else:
                            num_vis_faces.append(2)
                    else:
                        if azim_ in [0, np.pi/2, np.pi, 3*np.pi/2]:
                            num_vis_faces.append(2)
                        else:
                            num_vis_faces.append(3)

    kp_coords = np.zeros((len(azimuth_samples)*len(elevation_samples)*len(theta_samples)*len(dist_samples), len(xvert), 2), dtype=np.float32)
    kp_vis = np.zeros((len(azimuth_samples)*len(elevation_samples)*len(theta_samples)*len(dist_samples), len(xvert)), dtype=np.float32)
    xvert_ext = np.concatenate((xvert, pts), axis=0)
    for i, pose_ in enumerate(poses):
        azim_, elev_, theta_, dist_ = pose_

        C = np.zeros((3, 1))
        C[0] = dist_ * math.cos(elev_) * math.sin(azim_)
        C[1] = -dist_ * math.cos(elev_) * math.cos(azim_)
        C[2] = dist_ * math.sin(elev_)
        azimuth = -azim_
        elevation = - (math.pi / 2 - elev_)
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
        P = np.array([[viewport, 0, 0],
                        [0, viewport, 0],
                        [0, 0, -1]])
        x3d_ = np.hstack((xvert_ext, np.ones((len(xvert_ext), 1)))).T
        x3d_ = np.dot(R, x3d_)
        # x3d_r_ = np.dot(P, x3d_)
        x2d = np.dot(P, x3d_)
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        x2d = x2d[0:2, :]
        R2d = np.array([[math.cos(theta_), -math.sin(theta_)],
                        [math.sin(theta_), math.cos(theta_)]])
        x2d = np.dot(R2d, x2d).T
        x2d[:, 1] *= -1

        # principal = np.array([px_, py_], dtype=np.float32)
        # x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

        x2d = x2d[:len(xvert)]
        kp_coords[i] = x2d

        center3d = x3d_[:, len(xvert):]
        face_dist = np.sqrt(np.square(center3d[0, :]) + np.square(center3d[1, :]) + np.square(center3d[2, :]))
        ind = np.argsort(face_dist)[:num_vis_faces[i]]

        # 13 13 5 13x12 + 13x12 + 5x12 + 5x12 + 5x13 + 5x13
        # 8 17 6 17x8 + 17x8 + 6x8 + 6x8 + 6x17 + 6x17
        if 0 in ind:
            kp_vis[i, 0:mesh_face_breaks[0]] = 1
        if 1 in ind:
            kp_vis[i, mesh_face_breaks[0]:mesh_face_breaks[1]] = 1
        if 2 in ind:
            kp_vis[i, mesh_face_breaks[1]:mesh_face_breaks[2]] = 1
        if 3 in ind:
            kp_vis[i, mesh_face_breaks[2]:mesh_face_breaks[3]] = 1
        if 4 in ind:
            kp_vis[i, mesh_face_breaks[3]:mesh_face_breaks[4]] = 1
        if 5 in ind:
            kp_vis[i, mesh_face_breaks[4]:mesh_face_breaks[5]] = 1

    return poses, kp_coords, kp_vis


def pre_compute_kp_coords2(mesh_path, image_h, image_w, azimuth_samples=None, elevation_samples=None, theta_samples=None,
                           distance_samples=None, viewport=3000):
    """ Calculate vertex visibility for any mesh model.
    """
    xvert, xface = load_off(mesh_path)
    # xvert[:, 1] = -xvert[:, 1]
    # xvert = xvert[:, [0, 2, 1]]

    azimuth_samples = np.linspace(0, np.pi*2, 12, endpoint=False) if azimuth_samples is None else azimuth_samples
    elevation_samples = np.linspace(-np.pi/6, np.pi/3, 4) if elevation_samples is None else elevation_samples
    theta_samples = np.linspace(-np.pi/6, np.pi/6, 3) if theta_samples is None else theta_samples
    # dist_samples = np.linspace(4, 30, 9, endpoint=True) if distance_samples is None else distance_samples
    dist_samples = np.linspace(4, 6, 3, endpoint=True) if distance_samples is None else distance_samples

    poses = np.zeros((len(azimuth_samples)*len(elevation_samples)*len(theta_samples)*len(dist_samples), 4), dtype=np.float32)
    count = 0
    for azim_ in azimuth_samples:
        for elev_ in elevation_samples:
            for theta_ in theta_samples:
                for dist_ in dist_samples:
                    poses[count] = [azim_, elev_, theta_, dist_]
                    count += 1

    kp_coords = np.zeros((len(azimuth_samples)*len(elevation_samples)*len(theta_samples)*len(dist_samples), len(xvert), 2), dtype=np.float32)
    kp_vis = np.zeros((len(azimuth_samples)*len(elevation_samples), len(xvert)), dtype=np.float32)
    for i, pose_ in enumerate(poses):
        azim_, elev_, theta_, dist_ = pose_

        C = np.zeros((3, 1))
        C[0] = dist_ * math.cos(elev_) * math.sin(azim_)
        C[1] = -dist_ * math.cos(elev_) * math.cos(azim_)
        C[2] = dist_ * math.sin(elev_)
        azimuth = -azim_
        elevation = - (math.pi / 2 - elev_)
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
        P = np.array([[viewport, 0, 0],
                        [0, viewport, 0],
                        [0, 0, -1]])
        x3d_ = np.hstack((xvert, np.ones((len(xvert), 1)))).T
        x3d_ = np.dot(R, x3d_)
        # x3d_r_ = np.dot(P, x3d_)
        x2d = np.dot(P, x3d_)
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        x2d = x2d[0:2, :]
        R2d = np.array([[math.cos(theta_), -math.sin(theta_)],
                        [math.sin(theta_), math.cos(theta_)]])
        x2d = np.dot(R2d, x2d).T
        x2d[:, 1] *= -1

        # principal = np.array([px_, py_], dtype=np.float32)
        # x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

        kp_coords[i] = x2d

    for i in range(len(azimuth_samples)*len(elevation_samples)):
        azim_, elev_, theta_, dist_ = poses[i*len(theta_samples)*len(dist_samples)]
        
        x2d = kp_coords[i]

        x2d = np.rint(x2d).astype(np.int32)
        distance = np.sum((xvert - C.transpose((1, 0)))**2, axis=1)**0.5
        distance = (distance - distance.min()) / (distance.max() - distance.min())

        x2d_occ = x2d.copy()
        x2d_occ[:, 0] += image_w*2
        x2d_occ[:, 1] += image_h*2
        kpvis = cal_occ_one_image(points_2d=x2d_occ, distance=distance, triangles=xface, image_size=(image_h*4, image_w*4))

        kp_vis[i] = kpvis
    
    kp_vis = kp_vis[:, np.newaxis, :].repeat(len(theta_samples)*len(dist_samples), axis=1).reshape(-1, len(xvert))

    return poses, kp_coords, kp_vis
