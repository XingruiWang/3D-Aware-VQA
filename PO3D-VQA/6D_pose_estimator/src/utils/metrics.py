import os

import numpy as np
from scipy.linalg import logm
import torch

from .mesh import create_keypoints_pts
from .pose import cal_rotation_matrix
from .transforms import Transform6DPose

MIN_DIST = 5.0
MIN_ERR = np.pi/2

keypoints = {}
for c in ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike']:
    keypoints[c] = create_keypoints_pts(os.path.join('/home/xingrui/vqa/nemo_superclevr_copy/CAD_cate', c, '01.off'))


def pose_err(gt, pred):
    # return radius
    return ((logm(np.dot(np.transpose(pred), gt)) ** 2).sum()) ** 0.5 / (2. ** 0.5)


def pose_error(gt, pred):
    if pred is None:
        return np.pi
    azimuth_gt, elevation_gt, theta_gt = gt['azimuth'], gt['elevation'], gt['theta']
    dist_gt = gt['distance_resized'] if 'distance_resized' in gt else gt['distance']
    azimuth_pred, elevation_pred, theta_pred, dist_pred = pred['azimuth'], pred['elevation'], pred['theta'], pred['distance']
    anno_matrix = cal_rotation_matrix(theta_gt, elevation_gt, azimuth_gt, dist_gt)
    pred_matrix = cal_rotation_matrix(theta_pred, elevation_pred, azimuth_pred, dist_pred)
    if np.any(np.isnan(anno_matrix)) or np.any(np.isnan(pred_matrix)) or np.any(np.isinf(anno_matrix)) or np.any(np.isinf(pred_matrix)):
        error_ = np.pi
    else:
        error_ = pose_err(anno_matrix, pred_matrix)
    return error_


def voc_ap(rec, prec):
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def add_3d(gt, pred, kps=None, category=None):
    if kps is None:
        assert category is not None
        kps = keypoints[category]
    kps = torch.from_numpy(kps).float()
    gt_trans = Transform6DPose(gt['azimuth'], gt['elevation'], float(gt['theta']), gt['distance'], gt['principal'])
    pred_trans = Transform6DPose(pred['azimuth'], pred['elevation'], float(pred['theta']), pred['distance'], pred['principal'])
    pt1 = gt_trans(kps)
    pt2 = pred_trans(kps)
    d = np.mean(np.sqrt(np.sum((pt1.detach().cpu().numpy() - pt2.detach().cpu().numpy())**2, axis=1)))
    return np.mean(np.sqrt(np.sum((pt1.detach().cpu().numpy() - pt2.detach().cpu().numpy())**2, axis=1)))


def calculate_ap(all_pred, k_to_gts, gt_counter, category):
    sum_ap = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    count_tp = 0
    tp = [0] * len(all_pred)
    fp = [0] * len(all_pred)
    for idx in range(len(all_pred)):
        k = all_pred[idx]['img_name']
        gt_objs = k_to_gts[k]

        if len(gt_objs) == 0:
            fp[idx] = 1
            continue

        d = [add_3d(all_pred[idx], g, category=category) for g in gt_objs]

        gt_match = gt_objs[np.argmin(d)]
        min_d = np.min(d)

        err = pose_error(gt_match, all_pred[idx])

        if min_d < MIN_DIST and err < MIN_ERR:
            if not gt_match['used']:
                tp[idx] = 1
                gt_match['used'] = True
                gt_match['pose_error'] = err
                gt_match['add_3d'] = min_d
                count_tp += 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1
    
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    
    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec, rec, prec


def calculate_ap_2d(all_pred, k_to_gts, gt_counter, category):
    sum_ap = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    count_tp = 0
    tp = [0] * len(all_pred)
    fp = [0] * len(all_pred)
    for idx in range(len(all_pred)):
        k = all_pred[idx]['img_name']
        gt_objs = k_to_gts[k]

        if len(gt_objs) == 0:
            fp[idx] = 1
            continue

        # d = [add_3d(all_pred[idx], g, category=category) for g in gt_objs]
        d = []
        for g in gt_objs:
            d.append(np.sqrt(np.sum((all_pred[idx]['principal'] - g['principal'])**2)))

        gt_match = gt_objs[np.argmin(d)]
        min_d = np.min(d)

        err = pose_error(gt_match, all_pred[idx])

        if min_d < 100.0 and err < MIN_ERR:
            if not gt_match['used']:
                tp[idx] = 1
                gt_match['used'] = True
                gt_match['pose_error'] = err
                gt_match['add_3d'] = min_d
                count_tp += 1
            else:
                # fp[idx] = 1
                tp[idx] = 1
                gt_match['used'] = True
                gt_match['pose_error'] = min(err, gt_match['pose_error'])
                gt_match['add_3d'] = min(min_d, gt_match['add_3d'])
        else:
            fp[idx] = 1
    
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_counter
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    
    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec, rec, prec
