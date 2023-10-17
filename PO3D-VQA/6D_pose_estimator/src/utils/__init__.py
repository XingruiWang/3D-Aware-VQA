from .blender import notation_blender_to_pyt3d
from .flow_warp import flow_warp
from .mesh import create_cuboid_pts, load_off, save_off, camera_position_to_spherical_angle, angel_gradient_modifier, \
    decompose_pose, normalize, standard_loss_func_with_clutter, verts_proj, verts_proj_matrix, rotation_theta, \
    rasterize, translation_matrix, campos_to_R_T_det, campos_to_R_T, forward_interpolate, set_bary_coords_to_nearest, \
    vertex_memory_to_face_memory, center_crop_fun, pre_process_mesh_pascal, create_keypoints_pts, meshelize
from .metrics import pose_error, pose_err, calculate_ap, add_3d, calculate_ap_2d
from .plot import (
    plot_score_map,
    plot_mesh,
    plot_multi_mesh,
    fuse,
    alpha_merge_imgs,
    plot_loss_landscape
)
from .pose import (
    get_transformation_matrix,
    rotation_theta,
    cal_rotation_matrix
)
from .transforms import Transform6DPose
from .utils import str2bool, EasyDict, load_off, save_off, normalize_features, keypoint_score

MESH_FACE_BREAKS_1000 = {
    'car': [250, 500, 590, 680, 905, 1130],
    'motorbike': [144, 288, 366, 444, 756, 1068],
    'bus': [210, 420, 483, 546, 816, 1086],
    'bicycle': [154, 308, 594, 880, 971, 1062],
    'aeroplane': [289, 578, 714, 850, 986, 1122]
}

subcate_to_cate = {
    'jet': 'aeroplane',         'fighter': 'aeroplane',     'airliner': 'aeroplane',    'biplane': 'aeroplane',
    'minivan': 'car',           'suv': 'car',               'wagon': 'car',             'truck': 'car',
    'sedan': 'car',             'cruiser': 'motorbike',     'dirtbike': 'motorbike',    'chopper': 'motorbike',
    'scooter': 'motorbike',     'road': 'bicycle',          'tandem': 'bicycle',        'mountain': 'bicycle',
    'utility': 'bicycle',       'double': 'bus',            'regular': 'bus',           'school': 'bus',
    'articulated': 'bus'
}
