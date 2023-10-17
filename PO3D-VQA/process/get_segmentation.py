"""
from pred_prob to get the segmentation of each part / object
occlusion / not
"""
import argparse
import json
import os
import ipdb
import sys
import random
import numpy as np
sys.path.append('../6D_pose_estimator')
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftSilhouetteShader,
    look_at_view_transform, SoftPhongShader
)
sys.path.append('../')
from src.utils.plot import *
from src.utils.mesh import *
from tqdm import tqdm
from pycocotools import _mask as coco_mask
from pycocotools import mask as mask_util
import cv2


SUPERCLEVR_COLORS =  ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
SUPERCLEVR_MATERIALS = ['rubber', 'metal']
SUPERCLEVR_SIZES = ['large', 'small']
SUPERCLEVR_SHAPES = ['car', 'suv', 'wagon', 'minivan', 'sedan', 'truck', 'addi', 'bus', 'articulated', 'regular', 'double', 'school', 'motorbike', 'chopper', 'dirtbike', 'scooter', 'cruiser', 'aeroplane', 'jet', 'fighter', 'biplane', 'airliner', 'bicycle', 'road', 'utility', 'mountain', 'tandem']

subcate_to_cate = {
    'jet': 'aeroplane',         'fighter': 'aeroplane',     'airliner': 'aeroplane',    'biplane': 'aeroplane',
    'minivan': 'car',           'suv': 'car',               'wagon': 'car',             'truck': 'car',
    'sedan': 'car',             'cruiser': 'motorbike',     'dirtbike': 'motorbike',    'chopper': 'motorbike',
    'scooter': 'motorbike',     'road': 'bicycle',          'tandem': 'bicycle',        'mountain': 'bicycle',
    'utility': 'bicycle',       'double': 'bus',            'regular': 'bus',           'school': 'bus',
    'articulated': 'bus'
}

CG_part_path = "/home/xingrui/data/CGPart"

with open('/home/xingrui/publish/superclevr_3D_questions/image_generation/data/properties_cgpart.json') as f:
    info = json.load(f)
    

def create_colormap(mask):
    unique_indices = np.unique(mask)
    color_map = {}

    for index in unique_indices:
        color_map[index] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color_map[0] = (0, 0, 0)
    return color_map

def display_colored_mask(mask):
    color_map = create_colormap(mask)
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            colored_mask[i, j] = color_map[mask[i, j]]
    return colored_mask

def load_obj_file(file_path):
    verts, faces, aux = load_obj(file_path)
    x3d = verts.numpy()
    xface = faces.verts_idx.numpy()
    x3d = x3d[:, [0, 2, 1]]
    x3d[:, 1] = -x3d[:, 1]
    
    return x3d, xface

def obj2mesh(mesh_path, color=[1, 0.85, 0.85]):
    x3d, xface = load_obj_file(mesh_path)

    verts = torch.from_numpy(x3d).to('cuda:0')
    verts = pre_process_mesh_pascal(verts)
    faces = torch.from_numpy(xface).to('cuda:0')
    # verts_rgb = torch.ones_like(verts)[None]
    verts_rgb = torch.ones_like(verts)[None] * torch.Tensor(color).view(1, 1, 3).to(verts.device)
    textures = Textures(verts_rgb.to('cuda:0'))
    meshes = Meshes(verts=[verts], faces=[faces], textures=textures)

    return meshes

def load_all_meshes():
    all_meshes = {}
    all_shapes = info['shapes']
    all_shapes.pop('addi')
    for shape_names in all_shapes:
        all_meshes[shape_names] = {}
        super_class, obj_idx = info['shapes'][shape_names].split('/')
        for part_name in info['orig_info_part'][super_class]:
            part_mesh_path = os.path.join(CG_part_path, 'partobjs', super_class, obj_idx, f'{part_name}.obj')
            if not os.path.exists(part_mesh_path):
                continue
            all_meshes[shape_names][part_name] = obj2mesh(part_mesh_path)
        all_meshes[shape_names]['obj'] = obj2mesh(os.path.join(CG_part_path, 'models', super_class, obj_idx, 'models/model_normalized.obj'))
    return all_meshes


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf
    
def get_single_object_mesh(shape_name, azimuth, elevation, theta, distance, principal, \
                  down_sample_rate=8, fuse=True, size=1.0, color=[1, 0.85, 0.85], distance_rescale = 1):
    # idx = shape2index[shape_name]

    h, w, c = 480, 640, 3
    render_image_size = max(h, w)
    crop_size = (h, w)

    cameras = PerspectiveCameras(focal_length=12.0, device='cuda:0')
    raster_settings = RasterizationSettings(
        image_size=render_image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    phong_renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        # shader=HardPhongShader(device='cuda:0', lights=lights, cameras=cameras)
        shader=HardPhongShader(device='cuda:0', cameras=cameras)
    )

    C = camera_position_from_spherical_angles(distance, elevation, azimuth, degrees=False, device='cuda:0')
    R, T = campos_to_R_T(C, theta, device='cuda:0')
    

    # loop among all the parts
    all_images = {}
    for mesh_name in all_meshes[shape_name]:
        meshes = all_meshes[shape_name][mesh_name]

        # render image
        image, depth = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
        # image = image[:, ..., :3]
        depth = depth[0, :, :, 0]
        box_ = bbt.box_by_shape(crop_size, (render_image_size // 2,) * 2)
        bbox = box_.bbox

        # new ----

        depth = depth[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
        depth = torch.squeeze(depth).detach().cpu().numpy()

        cx, cy = principal
        dx = int(-cx + w/2)
        dy = int(-cy + h/2)

        depth_pad = np.pad(depth, ((abs(dy), abs(dy)), (abs(dx), abs(dx)),), mode='edge')
        depth = depth_pad[dy+abs(dy):dy+abs(dy)+depth.shape[0], dx+abs(dx):dx+abs(dx)+depth.shape[1]]

        depth = np.where(depth<0, depth, depth / distance_rescale)
        all_images[mesh_name] = depth 
    # remove the self-occluded mask
    # need to rename to the superclevr setting, TODO

    obj_depth = all_images['obj']
    obj_mask = obj_depth > 0

    mesh_name_list = list(all_images.keys())
    for mesh_name in mesh_name_list:
        if mesh_name == 'obj':
            continue
        visible_mask = (all_images[mesh_name] == obj_depth) * obj_mask
        visible_mask = (np.where(np.abs(all_images[mesh_name] - obj_depth)<1, 1, 0)) * obj_mask
        if visible_mask.sum() < 5:
            all_images.pop(mesh_name, None)
        else:
            all_images[mesh_name] = all_images[mesh_name] * visible_mask
    return all_images

def mask_to_bbox(mask):
    # Assuming the mask is a 2D numpy array with 0-1 values

    if np.max(mask) < 1:
        return [-1,-1, -1, -1]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Create the bounding box as [x_min, y_min, width, height]
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

    return bbox

def get_occlusion(objects):
    scene_depth = []
    for obj in objects:
        new_depth = np.where(obj['mesh']['obj'] > 0, obj['mesh']['obj'],np.inf)
        scene_depth.append(new_depth)

    scene_depth = np.stack(scene_depth, axis = 2)
    scene_index = np.argmin(scene_depth, axis = 2)
    scene_depth = np.min(scene_depth, axis = 2)
    scene_index = np.where(scene_depth != np.inf, scene_index+1, 0)

    # objs mask
    color_index = display_colored_mask(scene_index)

    # plot all parts
    scene_depth = []
    for obj in objects:
        for parts in obj['mesh']:
            if parts != 'obj':
                new_depth = np.where(obj['mesh'][parts] > 0, obj['mesh'][parts],np.inf)
            scene_depth.append(new_depth)
    # print(len(scene_depth))
    scene_depth = np.stack(scene_depth, axis = 2)
    scene_index = np.argmin(scene_depth, axis = 2)
    scene_depth = np.min(scene_depth, axis = 2)
    scene_index = np.where(scene_depth != np.inf, scene_index+1, 0)

    color_index_part = display_colored_mask(scene_index)
    color_index = np.where(color_index_part==0, color_index, color_index_part)

    # cv2.imwrite("scene.png", color_index)

    obj_parts_match = json.load(open('/home/xingrui/vqa/nemo_superclevr_copy/name_match.json'))

    for i, obj in enumerate(objects):
        '''
        obj
            - occlusion
                name: [occlusion value, by which]
            - part
                name: (bbox)

        
        '''
        obj['_parts'] = {}
        obj['parts'] = {}

        if 'obj' not in obj_parts_match[obj['shape_name']]:
            obj_parts_match[obj['shape_name']]['obj'] = ['obj']
        for new_part_name, part_name_list in obj_parts_match[obj['shape_name']].items():
        # for part_name in obj['mesh']:
            mask_before, mask_after = None, None
            for j, part_name in enumerate(part_name_list):
                if part_name not in obj['mesh']:
                    continue

                if j == 0 or mask_before is None:
                    mask_before = (obj['mesh'][part_name]>0).astype(np.uint8)
                    mask_after = (scene_depth == obj['mesh'][part_name]).astype(np.uint8)
                else:
                    new_mask_before = (obj['mesh'][part_name]>0).astype(np.uint8)
                    new_mask_after = (scene_depth == obj['mesh'][part_name]).astype(np.uint8)

                    mask_before = 1 - (1 - mask_before) * (1 - new_mask_before)
                    mask_after = 1 - (1 - mask_after) * (1 - new_mask_after)
            if mask_before is None:
                continue
            mask_occ = mask_before * (1 - mask_after)
            
            bbox_after = mask_to_bbox(mask_after)

            if np.sum(mask_occ)>0:
                by_list = (np.unique(scene_index*mask_occ)-1).tolist()
                if i in by_list:
                    by_list.remove(i)
                if -1 in by_list:
                    by_list.remove(-1)
                if len(by_list) > 0:
                    obj['_parts'][new_part_name] = {'bbox':bbox_after,
                                            'occlusion': int(np.sum(mask_occ)),
                                            'by_list': by_list,
                                        }
                else:
                    obj['_parts'][new_part_name] = {'bbox':bbox_after,
                                            'occlusion': 0,
                                        }  
                                      
            else:
                obj['_parts'][new_part_name] = {'bbox':bbox_after,
                                            'occlusion': 0,
                                    }
            if np.sum(mask_after)>5:
                obj['parts'][new_part_name] = {'bbox':bbox_after}             
        obj['parts']['obj'] = {'bbox':bbox_after}       
        obj['bbox'] = bbox_after       
        obj.pop('mesh')
        # print(obj)
    return objects, color_index

def render_scene(pred_scene, image_dir):
    # img = cv2.imread(os.path.join(image_dir, pred_scene[0]['image_filename']))
    for pred_obj in pred_scene:
        pred_obj['shape_name'] = SUPERCLEVR_SHAPES[np.argmax(pred_obj['shape'])]

        size = SUPERCLEVR_SIZES[np.argmax(pred_obj['size'])]
        distance_rescale = 3.5 / 1.8 if size == 'small' else 1
        
        azimuth = (pred_obj['azimuth']) % (2 * np.pi)
        pred_obj['mesh'] = get_single_object_mesh(pred_obj['shape_name'], azimuth, pred_obj['elevation'], \
                                    pred_obj['theta'], pred_obj['distance'], pred_obj['principal'], distance_rescale = distance_rescale)
    
    pred_scene, color_index = get_occlusion(pred_scene)
    return pred_scene, color_index
    
def handle_non_serializable_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_file', type=str, default = '/home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/anno_prob_1k.json', help='text')
    # parser.add_argument('--anno_file', type=str, default = '/home/xingrui/vqa/FaseRCNN3d_PNSVQA/data/fastRCNN3D/reason/scene_pred-final-0514.json', help='text')
    
    # parser.add_argument('--anno_file', type=str, default = '/home/xingrui/vqa/superclevr-NSVQA/data/pose-nemo/reason/scene_pred-nemo.json', help='text')
    # parser.add_argument('--anno_file', type=str, default = '/home/xingrui/vqa/superclevr-NSVQA/data/pose-nemo/reason/scene_pred-nemo.json', help='text')
    # parser.add_argument('--anno_file', type=str, default = '/home/xingrui/vqa/superclevr-NSVQA/data/superclevr_z_direction/reason/scene_pred-nemo.json', help='text')

    
    
    # parser.add_argument('--output_file', type=str, default = '/home/xingrui/vqa/nemo_superclevr_copy/output/json/0424_15000/anno_prob_part_test.json', help='text')
    # parser.add_argument('--output_file', type=str, default = '/home/xingrui/vqa/superclevr-NSVQA/data/pose-nemo/reason/scene_pred-add_occlusion.json', help='text')
    # parser.add_argument('--output_file', type=str, default = '/home/xingrui/vqa/superclevr-NSVQA/data/superclevr_z_direction/reason/scene_pred-add_occlusion.json', help='text')
    
    parser.add_argument('--image_dir', type=str, default = '/home/xingrui/vqa/super-clevr-gen/output/ver_mask_new/images', help='text')
    # parser.add_argument('--image_dir', type=str, default = '/home/xingrui/publish/superclevr_3D_questions/output/only_plane_in_sky_2/images', help='text')
    
    args = parser.parse_args()

    anno = json.load(open(args.anno_file))
    
    global all_meshes
    all_meshes = load_all_meshes()

    for scene_id in tqdm(anno):
        print(scene_id)
        if int(scene_id) < 31809:
            continue
        # if int(scene_id) > 32809:
        #     continue

        pred_scene = anno[scene_id]
        # import ipdb
        # ipdb.set_trace()
        _, color_index = render_scene(pred_scene, args.image_dir)
        cv2.imwrite('/home/xingrui/vqa/superclevr-NSVQA/data/pose-nemo/visualization/'+f'{int(scene_id):06d}.png', color_index)
    # with open(args.output_file, 'w') as f:
    #     json.dump(anno, f, default=handle_non_serializable_types)






