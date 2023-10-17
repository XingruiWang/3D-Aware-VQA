import BboxTools as bbt
import os
import numpy as np

from .calculate_occ import cal_occ_one_image
from .process_camera_parameters import get_anno, Projector3Dto2D, CameraTransformer
from ..utils import load_off, save_off


def normalization(value):
    return (value - value.min()) / (value.max() - value.min())


def box_include_2d(self_box, other):
    return np.logical_and(np.logical_and(self_box.bbox[0][0] <= other[:, 0], other[:, 0] < self_box.bbox[0][1]),
                          np.logical_and(self_box.bbox[1][0] <= other[:, 1], other[:, 1] < self_box.bbox[1][1]))


class MeshLoader(object):
    def __init__(self, path):
        categories = os.listdir(path)

        self.mesh_points_3d = {}
        self.mesh_triangles = {}

        for cate in os.listdir(path):
            self.mesh_points_3d[cate] = {}
            self.mesh_triangles[cate] = {}
            for subcate in os.listdir(os.path.join(path, cate)):
                points_3d, triangles = load_off(os.path.join(path, cate, subcate, 'mesh.off'))
                self.mesh_points_3d[cate][subcate] = points_3d
                self.mesh_triangles[cate][subcate] = triangles
    
    def get(self, cate, subcate):
        return self.mesh_points_3d[cate][subcate], self.mesh_triangles[cate][subcate]


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


class MeshConverter(object):
    def __init__(self, path):
        self.loader = MeshLoader(path=path)

    def get_one(self, annos, return_distance=False):
        cate = annos['cate']
        subcate = annos['subcate']
        
        points_3d, triangles = self.loader.get(cate, subcate)

        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(annos).get_camera_position() #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** .5
        distance_ = normalization(distance)
        h, w = get_anno(annos, 'height', 'width')
        map_size = (h, w)

        if_visible = cal_occ_one_image(points_2d=points_2d, distance=distance_, triangles=triangles, image_size=map_size)
        box_ori = bbt.from_numpy(get_anno(annos, 'box_ori'))
        box_cropped = bbt.from_numpy(get_anno(annos, 'box_obj').astype(np.int))
        box_cropped.set_boundary(get_anno(annos, 'box_obj').astype(np.int)[4::].tolist())

        if_visible = np.logical_and(if_visible, box_include_2d(box_ori, points_2d))
        
        projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)

        pixels_2d = projection_foo(points_2d)

        # handle the case that points are out of boundary of the image
        pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        # print(pixels_2d.shape, box_cropped.boundary, get_anno(annos, 'box_ori'), get_anno(annos, 'box_obj'))
        pixels_2d = np.min([np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1), pixels_2d], axis=0)

        if return_distance:
            return pixels_2d, if_visible, distance_

        return pixels_2d, if_visible


if __name__ == '__main__':
    from PIL import Image, ImageDraw
    name_ = 'n02814533_11997.JPEG'
    anno_path = '../PASCAL3D/annotations/car/'
    converter = MeshConverter()
    pixels, visibile, distance = converter.get_one(np.load(os.path.join(anno_path, name_.split('.')[0] + '.npz'), allow_pickle=True), return_distance=True)
    image = Image.open('../PASCAL3D/images/car/' + name_)

    imd = ImageDraw.ImageDraw(image)

    for p, v in zip(pixels, visibile):
        if v:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        box = bbt.box_by_shape((5, 5), p)
        imd.ellipse(box.pillow_bbox(), fill=color)
    image.show()

    image = Image.open('../PASCAL3D/images/car/' + name_)

    imd = ImageDraw.ImageDraw(image)

    for p, d in zip(pixels, distance):
        color = (0, 255 - int(255 * d), 0)

        box = bbt.box_by_shape((5, 5), p)
        imd.ellipse(box.pillow_bbox(), fill=color)
    image.show()
