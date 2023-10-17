import numpy as np
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Translate, Rotate
import torch


class Transform6DPose():
    def __init__(self, azimuth, elevation, theta, distance, principal, img_size=(320, 448), focal_length=3000, device='cpu'):
        R, T = look_at_view_transform([distance], [elevation], [azimuth], degrees=False, device=device)
        self.R = torch.bmm(R, self.rotation_theta(theta, device_=device))
        self.T = T + self.convert_principal_to_translation(distance, principal, img_size, focal_length).to(device)

    def rotation_theta(self, theta, device_=None):
        # cos -sin  0
        # sin  cos  0
        # 0    0    1
        if type(theta) == float:
            if device_ is None:
                device_ = 'cpu'
            theta = torch.ones((1, 1, 1)).to(device_) * theta
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
    
    def convert_principal_to_translation(self, distance, principal_, image_size_, focal_=3000):
        principal_ = np.array(principal_, dtype=np.float32)
        d_p = torch.Tensor(principal_).float() - torch.Tensor(image_size_).flip(0) / 2
        return torch.Tensor([[-d_p[0] * distance / focal_, -d_p[1] * distance / focal_, 0]])

    def __call__(self, points):
        T_ = Translate(self.T, device=self.T.device)
        R_ = Rotate(self.R, device=self.R.device)
        transforms = R_.compose(T_)
        return transforms.transform_points(points)
