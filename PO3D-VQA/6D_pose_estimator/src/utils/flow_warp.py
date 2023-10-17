import torch

Backward_tensorGrid = [{} for _ in range(9)]


def torch_warp(tensorInput, tensorFlow):
    device_id = -1 if tensorInput.device == torch.device('cpu') else tensorInput.device.index
    if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
        N, _, H, W = tensorFlow.size()
        tensorHorizontal = torch.linspace(-1.0, 1.0, W, device=tensorInput.device).view(
            1, 1, 1, W).expand(N, -1, H, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, H, device=tensorInput.device).view(
            1, 1, H, 1).expand(N, -1, -1, W)
        Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
            [tensorHorizontal, tensorVertical], 1)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
    return torch.nn.functional.grid_sample(input=tensorInput,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp
