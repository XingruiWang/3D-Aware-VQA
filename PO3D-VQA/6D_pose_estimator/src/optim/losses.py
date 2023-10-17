import torch


def loss_func_type_a(obj_s, clu_s, device):
    return torch.ones(1, device=device) - torch.mean(obj_s)


def loss_func_type_b(obj_s, clu_s, device):
    return torch.ones(1, device=device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))


def loss_func_type_c(obj_s, clu_s, z, device):
    noise_z = (torch.round(z) - z).detach()
    round_z = z + noise_z
    scores = torch.nn.functional.softmax(torch.cat([obj_s.unsqueeze(0), clu_s.unsqueeze(0)], dim=0), dim=0)
    return torch.ones(1, device=device) - torch.mean(torch.max(scores[0], scores[1]) - scores[1])
    # return torch.ones(1, device=device) - torch.mean(obj_s * scores[0, :] + clu_s * scores[1, :])
    # return torch.ones(1, device=device) - torch.mean(obj_s * scores[0, :] * round_z + clu_s * scores[1, :] * (1-round_z))
    # return torch.ones(1, device=device) - torch.mean(obj_s * z * round_z + clu_s * (1-z) * (1-round_z))
    # return torch.ones(1, device=device) - torch.mean(obj_s * z + clu_s * (1-z))


def loss_func_type_d(obj_s, clu_s, device):
    scores = torch.nn.functional.softmax(torch.cat([obj_s.unsqueeze(0), clu_s.unsqueeze(0)], dim=0), dim=0)
    return torch.ones(1, device=device) - torch.mean(torch.max(scores[0], scores[1]) - scores[1])
