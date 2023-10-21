import torch
torch.manual_seed(2000)
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import ipdb

PYTORCH_VER = torch.__version__


# class AttributeNetwork():

#     def __init__(self, opt):    
#         if opt.concat_img:
#             self.input_channels = 6
#         else:
#             self.input_channels = 3

#         if opt.load_checkpoint_path:
#             print('| loading checkpoint from %s' % opt.load_checkpoint_path)
#             checkpoint = torch.load(opt.load_checkpoint_path)
#             if self.input_channels != checkpoint['input_channels']:
#                 raise ValueError('Incorrect input channels for loaded model')
#             self.output_dim = checkpoint['output_dim']
#             self.net = _Net(self.output_dim, self.input_channels)
#             self.net.load_state_dict(checkpoint['model_state'])
#         else:
#             print('| creating new model')
#             output_dims = {
#                 'clevr': 18,
#             }
#             self.output_dim = output_dims[opt.dataset]
#             self.net = _Net(self.output_dim, self.input_channels)

#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)

#         self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
#         self.gpu_ids = opt.gpu_ids
#         if self.use_cuda:
#             self.net.cuda(opt.gpu_ids[0])

#         self.input, self.label = None, None
                
#     def set_input(self, x, y=None):
#         self.input = self._to_var(x)
#         if y is not None:
#             self.label = self._to_var(y)

#     def step(self):
#         self.optimizer.zero_grad()
#         self.forward()
#         self.loss.backward()
#         self.optimizer.step()

#     def forward(self):
#         self.pred = self.net(self.input)
#         if self.label is not None:
#             self.loss = self.criterion(self.pred, self.label)
            
#     def get_loss(self):
#         if PYTORCH_VER.startswith('0.4'):
#             return self.loss.data.item()
#         else:
#             return self.loss.data[0]

#     def get_pred(self):
#         return self.pred.data.cpu().numpy()

#     def eval_mode(self):
#         self.net.eval()

#     def train_mode(self):
#         self.net.train()

#     def save_checkpoint(self, save_path):
#         checkpoint = {
#             'input_channels': self.input_channels,
#             'output_dim': self.output_dim,
#             'model_state': self.net.cpu().state_dict()
#         }
#         torch.save(checkpoint, save_path)
#         if self.use_cuda:
#             self.net.cuda(self.gpu_ids[0])

#     def _to_var(self, x):
#         if self.use_cuda:
#             x = x.cuda()
#         return Variable(x)


class AttributeNet(nn.Module):

    def __init__(self, opt, output_dim, input_channels=3):
        super(AttributeNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        
        # remove the last layer
        layers.pop()

        # change the channel of first layer
        if input_channels != 3:
            layers.pop(0)
            layers.insert(0, nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))

        self.main = nn.Sequential(*layers)

        self.fc_shape = nn.Linear(512, output_dim['shape'])
        self.fc_color = nn.Linear(512, output_dim['color'])
        self.fc_size = nn.Linear(512, output_dim['size'])
        self.fc_material = nn.Linear(512, output_dim['material'])

        self.shape_ce = nn.CrossEntropyLoss()
        self.color_ce = nn.CrossEntropyLoss()
        self.material_ce = nn.CrossEntropyLoss()
        self.size_ce = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        shape = self.fc_shape(x)
        color = self.fc_color(x)
        size = self.fc_size(x)
        material = self.fc_material(x)
        return (shape, color, material, size)

    def loss(self, y, target):
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target
        shape_gt = shape_gt.cuda()
        color_gt = color_gt.cuda()
        material_gt = material_gt.cuda()
        size_gt = size_gt.cuda()

        loss_shape = self.shape_ce(shape, shape_gt)
        loss_color = self.color_ce(color, color_gt)
        loss_material = self.material_ce(material, material_gt)
        loss_size = self.size_ce(size, size_gt)
        
        loss = loss_shape + loss_color + loss_material + loss_size

        return loss

    def accuracy(self, y, target):
        bs = y[0].size(0)
        shape, color, material, size = y
        shape_gt, color_gt, material_gt, size_gt = target

        shape_acc = sum(torch.argmax(shape.data, dim=1) == torch.argmax(shape_gt.cuda(), dim=1)) / bs
        color_acc = sum(torch.argmax(color.data, dim=1) ==  torch.argmax(color_gt.cuda(), dim=1))  / bs
        material_acc = sum(torch.argmax(material.data, dim=1) == torch.argmax(material_gt.cuda(), dim=1))/ bs
        size_acc = sum(torch.argmax(size.data, dim=1) == torch.argmax(size_gt.cuda(), dim=1))/ bs
        
        return (shape_acc, color_acc, material_acc, size_acc)

    def save_checkpoint(self, save_path, best_acc, optimizer):
        checkpoint = {
            'model_state': self.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)



def get_model(opt):
    if opt.type == 'object':
        output_dim = {
            # 'shape': 129,
            'shape': 27,
            'color': 8,
            'material': 2,
            'size': 2,
        }
    elif opt.type == 'part':
        output_dim = {
            'shape': 129,
            # 'shape': 27,
            'color': 8,
            'material': 2,
            'size': 2,
        }
    # model = AttributeNet(opt, output_dim, input_channels=4)
    model = AttributeNet(opt, output_dim, input_channels=3)

    return model

if __name__ == '__main__':
    output_dim = {
        'shape': 129,
        'color': 8,
        'material': 2,
        'size': 2,
    }
    model = get_model(None)

    x = torch.randn((1, 3, 224, 224))

    
    target = (torch.tensor([0]).long(), torch.tensor([3]).long(), torch.tensor([1]).long(), torch.tensor([0]).long())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(10):
        optimizer.zero_grad()
        y = model(x)
        loss = model.loss(y, target)
        print(loss.data)
        loss.backward()
        optimizer.step()
        
