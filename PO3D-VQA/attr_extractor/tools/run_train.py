import os
import sys
sys.path.append('./')

from options import get_options
from datasets import get_dataloader
from model import get_model
from trainer import get_trainer
import torch
torch.manual_seed(2000)

opt = get_options('train')
print(opt)
train_loader = get_dataloader(opt, 'train')
val_loader = get_dataloader(opt, 'val')
print("build model")
model = get_model(opt)

trainer = get_trainer(opt, model, train_loader, val_loader)

if opt.resume:
    ckpt = torch.load(opt.resume)
    state_dict = ckpt['model_state']
    model.load_state_dict(state_dict)
    print("Load ckpt from", opt.resume)

    trainer.stats['best_val_acc'] = ckpt['best_acc']
    print(trainer.stats['best_val_acc'])



trainer.train()