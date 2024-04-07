import _init_paths

import argparse
from datetime import date, datetime
import logging
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
# from yan.exp import Logger

from src.datasets import SuperCLEVRTrain, ToTensor, Normalize, InfiniteSampler
from src.models import NearestMemoryManager, NetE2E, mask_remove_near
from src.utils import str2bool, load_off


def parse_args():
    parser = argparse.ArgumentParser(description='Train 6D NeMo on SuperCLEVR')

    # General args
    parser.add_argument('--exp_name', type=str, default='oct02_superclevr_car')
    parser.add_argument('--category', type=str, default='car')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/home/wufeim/nemo_superclevr/experiments')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    # Model args
    parser.add_argument('--backbone', type=str, default='resnetext')
    parser.add_argument('--d_feature', type=int, default=128)
    parser.add_argument('--local_size', type=int, default=1)
    parser.add_argument('--separate_bank', type=str2bool, default=False)
    parser.add_argument('--max_group', type=int, default=512)

    # Data args
    parser.add_argument('--mesh_path', type=str, default='CAD_cate')
    parser.add_argument('--dataset_path', type=str, default='/home/wufeim/nemo_superclevr/superclevr/superclevr')
    parser.add_argument('--partial_train', type=float, default=1.0)
    parser.add_argument('--filename_prefix', type=str, default='superCLEVR')
    parser.add_argument('--workers', type=str, default=4)

    # Training args
    parser.add_argument('--pretrain', type=str2bool, default=True)
    parser.add_argument('--iterations', type=int, default=25000)
    parser.add_argument('--log_itr', type=int, default=50)
    parser.add_argument('--save_itr', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--update_lr_itr', type=int, default=15000)
    parser.add_argument('--update_lr_ratio', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.92)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--train_accumulate', type=int, default=10)
    parser.add_argument('--distance_thr', type=int, default=48)
    parser.add_argument('--weight_noise', type=float, default=0.005)
    parser.add_argument('--num_noise', type=int, default=5)
    parser.add_argument('--T', type=float, default=0.07)
    parser.add_argument('--adj_momentum', type=float, default=0.96)

    args = parser.parse_args()

    args.mesh_path = os.path.join(args.mesh_path, args.category, '01.off')
    args.train_img_path = os.path.join(args.dataset_path, 'new', 'images')
    args.train_anno_path = os.path.join(args.dataset_path, 'new', 'annotations')

    return args


def prepare_data(args):
    train_transform = transforms.Compose([ToTensor(), Normalize()])
    train_dataset = SuperCLEVRTrain(
        img_path=args.train_img_path,
        anno_path=args.train_anno_path,
        prefix=f'{args.filename_prefix}_new',
        category=args.category,
        transform=train_transform,
        enable_cache=False,
        partial=args.partial_train
    )

    train_sampler = InfiniteSampler(dataset=train_dataset, shuffle=True, seed=args.seed, window_size=0.5)
    train_iterator = iter(DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers))

    val_dataset, val_iterator = None, None

    return train_dataset, train_iterator, val_dataset, val_iterator


def train_one_step(net, memory_bank, train_iterator, criterion, optimizer, args, itr):
    sample = next(train_iterator)
    img, kp, kpvis, obj_mask, distance = sample['img'], sample['kp'], sample['kpvis'], sample['obj_mask'], sample['distance']
    img, kp, kpvis, obj_mask, distance = img.cuda(), kp.cuda(), kpvis.cuda(), obj_mask.cuda(), distance.cuda()
    num_objs = sample['num_objs']

    kp = kp[:, :, [1, 0]]

    y_num = args.n
    index = torch.Tensor([[k for k in range(y_num)]] * img.shape[0])
    index = index.cuda()
    import ipdb
    ipdb.set_trace()
    # obj_mask, kp
    features = net.forward(img, keypoint_positions=kp, obj_mask=1-obj_mask.float())

    get, y_idx, noise_sim = memory_bank(features, index, kpvis)
    get /= args.T
    mask_distance_legal = mask_remove_near(kp, thr=args.distance_thr*5.0/distance, num_neg=args.num_noise * args.max_group,
                                            dtype_template=get, neg_weight=args.weight_noise)
    
    kpvis = kpvis.type(torch.bool).to(kpvis.device)
    loss = criterion(((get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2])))[kpvis.view(-1), :],
                        y_idx.view(-1)[kpvis.view(-1)])
    loss = torch.mean(loss)

    loss_main = loss.item()
    if args.num_noise > 0 and True:
        loss_reg = torch.mean(noise_sim) * 0.1
        loss += loss_reg
    else:
        loss_reg = torch.zeros(1)
    loss.backward()

    if itr % args.train_accumulate == 0:
        optimizer.step()
        optimizer.zero_grad()

    return {'loss': loss.item(), 'loss_main': loss_main, 'loss_reg': loss_reg.item(), 'lr': optimizer.param_groups[0]['lr']}


def main():
    args = parse_args()

    wandb.init(project='superclevr_nemo')
    wandb.config = vars(args)
    wandb.run.name = f'{args.exp_name}_{wandb.run.id}'
    wandb.run.save()

    # yan = Logger('path')
    msg = [f'Experiment {args.exp_name} has finished.',
           'The experiment arguments are summarized below:',
           str(vars(args)),
           'The experiment log is shown below:',
           '']

    save_path = os.path.join(args.save_path, args.exp_name)
    ckpt_path = os.path.join(save_path, 'ckpts')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    shutil.copyfile(__file__, os.path.join(save_path, os.path.basename(__file__)))
    wandb.save(__file__)

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(save_path, 'log.txt'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    net = NetE2E(net_type=args.backbone, local_size=[args.local_size, args.local_size], output_dimension=args.d_feature,
                 reduce_function=None, n_noise_points=args.num_noise, pretrain=args.pretrain, noise_on_mask=True)
    logging.info(f'num params {sum(p.numel() for p in net.net.parameters())}')
    net = nn.DataParallel(net).cuda().train()

    args.n = load_off(args.mesh_path)[0].shape[0]
    memory_bank = NearestMemoryManager(inputSize=args.d_feature, outputSize=args.n+args.num_noise*args.max_group,
                                        K=1, num_noise=args.num_noise, num_pos=args.n, momentum=args.adj_momentum)
    memory_bank = memory_bank.cuda()

    criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dataset, train_iterator, _, _ = prepare_data(args)
    logging.info(f'found {len(train_dataset)} for training ({args.category})')

    train_dataset.debug(0)

    logging.info('Start training:')
    logging.info(f'Experiment:     {args.exp_name}')
    logging.info(f'Category:       {args.category}')
    logging.info(f'Num imgs train: {len(train_dataset)}')
    logging.info(f'Total itr:      {args.iterations}')
    logging.info(f'LR:             {args.lr}')
    logging.info(f'Update LR itr:  {args.update_lr_itr}')
    logging.info(f'Updated LR:     {args.lr * args.update_lr_ratio}')

    log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []

    for itr in range(args.iterations):
        train_log_dict = train_one_step(net, memory_bank, train_iterator, criterion, optim, args, itr)
        log_train_loss.append(train_log_dict['loss'])
        log_train_loss_main.append(train_log_dict['loss_main'])
        log_train_loss_reg.append(train_log_dict['loss_reg'])

        if itr == 0 or (itr+1) % args.log_itr == 0:
            train_loss, train_loss_main, train_loss_reg = np.mean(log_train_loss), np.mean(log_train_loss_main), np.mean(log_train_loss_reg)
            if itr > 0:
                log_train_loss, log_train_loss_main, log_train_loss_reg = [], [], []
            wandb.log({'lr': train_log_dict['lr'],
                       'train_loss': train_loss,
                       'train_loss_main': train_loss_main,
                       'train_loss_reg': train_loss_reg})
            logging.info(f'[Itr {itr+1}] lr={train_log_dict["lr"]} train_loss={train_loss:.5f} train_loss_main={train_loss_main:.5f} train_loss_reg={train_loss_reg:.5f}')
        
        if itr == 0 or (itr+1) % args.save_itr == 0:
            ckpt = {}
            ckpt['state'] = net.state_dict()
            ckpt['memory'] = memory_bank.memory
            ckpt['timestamp'] = int(datetime.timestamp(datetime.now()))
            ckpt['args'] = vars(args)
            ckpt['step'] = itr+1
            ckpt['lr'] = train_log_dict["lr"]
            torch.save(ckpt, os.path.join(save_path, 'ckpts', f'saved_model_{itr+1}.pth'))
            msg[-1] += f'<br>[Itr {itr+1}] lr={train_log_dict["lr"]} train_loss={train_loss:.5f} train_loss_main={train_loss_main:.5f} train_loss_reg={train_loss_reg:.5f}'
        
        if (itr+1) >= args.update_lr_itr:
            lr = args.lr * args.update_lr_ratio
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            logging.info(f'update learning rate: {args.lr} -> {lr}')
    
    # state, msg = yan.send_message(
    #     f'Experiment {args.exp_name} completed!',
    #     msg
    # )


if __name__ == '__main__':
    main()
