import os
import json
import torch
import utils
from tqdm import tqdm
from enum import Enum


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)




class Trainer:

    def __init__(self, opt, model, train_loader, val_loader=None):
        self.num_iters = opt.num_iters
        self.num_epochs = opt.num_epochs
        self.run_dir = opt.run_dir
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.double().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.learning_rate )


        self.stats = {
            'train_losses': [],
            'train_losses_ts': [],
            'val_losses': [],
            'val_acc': [],
            'val_losses_ts': [],
            'best_val_loss': 9999,
            'best_val_acc': 0.0,
            'model_t': 0,
        }

    def train(self):
        print('| start training, running in directory %s' % self.run_dir)
        
        epoch = 0

        while epoch < self.num_epochs:

            losses = AverageMeter('train_losses')
            shape_acc = AverageMeter('shape acc')
            color_acc = AverageMeter('color acc')
            material_acc = AverageMeter('material acc')
            size_acc = AverageMeter('size acc')


            epoch += 1
            t = 0
            train_loop = tqdm(self.train_loader)
            for x, label in train_loop:
                t += 1
                self.optimizer.zero_grad()
                x = x.cuda()
                y = self.model(x)
                loss = self.model.loss(y, label)
                loss.backward()
                self.optimizer.step()
                
                accs = self.model.accuracy(y, label)

                losses.update(loss.item())
                shape_acc.update(accs[0].item())
                color_acc.update(accs[1].item())
                material_acc.update(accs[2].item())
                size_acc.update(accs[3].item())

                loss_data = loss.data

                train_loop.set_postfix({'loss': losses.avg, 
                                        'shape acc': shape_acc.avg,
                                        'color acc': color_acc.avg,
                                        'material acc': material_acc.avg,
                                        'size acc': size_acc.avg,
                                        })

                # if t % self.display_every == 0:
                #     self.stats['train_losses'].append(loss_data)
                #     print('| iteration %d / %d, epoch %d, loss %f' % (t, self.num_iters, epoch, loss))
                #     self.stats['train_losses_ts'].append(t)

            if epoch % self.checkpoint_every == 0 or epoch >= self.num_epochs:
                if self.val_loader is not None:    
                    print('| checking validation Accuracy')
                    val_acc = self.evaluate()
                    print('| validation acc %f' % val_acc)
                    if val_acc >= self.stats['best_val_acc']:
                        print('| best model')
                        self.stats['best_val_acc'] = val_acc
                        self.stats['model_t'] = t
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.run_dir, self.stats['best_val_acc'], self.optimizer)
                    self.stats['val_losses'].append(0)
                    self.stats['val_acc'].append(val_acc)
                    self.stats['val_losses_ts'].append(t)
                print('| saving checkpoint')
                self.model.save_checkpoint(os.path.join(self.run_dir, 'checkpoint.pt'), self.stats['best_val_acc'], self.optimizer)
                with open('%s/stats.json' % self.run_dir, 'w') as fout:
                    json.dump(self.stats, fout)

                if t >= self.num_iters:
                    break

    def evaluate(self):
        self.model.eval()

        losses = AverageMeter('train_losses')
        shape_acc = AverageMeter('shape acc')
        color_acc = AverageMeter('color acc')
        material_acc = AverageMeter('material acc')
        size_acc = AverageMeter('size acc')

        loss = 0
        t = 0
        val_loop = tqdm(self.val_loader)
        for x, label in val_loop:
            x = x.cuda()
            y = self.model(x)
            loss_output = self.model.loss(y, label)
            accs = self.model.accuracy(y, label)

            loss += loss_output.item()
            t += 1

            losses.update(loss_output.item())

            shape_acc.update(accs[0].item())
            color_acc.update(accs[1].item())
            material_acc.update(accs[2].item())
            size_acc.update(accs[3].item())

            val_loop.set_postfix({'loss': losses.avg, 
                                    'shape acc': shape_acc.avg,
                                    'color acc': color_acc.avg,
                                    'material acc': material_acc.avg,
                                    'size acc': size_acc.avg,
                                    })
        self.model.train()

        
        # return loss / t if t != 0 else 0
        return (shape_acc.avg + color_acc.avg + material_acc.avg + size_acc.avg) / 4


def get_trainer(opt, model, train_loader, val_loader=None):
    return Trainer(opt, model, train_loader, val_loader)