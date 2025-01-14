import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

import datetime
from collections import OrderedDict
import os
import itertools

import argparse
import time
import yaml
from nwtk.utils import create_log, seed_everything

from timm import utils
from timm.models import load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs

from torchmetrics.functional import r2_score

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('--data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--batch-size', type=int, default=32, metavar='N',
                   help='Input batch size for training (default: 32)')
group.add_argument('--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
group.add_argument('--lr', type=float, default=0.001, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
#parser.add_argument('--model', default='spectrumdavit', type=str, metavar='MODEL',
                    #help='Name of model to train (default: "countception"')
parser.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "loss"')
parser.add_argument('--work-dir', default='', type=str, metavar='DIR',
                   help='Path to load the model')
parser.add_argument('--load-epoch', default='', metavar='N',
                   help='Which epoch to load')

# Model parameters
group.add_argument('--in-chans', type=int, default=1, metavar='N',
                   help='Image input channels (default: for spectrum == 1)')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--drop', type=float, default=0.3, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-path', type=float, default=0.3, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--depths', type=int, default=[3, 1], nargs='*', metavar='*N',
                   help='Model depths')
group.add_argument('--embed-dims', type=int, default=[384, 192], nargs='*', metavar='*N',
                   help='Model embed dimentions')
group.add_argument('--window-size', type=int, default=70, metavar='N',
                   help='Model depths')
group.add_argument('--num-heads', type=int, default=[12, 3], nargs='*', metavar='*N',
                   help='Numbers of attention heads, must able to be divided by embed dimentions')


# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument("--local-rank", default=2, type=int)

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    #args = parser.parse_args()
    #with open("config.yaml", "w") as f:
        #yaml.safe_dump(args.__dict__, f)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
    

class MyDataset(Dataset):
    def __init__(self, datadic):
        self.data = torch.load(datadic)
        self.norm = torch.norm(self.data[:,-1])
    def __getitem__(self, idx):
        tcd = self.data[idx, -1]
        dos = self.data[idx,:-1]
        return dos, tcd / self.norm

    def __len__(self):
        return self.data.size()[0]

def mask_data(src, length): #only for 2D tensor,
    C, L = src.shape
    i = 0
    while True:
        if i < L:
            yield torch.zeros(C, L).scatter_(0, torch.arange(0,C).repeat(i).view(-1,C).permute(1,0), src)
            i = i + length
        else:
            yield src
            break
            
def batch_locpad(src, loc, target, device):
    N, C, L = src.shape
    assert N, C == target.shape[:2]
    length = target.shape[-1]
    src = src.view(-1, L).to(device)
    
    return torch.cat((src[..., :loc], src[...,loc:].scatter_(0, torch.arange(0,N *C).repeat(length).view(-1,N *C).permute(1,0).to(device), target.view(-1, length))), dim = 1).view(N, C, L)

def rmse(x1 ,x2):
    return x1.add(-x2).pow(2).mean().sqrt()
        
def compute_r2score_and_loss(model, train_loader, loss_fn, device):

    currnet_loss = utils.AverageMeter()
    r2score = utils.AverageMeter()
    
    for features, targets in train_loader:
        features = features.to(device)
        targets = targets.to(device)
        output = model(features)
        currnet_loss.update(loss_fn(output.squeeze(), targets))
            
        r2score.update(r2_score(output.squeeze(), targets))

        
    return currnet_loss.avg , r2score.avg

def r2_score_loss(feature, target):
    return 1 - r2_score(feature, target)
    
def main():
    args, _ = _parse_args()

    seed_everything(args.seed)

    from regress_model import nn_model
    model = nn_model.NNModel(num_features = 4096)
    
    args.model = model.__class__.__name__
    
    args.load = None
    if not args.work_dir:
        args.work_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + args.model
        os.mkdir(args.work_dir)
        _logger = create_log(os.path.join(args.work_dir, '{}.log'.format(args.model)))()
    else:
        args.load = True
        
    with open(os.path.join(args.work_dir, "args_config.yaml") , "w")  as f:
        yaml.safe_dump(args.__dict__, f)


    device = torch.device("cuda:%d" % args.local_rank)
    model = model.to(device)
    #from mynet import MyNet
    #model = MyNet(num_classes=148).to(device)
    
    #from inceptionx import InceptionX
    
    r'''
    if args.load:
        if not args.load_epoch:
            checkpoint_path = os.path.join(args.work_dir, 'model_best.pth.tar')
        else:
            checkpoint_path = os.path.join(args.work_dir, 'checkpoint-{}.pth.tar'.format(args.load_epoch))
            
        return load_checkpoint(model=model, checkpoint_path=checkpoint_path, use_ema = False, device = device)
    '''
    _logger.info('Training with a single process on 1 GPUs on rank %d.' % args.local_rank)

    generator = torch.Generator().manual_seed(args.seed)
    data = MyDataset(args.data)
    split_data = random_split(data,
                              [0.7, 0.3],
                              generator=generator)
    train_loader, valid_loader= [DataLoader(x,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True) for x in split_data]
    from torch.optim import lr_scheduler

    #loss_fn = nn.MSELoss()
    loss_fn = r2_score_loss
    #loss_fn = nn.HingeEmbeddingLoss()

    learning_rate = args.lr
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    #lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)
    
    updates_per_epoch = len(train_loader)
    decreasing_metric = args.eval_metric == 'loss'
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_time = time.time()

    saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            checkpoint_dir=args.work_dir,
            #recovery_dir="xnet_checkpoint_recovery",
            #max_history=50
            decreasing=decreasing_metric,
        )
    
    for epoch in range(num_epochs):
        #_logger.info("---------Epoch:{}, LR：{}---------".format((epoch + 1), lr_scheduler.get_last_lr()))
        _logger.info("---------Epoch:{}, LR：{}---------".format((epoch + 1), lr_scheduler._get_lr(epoch)))
        count_train = 0

        
        model.train() # 将网络设置为训练模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
        for (features, targets) in train_loader:
            
            targets = targets.to(device)
            features = features.to(device)  
            output = model(features)
            
            optimizer.zero_grad()
            loss = loss_fn(output.squeeze(), targets)
            loss.backward()
            optimizer.step()
            count_train += 1

        if count_train % 10 == 0 | count_train == len(train_loader):
 
            end_time = time.time()
            print(f"batch:{count_train}/{len(data) }，loss：{loss.item():.3f}，time：{(end_time - start_time):.2f}" )


        model.eval()
        with torch.no_grad():

            valid_avg_loss, valid_avg_r2score = compute_r2score_and_loss(model, valid_loader, loss_fn, device=device)

            train_avg_loss, train_avg_r2score = compute_r2score_and_loss(model, train_loader, loss_fn, device=device)

            #test_loss_lst.append(test_loss)
            #test_acc_lst.append(test_accuracy)
        
            _logger.info(
                f'Epoch: {epoch:03d}'
                f'Train Loss.: {train_avg_loss:.2f}' f' | Validation Loss.: {valid_avg_loss:.2f}'
                f'Train R2.: {train_avg_r2score:.2f}' f' | Validation R2.: {valid_avg_r2score:.2f}'
            )


        elapsed = (time.time() - start_time) / 60
        _logger.info(f'time used this round: {elapsed:.2f} min')

        lr_scheduler.step(epoch)
        
        eval_metrics = OrderedDict([('loss', train_avg_loss), ('rmse', train_avg_r2score)])
        latest_metrics = eval_metrics[args.eval_metric]
        
        if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metrics)
            
            
        _logger.info('DONE！best_metric {:.3f}, best_epoch {}'.format(best_metric, best_epoch))
if __name__ == "__main__":
    main()
