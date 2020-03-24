import argparse
import time
import random
import yaml
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from apex import amp
import apex.parallel

import torch.backends.cudnn as cudnn

from mplog import init_log
from modules.hypernet import Hypernet_Large, Hypernet_Small
from modules.label_smooth import LabelSmoothingCELoss
from utils import *
from dataloader import *

import logging

logger = logging.getLogger(__name__)

def print_local(*text, **args):
    if local_rank == 0:
        print(*text, **args)


def evaluate(val_loader, model, criterion, paths=None,training=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    if paths is None:
        paths = distributed_gen_paths(num_layers,num_choices ,local_rank)

    model.eval()

    with torch.no_grad():
        begin_time = time.time()
        for step, data in enumerate(val_loader):
            data = tuple(t.cuda() for t in data)
            images, labels = data
            loss_list = []
            prec1_list = []
            for path in paths:
                output = model(images, path)
                loss = criterion(output, labels)
                prec1, = accuracy(output.detach(), labels, topk=(1,))

                loss_list.append(reduce_tensor(loss.detach()).item())
                prec1_list.append(reduce_tensor(prec1).item())

            loss_list = np.array(loss_list)
            prec1_list = np.array(prec1_list)

            top1.update(prec1_list.mean(), images.shape[0])
            losses.update(loss_list.mean(), images.shape[0])
            batch_time.update(time.time()-begin_time)

            if not training:
                logger.info('Val  : epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} loss_dev:{:.4f} loss_avg:{:.4f} acc:{:.3f} acc_dev:{:.4f} acc_avg={:.3f}'.format(
                    0, step, batch_time.avg, losses.val, loss_list.std(), losses.avg, top1.val, prec1_list.std(), top1.avg))

            begin_time = time.time()

    return top1


def train(train_loader, model, criterion,  optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)

    model.train()


    begin_time = time.time()

    for step, data in enumerate(train_loader):
        train_loader.sampler.set_epoch(epoch)
        data = tuple(t.cuda() for t in data)
        images, labels = data

        paths = distributed_gen_paths(num_layers,num_choices ,local_rank)

        optimizer.zero_grad()
        loss_list = []
        prec1_list = []
        for path in paths:
            output = model(images, path)
            loss = criterion(output, labels)
            prec1, = accuracy(output.detach(), labels, topk=(1,))

            loss_list.append(reduce_tensor(loss.detach()).item())
            prec1_list.append(reduce_tensor(prec1).item())

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()

        optimizer.step()

        loss_list = np.array(loss_list)
        prec1_list = np.array(prec1_list)
        top1.update(prec1_list.mean(), images.shape[0])
        losses.update(loss_list.mean(), images.shape[0])
        batch_time.update(time.time()-begin_time)

        if step % visualization_config['display_freq'] == 0:
            logger.info('Train: epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} loss_dev:{:.4f} loss_avg:{:.4f} acc:{:.3f} acc_dev:{:.4f} acc_avg={:.3f}'.format(
                epoch, step, batch_time.avg, losses.val, loss_list.std(), losses.avg, top1.val, prec1_list.std(), top1.avg))

        begin_time = time.time()

    return top1


def main():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # cudnn auto-tunner

    # device_ids=list(range(torch.cuda.device_count()))
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)

    dist.init_process_group(backend="nccl")
    # assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    train_batch_size = train_config['train_batch_size']//n_gpu
    val_batch_size = train_config['val_batch_size']//n_gpu
    train_loader, val_loader = load_cifar100(
        data_path, train_batch_size, val_batch_size, num_workers=2, is_distributed=True)

    if train_config['model'] == 'small':
        model = Hypernet_Small(num_classes=100)
    elif train_config['model'] == 'large':
        model = Hypernet_Large(num_classes=100)
    else:
        raise ValueError('only support model "small" & "large"')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        # momentum=0.9,
        weight_decay=4e-5,
        # nesterov=True
    )

    # model, optimizer = amp.initialize(model, optimizer,
    #                                   opt_level='O0',
    #                                   #   loss_scale=8.0
    #                                   )

    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                check_reduction=False)

    schedule_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        train_config['epoch'],
        eta_min=1e-5
    )

    if train_config['loss']=='LSCE':
        criterion = LabelSmoothingCELoss().cuda()
    elif train_config['loss']=='CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError('only support loss "LSCE" & "CE"')

    if args.do_eval:
        checkpoint = load_checkpoint(
            val_config['checkpoint_filepath'], rank=local_rank)
        model.load_state_dict(checkpoint['state_dict'])
        begin_time = time.time()
        prec1_val = evaluate(val_loader, model, criterion)
        logger.info('val acc :{:.4f} time: {:.3f} s'.format(
            prec1_val.avg, time.time()-begin_time))
        return

    for epoch in range(train_config['epoch']):
        logger.info('epoch {} start'.format(epoch))
        logger.info('current lr: {}'.format(schedule_lr.get_lr()[0]))

        begin_time = time.time()

        prec1_train = train(train_loader, model, criterion,  optimizer, epoch)
        prec1_val = evaluate(val_loader, model, criterion, training=True)
        schedule_lr.step()
        logger.info('train acc: {:.4f}'.format(prec1_train.avg))
        logger.info('val acc: {:.4f}'.format(prec1_val.avg))
        logger.info('epoch {} time: {:.3f} s'.format(
            epoch, time.time()-begin_time))
        logger.info('\n\n')

        if local_rank == 0 and (epoch+1) % train_config['save_freq'] == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(train_config['checkpoint_path'],
                             'hypernet_{}.pth'.format(epoch+1))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.yaml',
                        type=str, help='config file path')
    parser.add_argument('--log_path', default=None,
                        type=str, help='log file path')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()
    local_rank = args.local_rank
    config_path = args.config_path

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    data_path = config['data_path']
    seed = config['seed']
    train_config = config['train_hypernet_config']
    val_config = config['val_hypernet_config']
    visualization_config = config['visualization_config']
    num_layers=10 if train_config['model'] == 'small' else 14
    num_choices=12
    
    init_log(args.log_path)
    main()
