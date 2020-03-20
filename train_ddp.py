import argparse
import time
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from apex import amp
import apex.parallel

import torch.backends.cudnn as cudnn


from hypernet import Hypernet
from label_smooth import LabelSmoothingCELoss
from utils import *
from dataloader import *


def print_local(*text, **args):
    if local_rank == 0:
        print(*text, **args)


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
        paths = distributed_gen_paths(14, 12, local_rank)

        optimizer.zero_grad()
        loss_list = []
        prec1_list = []
        for path in paths:
            output = model(images, path)
            loss = criterion(output, labels)
            prec1, = accuracy(output.detach(), labels, topk=(1,))

            loss_list.append(reduce_tensor(loss.detach()).item())
            prec1_list.append(reduce_tensor(prec1).item())

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # for i,param in enumerate(model.parameters()):
        #     if i==1000:
        #         print('rank{} grad:{}'.format(local_rank,param.grad))
        #         break

        # time.sleep(3)
        # exit()
        optimizer.step()

        loss_list = np.array(loss_list)
        prec1_list = np.array(prec1_list)
        top1.update(prec1_list.mean(), images.shape[0])
        losses.update(loss_list.mean(), images.shape[0])
        batch_time.update(time.time()-begin_time)

        if step % train_config['display_freq'] == 0:
            print_local('Train: epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} loss_dev:{:.4f} loss_avg:{:.4f} acc:{:.3f} acc_dev:{:.4f} acc_avg={:.3f}'.format(
                epoch, step, batch_time.avg, losses.val, loss_list.std(), losses.avg, top1.val, prec1_list.std(), top1.avg))

        begin_time = time.time()


def main():
    seed = train_config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # cudnn auto-tunner

    # device_ids=list(range(torch.cuda.device_count()))
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl")
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    train_batch_size = train_config['train_batch_size']//n_gpu
    val_batch_size = train_config['val_batch_size']//n_gpu
    train_loader, val_loader = load_cifar100(
        train_config['data_path'], train_batch_size, val_batch_size, num_workers=2, is_distributed=True)

    model = Hypernet(num_classes=100)
    model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=4e-5,
        nesterov=True
    )

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O0',
                                      #   loss_scale=8.0
                                      )

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

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion=LabelSmoothingCELoss().cuda()
    for epoch in range(train_config['epoch']):
        print_local('epoch {} start'.format(epoch))
        print_local('current lr: {}'.format(schedule_lr.get_lr()[0]))

        begin_time = time.time()

        # if epoch >= train_config['start_dropout_schedule_epoch']:
        #     update_dropout_schedule(model)

        train(train_loader, model, criterion,  optimizer, epoch)
        schedule_lr.step()

        print_local('epoch {} time: {:.3f} s'.format(
            epoch, time.time()-begin_time))
        print_local('\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config.json',
                        type=str, help="config file path")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()
    local_rank = args.local_rank
    config_path = args.config_path

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = json.load(f)

    train_config = config['train_hypernet_config']
    main()
