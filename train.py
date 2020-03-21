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


from modules.hypernet import Hypernet_Large,Hypernet_Small
from modules.label_smooth import LabelSmoothingCELoss
from utils import *
from dataloader import *


def train(train_loader, model, criterion,  optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)

    model.train()

    begin_time = time.time()

    for step, data in enumerate(train_loader):
        data = tuple(t.cuda() for t in data)
        images, labels = data

        paths = gen_paths(14, 12)
        optimizer.zero_grad()
        loss_list = []
        prec1_list = []
        for path in paths:
            output = model(images, path)
            loss = criterion(output, labels)
            prec1, = accuracy(output.detach(), labels, topk=(1,))
            loss_list.append(loss.detach().item())
            prec1_list.append(prec1.item())

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
            print('Train: epoch:{:>4}: iter:{:>4} avg_batch_time: {:.3f} s loss:{:.4f} loss_dev:{:.4f} loss_avg:{:.4f} acc:{:.3f} acc_dev:{:.4f} acc_avg={:.3f}'.format(
                epoch, step, batch_time.avg, losses.val, loss_list.std(), losses.avg, top1.val, prec1_list.std(), top1.avg))

        begin_time = time.time()


def main():
    seed = train_config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # cudnn auto-tunner

    device_ids = list(range(torch.cuda.device_count()))
    train_batch_size = train_config['train_batch_size']
    val_batch_size = train_config['val_batch_size']
    train_loader, val_loader = load_cifar100(
       data_path, train_batch_size, val_batch_size, num_workers=2)

    model = Hypernet_Large(num_classes=100)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_config['lr'],
        momentum=0.9,
        weight_decay=4e-5,
        nesterov=True
    )

    # there is a bug in apex with DataParallel with opt_level='O0'
    # see https://github.com/NVIDIA/apex/issues/227
    # model, optimizer=amp.initialize(model, optimizer,
    #                                   opt_level='O1',
    #                                   loss_scale=1.0
    #                                   )

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model)

    schedule_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        train_config['epoch'],
        eta_min=1e-5
    )

    # criterion=torch.nn.CrossEntropyLoss().cuda()
    criterion = LabelSmoothingCELoss().cuda()
    for epoch in range(train_config['epoch']):
        print('epoch {} start'.format(epoch))
        print('current lr: {}'.format(schedule_lr.get_lr()[0]))

        begin_time = time.time()

        # if epoch >= train_config['start_dropout_schedule_epoch']:
        #     update_dropout_schedule(model)

        train(train_loader, model, criterion,  optimizer, epoch)
        schedule_lr.step()

        print('epoch {} time: {:.3f} s'.format(epoch, time.time()-begin_time))
        print('\n\n')

        if (epoch+1) % train_config['save_freq'] == 0:
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
                        type=str, help="config file path")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='enable resume and specify the checkpoint file')
    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path, mode='r', encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
    
    data_path=config['data_path']
    train_config = config['train_hypernet_config']
    visualization_config=config['visualization_config']
    main()
