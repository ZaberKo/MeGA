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

from modules.hypernet import Hypernet
from modules.label_smooth import LabelSmoothingCELoss
from modules.NoBiasDecay import noBiasDecay_hypernet
from modules.cosine_annearing_with_warmup import *
from utils import *
from dataloader import *



from mplog import MPLog


logger = None


def evaluate(val_loader, model, criterion, paths=None, training=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if paths is None:
        paths = distributed_gen_paths(num_layers, num_choices, local_rank)

    model.eval()

    val_loader=data_prefetcher(val_loader)

    with torch.no_grad():
        begin_time = time.time()
        for step, data in enumerate(val_loader):
            # data = tuple(t.cuda() for t in data)
            images, labels = data
            loss_list = []
            prec1_list = []
            prec5_list = []
            for path in paths:
                output = model(images, path)
                loss = criterion(output, labels)
                prec1, prec5 = accuracy(output.detach(), labels, topk=(1, 5))

                loss_list.append(reduce_tensor(loss.detach()).item())
                prec1_list.append(reduce_tensor(prec1).item())
                prec5_list.append(reduce_tensor(prec5).item())

            loss_list = np.array(loss_list)
            prec1_list = np.array(prec1_list)
            prec5_list = np.array(prec5_list)

            top1.update(prec1_list.mean(), images.shape[0])
            top5.update(prec5_list.mean(), images.shape[0])
            losses.update(loss_list.mean(), images.shape[0])
            batch_time.update(time.time()-begin_time)

            if not training:
                logger.log(
                    f'Val  : iter:{step:>4} avg_batch_time:{batch_time.avg:.3f}s loss:{losses.val:.4f}[std={loss_list.std():.4f}] loss_avg:{losses.avg:.4f} acc:{top1.val:.3f}[std={prec1_list.std():.4f}] acc5:{top5.val:.3f}[std={prec1_list.std():.4f}] acc_avg:{top1.avg:.3f} acc5_avg:{top5.avg:.3f}')

            begin_time = time.time()

    return top1, top5


def train(train_loader, model, criterion,  optimizer, lr_scheduler,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)

    model.train()

    train_loader.sampler.set_epoch(epoch)
    train_loader=data_prefetcher(train_loader)

    for step, data in enumerate(train_loader):
        begin_time = time.time()
        # data = tuple(t.cuda() for t in data)
        images, labels = data

        paths = distributed_gen_paths(num_layers, num_choices, local_rank)

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
        lr_scheduler.step()

        loss_list = np.array(loss_list)
        prec1_list = np.array(prec1_list)
        top1.update(prec1_list.mean(), images.shape[0])
        losses.update(loss_list.mean(), images.shape[0])
        batch_time.update(time.time()-begin_time)

        if step % visualization_config['display_freq'] == 0:
            logger.log(
                f'Train: epoch:{epoch:>4}: iter:{step:>4} avg_batch_time:{batch_time.avg:.3f}s loss:{losses.val:.4f}[std={loss_list.std():.4f}] loss_avg:{losses.avg:.4f} acc:{top1.val:.3f}[std={prec1_list.std():.4f}] acc_avg:{top1.avg:.3f}')

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
    val_batch_size = val_config['val_batch_size']//n_gpu

    dataset_type=train_config['dataset'].lower()
    assert dataset_type in ['cifar100','imagenet'], 'currently dataset is only support CIFAR100 & ImageNet'
    if dataset_type=='cifar100':
        train_loader, val_loader = load_cifar100(
            data_path, train_batch_size, val_batch_size, num_workers=train_config['num_workers'], is_distributed=True,big_size=True)
        num_classes=100
        train_dataset_len=50000
    else:
        train_loader, val_loader=load_imagenet(data_path,train_batch_size, val_batch_size, num_workers=train_config['num_workers'], is_distributed=True,delay_toTensor=True)
        num_classes=1000
        train_dataset_len=1281167

    assert train_config['model'] in ['small' ,'large'], 'only support model "small" & "large"'


    model = Hypernet(mode=train_config['model'], num_classes=num_classes,dropfc_rate=train_config['dropfc_rate'])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        noBiasDecay_hypernet(model, lr=train_config['lr'],weight_decay=train_config['weight_decay'],num_choices=num_choices),
        momentum=0.9,
        nesterov=True
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

    lr_scheduler=CosineWarmupLR(
        optimizer,
        epochs=train_config['epoch'],
        iter_in_one_epoch=train_dataset_len//train_config['train_batch_size'],
        lr_min=1e-5,
        warmup_epochs=2
    )

    # lr_scheduler=CosineLR(
    #     optimizer,
    #     100,
    #     eta_min=1e-5,
    #     T_mult=1,
    #     warmup_epochs=10,
    #     decay_rate=1
    # )

    if train_config['loss'] == 'LSCE':
        criterion = LabelSmoothingCELoss().cuda()
    elif train_config['loss'] == 'CE':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError('only support loss "LSCE" & "CE"')

    if args.do_eval:
        checkpoint = load_checkpoint(
            val_config['checkpoint_filepath'], rank=local_rank)
        model.load_state_dict(checkpoint['state_dict'])
        begin_time = time.time()
        prec1_val = evaluate(val_loader, model, criterion)
        logger.log('val acc :{:.4f} time: {:.3f} s'.format(
            prec1_val.avg, time.time()-begin_time))
        return

    for epoch in range(train_config['epoch']):
        logger.log(f'epoch {epoch} start')
        logger.log(f'current lr: {lr_scheduler.get_lr()[-1]}')

        begin_time = time.time()

        prec1_train = train(train_loader, model, criterion,  optimizer, lr_scheduler,epoch)
        prec1_val, prec5_val = evaluate(
            val_loader, model, criterion, training=True)
        # lr_scheduler.step()

        logger.log(f'train acc: {prec1_train.avg:.4f}')
        logger.log(f'val acc: top1: {prec1_val.avg:.4f} top5: {prec5_val.avg}')
        logger.log(f'epoch {epoch} time: {time.time()-begin_time:.3f} s')
        logger.log('\n\n')

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
    parser.add_argument('--log_path', default='debug.log',
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

    num_layers = 10 if train_config['model'] == 'small' else 14
    num_choices = 13


    logger = MPLog(args.log_path, local_rank)
    main()
