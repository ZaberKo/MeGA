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

from modules.mega import MeGA
from modules.label_smooth import LabelSmoothingCELoss
from modules.dropout_modules import init_dropout_schedule, update_dropout_schedule
from modules.cosine_annearing_with_warmup import *
from utils import *
from dataloader import *
from modules.NoBiasDecay import noBiasDecay

from mplog import MPLog

logger = None


def evaluate(val_loader, model, criterion, training=False,prefetch_fn=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    if prefetch_fn is not None:
        val_loader=prefetch_fn(val_loader)

    with torch.no_grad():
        begin_time = time.time()
        for step, data in enumerate(val_loader):
            if prefetch_fn is None:
                data = tuple(t.cuda() for t in data)
            images, labels = data

            output = model(images)
            loss = criterion(output, labels)
            prec1, prec5 = accuracy(output.detach(), labels, topk=(1, 5))

            reduced_loss = reduce_tensor(loss.detach()).item()
            reduced_prec1 = reduce_tensor(prec1).item()
            reduced_prec5 = reduce_tensor(prec5).item()

            top1.update(reduced_prec1, images.shape[0])
            top5.update(reduced_prec5, images.shape[0])
            losses.update(reduced_loss, images.shape[0])
            batch_time.update(time.time()-begin_time)

            if not training:
                logger.log(
                    f'Val  : epoch:{0:>4} iter:{step:>4} avg_batch_time:{batch_time.avg:.3f}s loss:{losses.val:.4f} loss_avg:{losses.avg:.4f} top1:{top1.val:.3f} top1_avg:{top1.avg:.3f} top5:{top5.val:.3f} top5_avg:{top5.avg:.3f}')

            begin_time = time.time()

    return top1, top5


def train(train_loader, model, criterion,  optimizer, lr_scheduler,epoch,prefetch_fn=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # torch.autograd.set_detect_anomaly(True)

    model.train()

    train_loader.sampler.set_epoch(epoch)
    if prefetch_fn is not None:
        train_loader=prefetch_fn(train_loader)

    for step, data in enumerate(train_loader):
        begin_time = time.time()
        if prefetch_fn is None:
            data = tuple(t.cuda() for t in data)
        images, labels = data

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        prec1, = accuracy(output.detach(), labels, topk=(1,))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        reduced_loss = reduce_tensor(loss.detach()).item()
        reduced_prec1 = reduce_tensor(prec1).item()
        
        top1.update(reduced_prec1, images.shape[0])
        losses.update(reduced_loss, images.shape[0])
        batch_time.update(time.time()-begin_time)

        if step % visualization_config['display_freq'] == 0:
            logger.log(
                f'Train: epoch:{epoch:>4}: iter:{step:>4} avg_batch_time:{batch_time.avg:.3f}s loss:{losses.val:.4f} loss_avg:{losses.avg:.4f} acc:{top1.val:.3f} acc_avg:{top1.avg:.3f}')

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
    val_batch_size = val_config['val_batch_size']//n_gpu

    dataset_type=train_config['dataset'].lower()
    assert dataset_type in ['cifar100','imagenet'], 'currently dataset is only support CIFAR100 & ImageNet'
    if dataset_type=='cifar100':
        train_loader, val_loader = load_cifar100(
            data_path, train_batch_size, val_batch_size, num_workers=train_config['num_workers'], is_distributed=True,big_size=True)
        num_classes=100
        train_dataset_len=50000
        prefetch_fn = None
    else:
        train_loader, val_loader=load_imagenet(data_path,train_batch_size, val_batch_size, num_workers=train_config['num_workers'], is_distributed=True,delay_toTensor=True)
        num_classes=1000
        train_dataset_len=1281167
        prefetch_fn = data_prefetcher

    model = MeGA(model_config, cifar_flag=False,dropfc_rate=train_config['dropfc_rate'],num_classes=num_classes)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    optimizer=torch.optim.SGD(
        noBiasDecay(model,lr=train_config['lr'],weight_decay=train_config['weight_decay']),
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

    # init_dropout_schedule(
    #     model,
    #     train_config['start_dropblock_rate'],
    #     train_config['end_dropblock_rate'],
    #     train_config['epoch'] if train_config['dropblock_schedule_steps'] <= 0 else train_config['dropblock_schedule_steps']
    # )

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     train_config['epoch'],
    #     eta_min=1e-5
    # )



    lr_scheduler=CosineWarmupLR(
        optimizer,
        epochs=train_config['epoch'],
        iter_in_one_epoch=train_dataset_len//train_config['train_batch_size'],
        lr_min=1e-5,
        warmup_epochs=2
    )

    # lr_scheduler=CosineAnnealingWarmUpRestarts(
    #     optimizer,
    #     train_config['epoch']//3,
    #     eta_max=1e-5,
    #     T_up=10,
    #     gamma=1
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
        prec1_val, prec5_val = evaluate(val_loader, model, criterion,training=False, prefetch_fn=prefetch_fn)
        logger.log(
            f'val acc :{prec1_val.avg:.4f} time: {time.time()-begin_time:.3f} s')
        return

    for epoch in range(train_config['epoch']):
        logger.log(f'epoch {epoch} start')
        logger.log(f'current lr: {lr_scheduler.get_lr()[-1]}')

        begin_time = time.time()

        prec1_train = train(train_loader, model, criterion,  optimizer,lr_scheduler, epoch,prefetch_fn=prefetch_fn)
        prec1_val, prec5_val = evaluate(val_loader, model, criterion, training=True,prefetch_fn=prefetch_fn)
        # lr_scheduler.step()
        # update_dropout_schedule(model)

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
                             f'hypernet_{epoch+1}.pth')
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
    train_config = config['train_config']
    val_config = config['val_config']
    visualization_config = config['visualization_config']

    logger=MPLog(args.log_path,local_rank)
    # model_config = gene2config([3 for i in range(14)],cifar=True)
    model_config = gene2config(gene='large',multiplier=1)
    main()
