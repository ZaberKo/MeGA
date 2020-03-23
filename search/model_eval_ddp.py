import argparse
import time
import random
import yaml
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from multiprocessing import Pipe
import threading

from apex import amp
import apex.parallel

import torch.backends.cudnn as cudnn


from modules.hypernet import Hypernet_Large, Hypernet_Small
from modules.label_smooth import LabelSmoothingCELoss
from utils import *
from dataloader import *

def evaluate(val_loader, model, path):
    top1 = AverageMeter()
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            data = tuple(t.cuda() for t in data)
            images, labels = data
            output = model(images, path)
            prec1, = accuracy(output.detach(), labels, topk=(1,))
            reduced_prec1 = reduce_tensor(prec1).item()
            top1.update(reduced_prec1, images.shape[0])

    return top1


def main_worker(local_rank, ngpus_per_node, config, children_conns):
    data_path = config['data_path']
    seed = config['seed']
    search_config = config['search_config']
    pipe_conn = children_conns[local_rank]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True  # cudnn auto-tunner

    # device_ids=list(range(torch.cuda.device_count()))
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)

    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:11451",
                            world_size=ngpus_per_node, rank=local_rank)

    val_batch_size = search_config['val_batch_size']//n_gpu
    _, val_loader = load_cifar100(
        data_path, 1, val_batch_size, num_workers=2, is_distributed=True)

    model = Hypernet_Large(num_classes=100)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                check_reduction=False)

    checkpoint = load_checkpoint(
        search_config['checkpoint_filepath'], rank=local_rank)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    while True:
        path = pipe_conn.recv()
        # print(f'rank{local_rank}: recv success')
        dist.barrier()
        prec1_val = evaluate(val_loader, model, path)
        pipe_conn.send(prec1_val)
        # print(f'rank{local_rank}: send success')


_parent_conns = []


def start_eval_processes(config):
    ngpus_per_node = torch.cuda.device_count()
    children_conns = []
    for i in range(ngpus_per_node):
        conn1, conn2 = Pipe(duplex=True)
        _parent_conns.append(conn1)
        children_conns.append(conn2)

    def start(ngpus_per_node, config, children_conns):
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(
            ngpus_per_node, config, children_conns), join=True)

    ddp_start_thread=threading.Thread(target=start,args=(ngpus_per_node, config, children_conns))
    ddp_start_thread.start()
    return ddp_start_thread


def evaluate_arch(path):
    # time.sleep(10)  
    # send
    for conn in _parent_conns:
        conn.send(path)

    # recv
    top1_list = []
    for conn in _parent_conns:
        top1_list.append(conn.recv())

    # for i, top1 in enumerate(top1_list):
    #     print(f'rank{i} acc:{top1.avg}')

    return top1_list[0]
