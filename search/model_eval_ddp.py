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
import inspect
import ctypes

from apex import amp
import apex.parallel

import torch.backends.cudnn as cudnn


from modules.hypernet import Hypernet
from modules.label_smooth import LabelSmoothingCELoss
from utils import *
from dataloader import *

ddp_start_thread=None

def evaluate(val_loader, model, path ,prefetch_fn=None):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # model.eval()
    if prefetch_fn is not None:
        val_loader = prefetch_fn(val_loader)


    with torch.no_grad():
        for step, data in enumerate(val_loader):
            if prefetch_fn is None:
                data = tuple(t.cuda() for t in data)
            images, labels = data
            output = model(images, path)
            prec1,prec5 = accuracy(output.detach(), labels, topk=(1,5))
            reduced_prec1 = reduce_tensor(prec1).item()
            reduced_prec5 = reduce_tensor(prec5).item()
            top1.update(reduced_prec1, images.shape[0])
            top5.update(reduced_prec5, images.shape[0])

    return top1,top5


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
    train_batch_size=1

    dataset_type=search_config['dataset'].lower()
    assert dataset_type in ['cifar100','imagenet'], 'currently dataset is only support CIFAR100 & ImageNet'
    if dataset_type=='cifar100':
        train_loader, val_loader = load_cifar100(
            data_path, train_batch_size, val_batch_size, num_workers=search_config['num_workers'], is_distributed=True,big_size=True)
        num_classes=100
        prefetch_fn = None
    else:
        train_loader, val_loader=load_imagenet(data_path,train_batch_size, val_batch_size, num_workers=search_config['num_workers'], is_distributed=True,delay_toTensor=True)
        num_classes=1000
        prefetch_fn = data_prefetcher
    
    model_mode=search_config['model'].lower()
    assert search_config['model'] in ['small' ,'large'], 'only support model "small" & "large"'
    model = Hypernet(mode=model_mode,num_classes=num_classes,dropfc_rate=0)
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
    print(f'success_load model from {search_config["checkpoint_filepath"]}')
    # start service
    while True:
        path = pipe_conn.recv()
        # print(f'rank{local_rank}: recv success')
        dist.barrier()
        prec1_val,prec5_val = evaluate(val_loader, model, path,prefetch_fn=prefetch_fn)
        pipe_conn.send((prec1_val,prec5_val))
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

    global ddp_start_thread
    ddp_start_thread=threading.Thread(target=start,args=(ngpus_per_node, config, children_conns))
    ddp_start_thread.start()
    return ddp_start_thread






def close_eval_processes():
    def _async_raise(tid, exctype):
        """raises the exception, performs cleanup if needed"""
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    
    
    global ddp_start_thread
    _async_raise(ddp_start_thread.get_ident(), SystemExit)

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
