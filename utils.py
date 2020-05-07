import torch
import torch.distributed as dist
from torch import nn
import math
import numpy as np
import os

# note: this equation selects the maximum padding it can get


def calc_padding_ori(input_size, output_size, kernel_size, stride):
    return math.ceil((stride*(output_size-1)+kernel_size-input_size)/2)


def calc_padding(size, kernel_size, stride):
    '''
    the case when input_size=ouput_size
    '''
    return calc_padding_ori(size, size, kernel_size, stride)


def choice_idx2name(choice_idx):
    if choice_idx == 0:
        choice = '3x3_e3'
    elif choice_idx == 1:
        choice = '3x3_e3_se'
    elif choice_idx == 2:
        choice = '5x5_e3'
    elif choice_idx == 3:
        choice = '5x5_e3_se'
    elif choice_idx == 4:
        choice = '7x7_e3'
    elif choice_idx == 5:
        choice = '7x7_e3_se'
    elif choice_idx == 6:
        choice = '3x3_e6'
    elif choice_idx == 7:
        choice = '3x3_e6_se'
    elif choice_idx == 8:
        choice = '5x5_e6'
    elif choice_idx == 9:
        choice = '5x5_e6_se'
    elif choice_idx == 10:
        choice = '7x7_e6'
    elif choice_idx == 11:
        choice = '7x7_e6_se'
    elif choice_idx==12:
        choice='skip'
    else:
        raise ValueError('incorrect choice idx, must between 0 and 12')
    return choice

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def gene2config(gene=None,multiplier=1):
    # template based on mobilenetv3 Large
    template = [
        # in_size,out_size,expansion_rate,kernel_size,use_se,activation,stride,dropblock_size
        [16, 16, 1, 3, 'none', 'relu', 1, 0], # fixed layer
        [16, 24, 4, 3, 'none', 'relu', 2, 0],  # 112x112->56x56
        [24, 24, 3, 3, 'none', 'relu', 1, 0],
        [24, 40, 3, 5, 'se', 'relu', 2, 0],  # 56x56->28x28
        [40, 40, 3, 5, 'se', 'relu', 1, 0],
        [40, 40, 3, 5, 'se', 'relu', 1, 0],
        [40, 80, 6, 3, 'none', 'hswish', 2, 0],  # 28x28->14x14
        [80, 80, 2.5, 3, 'none', 'hswish', 1, 5],
        [80, 80, 2.3, 3, 'none', 'hswish', 1, 5],
        [80, 80, 2.3, 3, 'none', 'hswish', 1, 5],
        [80, 112, 6, 3, 'se', 'hswish', 1, 5],
        [112, 112, 6, 3, 'se', 'hswish', 1, 5],
        [112, 160, 6, 5, 'se', 'hswish', 2, 0],  # 14x14->7x7
        [160, 160, 4.2, 5, 'se', 'hswish', 1, 0],
        [160, 160, 6, 5, 'se', 'hswish', 1, 0],  # final channel:960
    ]


    def update(layer, kernel_size, expansion_rate, use_se):
        template[layer][3] = kernel_size
        template[layer][2] = expansion_rate
        template[layer][5] = 'se' if use_se else 'none'

    if gene is not None:
        for i, choice_idx in enumerate(gene):
            layer = i+1
            if multiplier!=1:
                template[layer][0]=_make_divisible(template[layer][0]*multiplier)
                template[layer][1]=_make_divisible(template[layer][1]*multiplier)
            if choice_idx == 0:
                # '3x3_e3'
                update(layer, 3, 3, False)
            elif choice_idx == 1:
                # '3x3_e3_se'
                update(layer, 3, 3, True)
            elif choice_idx == 2:
                # '5x5_e3'
                update(layer, 5, 3, False)
            elif choice_idx == 3:
                # '5x5_e3_se'
                update(layer, 5, 3, True)
            elif choice_idx == 4:
                # '7x7_e3'
                update(layer, 7, 3, False)
            elif choice_idx == 5:
                # '7x7_e3_se'
                update(layer, 7, 3, True)
            elif choice_idx == 6:
                # '3x3_e6'
                update(layer, 3, 6, False)
            elif choice_idx == 7:
                # '3x3_e6_se'
                update(layer, 3, 6, True)
            elif choice_idx == 8:
                # '5x5_e6'
                update(layer, 5, 6, False)
            elif choice_idx == 9:
                # '5x5_e6_se'
                update(layer, 5, 6, True)
            elif choice_idx == 10:
                # '7x7_e6'
                update(layer, 7, 6, False)
            elif choice_idx == 11:
                # '7x7_e6_se'
                update(layer, 7, 6, True)
            elif choice_idx==12:
                # 'skip'
                update(layer,0,0,False)

    return template


def get_permutation(num_layers, num_choices):
    layers_idx = []
    for i in range(num_layers):
        layers_idx.append(np.random.permutation(num_choices))

    paths = []
    for i in range(num_choices):
        path = []
        for layer_idx in layers_idx:
            path.append(layer_idx[i])
        paths.append(path)

    return paths

def path_idx2name(paths):
    new_paths = []
    for path in paths:
        new_path = []
        for choice in path:
            new_path.append(choice_idx2name(choice))
        new_paths.append(new_path)

    return new_paths
def gen_paths(num_layers, num_choices):
    paths = get_permutation(num_layers, num_choices)
    return path_idx2name(paths)


def distributed_gen_paths(num_layers, num_choices, rank):
    if rank == 0:
        paths = get_permutation(num_layers, num_choices)
        paths = torch.tensor(paths, dtype=torch.int32).cuda()
    else:
        paths = torch.zeros(size=(num_choices, num_layers),
                            dtype=torch.int32).cuda()

    dist.broadcast(paths, 0)
    paths = paths.tolist()
    return path_idx2name(paths)


def reduce_tensor(tensor, avg=True):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if avg:
        world_size = dist.get_world_size()
        rt /= world_size

    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def correct_cnt(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # res.append(correct_k.div_(batch_size))
        res.append(correct_k)
    return res


def save_checkpoint(state, filename='./checkpoint.pth'):
    dir_path=os.path.dirname(filename)
    if len(dir_path)>0 and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(state, filename)

def load_checkpoint(path,rank=0):
    assert os.path.isfile(path),"=> no checkpoint found at '{}'".format(path)
    checkpoint = torch.load(path, map_location = lambda storage, loc: storage.cuda(rank))
    return checkpoint
    

        