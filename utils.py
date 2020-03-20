import torch
import torch.distributed as dist
from torch import nn
import math
import numpy as np

# note: this equation selects the maximum padding it can get


def calc_padding_ori(input_size, output_size, kernel_size, stride):
    return math.ceil((stride*(output_size-1)+kernel_size-input_size)/2)


def calc_padding(size, kernel_size, stride):
    '''
    the case when input_size=ouput_size
    '''
    return calc_padding_ori(size, size, kernel_size, stride)


def choice_idx2name(idx):
    if idx == 0:
        choice = '3x3_e3'
    elif idx == 1:
        choice = '3x3_e3_se'
    elif idx == 2:
        choice = '5x5_e3'
    elif idx == 3:
        choice = '5x5_e3_se'
    elif idx == 4:
        choice = '7x7_e3'
    elif idx == 5:
        choice = '7x7_e3_se'
    elif idx == 6:
        choice = '3x3_e6'
    elif idx == 7:
        choice = '3x3_e6_se'
    elif idx == 8:
        choice = '5x5_e6'
    elif idx == 9:
        choice = '5x5_e6_se'
    elif idx == 10:
        choice = '7x7_e6'
    elif idx == 11:
        choice = '7x7_e6_se'
    return choice


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

def gen_paths(num_layers, num_choices):
    paths=get_permutation(num_layers,num_choices)
    new_paths = []
    for path in paths:
        new_path = []
        for choice in path:
            new_path.append(choice_idx2name(choice))
        new_paths.append(new_path)

    return new_paths

def distributed_gen_paths(num_layers, num_choices,rank):
    if rank == 0:
        paths = get_permutation(num_layers,num_choices)
        paths = torch.tensor(paths, dtype=torch.int32).cuda()
    else:
        paths = torch.zeros(size=(num_choices, num_layers), dtype=torch.int32).cuda()

    dist.broadcast(paths, 0)
    paths=paths.tolist()

    new_paths = []
    for path in paths:
        new_path = []
        for choice in path:
            new_path.append(choice_idx2name(choice))
        new_paths.append(new_path)

    return new_paths

if __name__ == "__main__":
    paths = get_permutation(14, 12)
    from pprint import pprint
    pprint(paths)
    print(torch.tensor(paths).shape)


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
        res.append(correct_k.mul_(100/batch_size))
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
