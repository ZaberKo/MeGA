import torch
import torch.nn as nn
import torch.nn.functional as F
from .dropblock import ScheduleDropBlock


class hswish(nn.Module):
    def __init__(self, inplace=False):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class hsigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class ChoiceLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, nolinear: str, stride: int):
        super(ChoiceLayer, self).__init__()
        self.choices = nn.ModuleDict()
        hidden_size3 = in_size*3
        hidden_size6 = in_size*6

        def gen_nolinear(nolinear):
            if nolinear == 'hswish':
                return hswish(inplace=True)
            elif nolinear == 'relu':
                return nn.ReLU(inplace=True)

        self.choices['3x3_e3'] = Block(
            3, in_size, hidden_size3, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['3x3_e3_se'] = Block(
            3, in_size, hidden_size3, out_size, gen_nolinear(nolinear), SeModule(hidden_size3), stride)
        self.choices['5x5_e3'] = Block(
            5, in_size, hidden_size3, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['5x5_e3_se'] = Block(
            5, in_size, hidden_size3, out_size, gen_nolinear(nolinear), SeModule(hidden_size3), stride)
        self.choices['7x7_e3'] = Block(
            7, in_size, hidden_size3, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['7x7_e3_se'] = Block(
            7, in_size, hidden_size3, out_size, gen_nolinear(nolinear), SeModule(hidden_size3), stride)
        self.choices['3x3_e6'] = Block(
            3, in_size, hidden_size6, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['3x3_e6_se'] = Block(
            3, in_size, hidden_size6, out_size, gen_nolinear(nolinear), SeModule(hidden_size6), stride)
        self.choices['5x5_e6'] = Block(
            5, in_size, hidden_size6, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['5x5_e6_se'] = Block(
            5, in_size, hidden_size6, out_size, gen_nolinear(nolinear), SeModule(hidden_size6), stride)
        self.choices['7x7_e6'] = Block(
            7, in_size, hidden_size6, out_size, gen_nolinear(nolinear), None, stride)
        self.choices['7x7_e6_se'] = Block(
            7, in_size, hidden_size6, out_size, gen_nolinear(nolinear), SeModule(hidden_size6), stride)
        # self.choices['skip']=SkipOP(in_size,out_size,stride)
    def forward(self, x, choice):
        return self.choices[choice](x)


class SkipOP(nn.Module):
    def __init__(self, in_size, out_size, stride):
        assert not (stride == 2 and in_size > out_size)
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        if self.stride == 2 or in_size != out_size:
            self.conv = nn.Conv2d(in_size, out_size, 3, stride=2, padding=1)
        super(SkipOP, self).__init__()

    def forward(self, x):
        if self.stride == 1:
            return x
        elif self.stride == 2 or in_size != out_size:
            return self.conv(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride, dropblock_size=None):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        # expansion PW
        self.conv1 = nn.Conv2d(in_size, expand_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # DW
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        # projection PW
        self.conv3 = nn.Conv2d(expand_size, out_size,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            # use 1x1conv to change the prev input channel size from in_size to out_size
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class Block_Enhanced(Block):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,
                 dropblock_size):
        super(Block_Plus, self).__init__(kernel_size, in_size,
                                         expand_size, out_size, nolinear, semodule, stride)
        if dropblock_size>=0:
            self.dropblock=ScheduleDropBlock(dropblock_size,start_dropout_rate, stop_dropout_rate, steps)

    def forward(self, x):
        out=super(Block_Plus, self).forward(x)
        if self.dropblock:
            out=self.dropblock(out)
        return out
