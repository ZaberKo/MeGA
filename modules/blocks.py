import torch
import torch.nn as nn
import torch.nn.functional as F
from .dropout_modules import ScheduleDropBlock


class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SeModule_old(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule_old, self).__init__()
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


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(in_size, in_size//reduction, bias=False),
            nn.BatchNorm1d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_size//reduction, in_size, bias=False),
            nn.BatchNorm1d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        w = self.gap(x).view(x.size(0), -1)
        w = self.se(w).view(x.size(0), -1, 1, 1)
        return x * w


def gen_nolinear(nolinear):
    if nolinear == 'hswish':
        return hswish(inplace=True)
    elif nolinear == 'relu':
        return nn.ReLU(inplace=True)


class ChoiceLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, nolinear: str, stride: int):
        super(ChoiceLayer, self).__init__()
        self.choices = nn.ModuleDict()
        hidden_size3 = in_size*3
        hidden_size6 = in_size*6

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
        self.choices['skip'] = SkipOP(
            in_size, out_size, stride, gen_nolinear(nolinear))

    def forward(self, x, choice):
        return self.choices[choice](x)


class SkipOP(nn.Module):
    def __init__(self, in_size, out_size, stride, nolinear):
        super(SkipOP, self).__init__()
        assert stride == 2 or stride == 1

        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        if self.stride == 2 or self.in_size != self.out_size:
            # use a good Block
            self.conv = Block(5, in_size, in_size*6, out_size,
                              nolinear, SeModule(in_size*6), stride)

    def forward(self, x):
        if self.stride == 2 or self.in_size != self.out_size:
            return self.conv(x)
        else:
            return x


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        # expansion PW
        self.expansion_pw = nn.Sequential(
            nn.Conv2d(in_size, expand_size,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            nolinear
        )

        # DW
        self.dw = nn.Sequential(
            nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            semodule if semodule is not None else nn.Identity(),
            nolinear
        )

        # projection PW
        self.projection_pw = nn.Sequential(
            nn.Conv2d(expand_size, out_size,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_size)
        )

        self.shortcut = nn.Identity()
        if stride == 1 and in_size != out_size:
            # use 1x1conv to change the prev input channel size from in_size to out_size
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )
        # self.use_res_connect=stride==1 and in_size==out_size

    def forward(self, x):
        out = self.expansion_pw(x)
        out = self.dw(out)
        out = self.projection_pw(out)

        if self.stride == 1:
            out = out + self.shortcut(x)
        return out


class Block_Enhanced(Block):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride,
                 dropblock_size=0):
        super(Block_Enhanced, self).__init__(kernel_size, in_size,
                                             expand_size, out_size, nolinear, semodule, stride)
        if dropblock_size > 0:
            self.dropblock = ScheduleDropBlock(
                dropblock_size, per_channel=True)

    # def forward(self, x):
    #     # TODO: test different position of dropblock
    #     out = self.nolinear1(self.bn1(self.conv1(x)))
    #     out = self.nolinear2(self.bn2(self.conv2(out)))
    #     if self.se != None:
    #         out = self.se(out)
    #     out = self.bn3(self.conv3(out))
    #     out = out + self.shortcut(x) if self.stride == 1 else out

    #     return out
