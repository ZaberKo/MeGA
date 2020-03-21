import torch
import torch.nn as nn
from torch.nn import init
from modules import *


class MeGA(nn.Module):
    def __init__(self, model_config, num_classes=1000):
        super(MeGA, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )

        self.choic_block1 = nn.ModuleList()


        mb_list = list()
        for layer_config in model_config:
            in_size, out_size, expansion_rate, kernel_size, use_se, nolinear, stride = layer_config
            mb_list.append(
                Block(kernel_size, in_size, round(in_size*expansion_rate), out_size,
                      nolinear=hswish(inplace=True) if nolinear == 'hswish' else nn.ReLU(
                          inplace=True),
                      semodule=SeModule(round(in_size*expansion_rate)),
                      stride=stride
                      )
            )
        self.bneck=nn.Sequential(*mb_list)
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.output_layer = nn.Sequential(

        )
        self.fc3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish(inplace=True)
        self.fc4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.stem(x)

        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out)
        out = out.view(out.size(0), -1)

        # SyncBatchNorm bugs, update to torch>=1.4.0
        out = self.hs3(self.bn3(self.fc3(out)))
        out = self.fc4(out)
        return out

