import torch
import torch.nn as nn
from torch.nn import init
from .blocks import *

r'''
    stand-alone model, used for retraining from scratch
'''


class MeGA(nn.Module):
    def __init__(self, model_config,cifar_flag,dropfc_rate=0.5, num_classes=1000):
        super(MeGA, self).__init__()
        assert len(model_config)==15 or len(model_config)==11, 'wrong model config'
        largeModel_flag=len(model_config)==15
        
        stem_stride=2
        if cifar_flag:
            #change stride
            model_config[1][6]=1
            stem_stride=1

        self.stem = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=stem_stride, padding=1, bias=False),
                nn.BatchNorm2d(16),
                hswish(inplace=True)
        )

        self.choic_block1 = nn.ModuleList()

        mb_list = list()
        for layer_config in model_config:
            in_size, out_size, expansion_rate, kernel_size, use_se, nolinear, stride, dropblock_size = layer_config
            if kernel_size==0:
                assert expansion_rate==0 and use_se==False
                if stride==2 and in_size!=out_size:
                    mb_list.append(SkipOP(in_size,out_size,stride))
                continue

            mb_list.append(
                Block(
                    kernel_size, 
                    in_size, 
                    round(in_size*expansion_rate), 
                    out_size,
                    nolinear=hswish(inplace=True) if nolinear == 'hswish' else nn.ReLU(inplace=True),
                    semodule=SeModule(round(in_size*expansion_rate)),
                    stride=stride,
                    # dropblock_size=dropblock_size
                )
            )
        self.bneck = nn.Sequential(*mb_list)

        c_in=model_config[-1][1]
        if largeModel_flag:
            c=960
            n_linear=1280
        else:
            c=576
            n_linear=1024

        self.conv2 = nn.Conv2d(c_in, c, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.hs2 = hswish(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc3 = nn.Linear(c, n_linear)
        self.bn3 = nn.BatchNorm1d(n_linear)
        self.hs3 = hswish(inplace=True)

        self.dropout_fc4=nn.Dropout(dropfc_rate)
        self.fc4 = nn.Linear(n_linear, num_classes)
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

        # SyncBatchNorm bugs, update to torch>=1.4.0 if use DDP
        out = self.hs3(self.bn3(self.fc3(out)))
        out = self.fc4(self.dropout_fc4(out))
        return out

