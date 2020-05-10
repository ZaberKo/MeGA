import torch
import torch.nn as nn
from torch.nn import init
from .blocks import *


class Hypernet(nn.Module):
    def __init__(self, mode='large',dropfc_rate=0.2,num_classes=1000):
        super(Hypernet, self).__init__()
        assert mode.lower() in ['small','large']
        largeModel_flag=mode.lower()=='large'


        self.stem=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(inplace=True)
        )

        self.choic_block1=nn.ModuleList()

        self.bneck_layer1=Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1 if largeModel_flag else 2)
        
        if largeModel_flag:
            self.bneck = nn.ModuleList([
                ChoiceLayer(16,24,'relu',2),
                ChoiceLayer(24,24,'relu',1),
                ChoiceLayer(24,40,'relu',2),
                ChoiceLayer(40,40,'relu',1),
                ChoiceLayer(40,40,'relu',1),
                ChoiceLayer(40,80,'hswish',2),
                ChoiceLayer(80,80,'hswish',1),
                ChoiceLayer(80,80,'hswish',1),
                ChoiceLayer(80,80,'hswish',1),
                ChoiceLayer(80,112,'hswish',1),
                ChoiceLayer(112,112,'hswish',1),
                ChoiceLayer(112,160,'hswish',2),
                ChoiceLayer(160,160,'hswish',1),
                ChoiceLayer(160,160,'hswish',1)
            ])
        else:
            self.bneck = nn.ModuleList([
                ChoiceLayer(16,24,'relu',2),
                ChoiceLayer(24,24,'relu',1),
                ChoiceLayer(24,40,'hswish',2),
                ChoiceLayer(40,40,'hswish',1),
                ChoiceLayer(40,40,'hswish',1),
                ChoiceLayer(40,48,'hswish',1),
                ChoiceLayer(48,48,'hswish',1),
                ChoiceLayer(48,96,'hswish',2),
                ChoiceLayer(96,96,'hswish',1),
                ChoiceLayer(96,96,'hswish',1)
            ])

        
        if largeModel_flag:
            c_in=160
            c=960
            n_linear=1280
        else:
            c_in=96
            c=576
            n_linear=1024            

        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            SeModule(c),
            hswish(inplace=True)
        ) 

        self.gap=nn.AdaptiveAvgPool2d(1)

        self.classifier=nn.Sequential(
            nn.Linear(c, n_linear),
            # nn.BatchNorm1d(n_linear),
            hswish(inplace=True),
            nn.Dropout(dropfc_rate),
            nn.Linear(n_linear, num_classes)
        )



        self.fixed_modules=[self.stem,self.bneck_layer1,self.conv2,self.classifier]
        self.choice_modules=list(self.bneck)


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

    def forward(self, x,gene):
        out = self.stem(x)

        out = self.bneck_layer1(out)

        for choice_block,choice in zip(self.bneck ,gene):
            out=choice_block(out,choice)

        out = self.conv2(out)
        out =self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out






