import torch
import torch.nn as nn
from modules import *
from torch.nn import init



class Hypernet(nn.Module):
    def __init__(self, num_classes=1000):
        super(Hypernet, self).__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish()
        )

        self.choic_block1=nn.ModuleList()

        self.bneck_layer1=Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1)
        
        self.bneck = nn.ModuleList([
            ChoiceLayer(16,24,nn.ReLU(inplace=True),2),
            ChoiceLayer(24,24,nn.ReLU(inplace=True),1),
            ChoiceLayer(24,40,nn.ReLU(inplace=True),2),
            ChoiceLayer(40,40,nn.ReLU(inplace=True),1),
            ChoiceLayer(40,40,nn.ReLU(inplace=True),1),
            ChoiceLayer(40,80,hswish(),2),
            ChoiceLayer(80,80,hswish(),1),
            ChoiceLayer(80,80,hswish(),1),
            ChoiceLayer(80,80,hswish(),1),
            ChoiceLayer(80,112,hswish(),2),
            ChoiceLayer(112,112,hswish(),1),
            ChoiceLayer(112,160,hswish(),2),
            ChoiceLayer(160,160,hswish(),1),
            ChoiceLayer(160,160,hswish(),1)
        ])

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.output_layer=nn.Sequential(

        )
        self.fc3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
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

    def forward(self, x,gene):
        out = self.stem(x)

        out = self.bneck_layer1(out)

        for choice_block,choice in zip(self.bneck ,gene):
            out=choice_block(out,choice)

        out = self.hs2(self.bn2(self.conv2(out)))
        out =self.gap(out)
        out = out.view(out.size(0), -1)
        
        # for SyncBatchNorm bugs. or update to torch>=1.4.0
        # out=self.fc3(out) #(B,C)
        # out=out.unsqueeze(dim=-1)
        # out=self.bn3(out)
        # out=out.squeeze(dim=-1)
        # out=self.hs3(out)

        out = self.hs3(self.bn3(self.fc3(out)))
        out = self.fc4(out)
        return out



