from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from .backbones.resnet import ResNet, BasicBlock, Bottleneck

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class HorizontalMaxPool2d(nn.Module):
    def __init__(self):
        super(HorizontalMaxPool2d, self).__init__()

    def forward(self, x):
        inp_size = x.size()
        return nn.functional.max_pool2d(input=x, kernel_size=(1, inp_size[3]))


class AlignedResNet50(nn.Module):

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(AlignedResNet50, self).__init__()

        self.base = ResNet(last_stride=last_stride,
                           block=Bottleneck,
                           layers=[3, 4, 6, 3])
        self.base.load_param(model_path)
    # def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
    #     super(ResNet50, self).__init__()

        self.loss = 'softmax'
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        # self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.feat_dim = 2048 # feature dimension
        self.aligned = True
        self.horizon_pool = HorizontalMaxPool2d()
        self.bnck_128 = nn.BatchNorm1d(128)
        self.bnck_128.bias.requires_grad_(False)
        self.bnck_2048 = nn.BatchNorm1d(2048)
        self.bnck_2048.bias.requires_grad_(False)

        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        # if self.training:
        lf1 = self.bn(x)
        lf2 = self.relu(lf1)
        lf3 = self.horizon_pool(lf2)
        lf = self.conv1(lf3)
        # else:
        #     lf = self.horizon_pool(x)

        lf = lf.view(lf.size()[0:3])
        lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()


        lf_bn = lf.permute(0, 2, 1).contiguous()
        # import pdb
        # pdb.set_trace()
        lf_bn = lf_bn.view(lf.shape[0] * lf.shape[2], lf.shape[1])
        # print('##########################', lf_bn.shape)
        # import pdb
        # pdb.set_pdb
        # pdb.set_trace()
        lf_bn = self.bnck_128(lf_bn)

        lf_final = lf_bn.view(lf.shape[0], lf.shape[2], lf.shape[1])
        lf_final = lf_final.permute(0, 2, 1)


        x1 = F.avg_pool2d(x, x.size()[2:])
        f = x1.view(x.size(0), -1)
        f = self.bnck_2048(f)
        #f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        # if not self.training:
        #     return f, lf
        y = self.classifier(f)

        # import pdb
        # pdb.set_trace()

        if self.training:
            return y, [f, lf_final]
        else:
            return f

