# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import math
import random
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class AddAttack(object):
    def __init__(self, att_path):
        import numpy as np
        self.attack = torch.tensor(np.load(att_path)).type(torch.FloatTensor)
        if len(self.attack.shape) == 4:
            self.attack = self.attack[0]

    def __call__(self, img):
        img = img + self.attack
        img = torch.clamp(img, 0, 1)
        return img


class MultiplyAttack(object):
    def __init__(self, att_path):
        import numpy as np
        self.attack = torch.tensor(np.load(att_path)).type(torch.FloatTensor)
        if len(self.attack.shape) == 4:
            self.attack = self.attack[0]

    def __call__(self, img):
        img = img * self.attack + img
        img = torch.clamp(img, 0, 1)
        return img


# class MA(object):
#     def __init__(self, att_path):
#         import numpy as np
#         add_path = att_path.split('.')[0] + '_addi.npy'
#         mul_path = att_path.split('.')[0] + '_mul.npy'
#         self.add = torch.tensor(np.load(add_path)).type(torch.FloatTensor)
#         self.mul = torch.tensor(np.load(mul_path)).type(torch.FloatTensor)
#         print(add_path)
#         print(mul_path)
#         if len(self.add) == 4:
#             self.add = self.add[0]
#             self.mul = self.mul[0]
#
#     def __call__(self, img):
#         img = img * self.mul + img
#         img = torch.clamp(img, 0, 1)
#         return img


class MA(object):
    def __init__(self, att_path):
        import numpy as np
        mul_path = att_path.split('.')[0] + '_mul.npy'
        self.mul = torch.tensor(np.load(mul_path)).type(torch.FloatTensor)
        print(mul_path)
        if len(self.mul) == 4:
            self.mul = self.mul[0]

    def __call__(self, img):
        img = img * self.mul + img
        img = torch.clamp(img, 0, 1)
        return img


class MulAdd_bk(object):
    def __init__(self, att_path0, radiu):
        import numpy as np
        import os
        att_path_mul = os.path.join(os.path.dirname(att_path0),
                     os.path.basename(att_path0).split('.')[0] + '_mul.npy')
        att_path_mul1 = os.path.join(os.path.dirname(att_path0),
                                os.path.basename(att_path0).split('.')[0] + '_mul1.npy')

        att_path_add = os.path.join(os.path.dirname(att_path0),
                     os.path.basename(att_path0).split('.')[0] + '_addi.npy')



        self.attack_mul = torch.tensor(np.load(att_path_mul)).type(torch.FloatTensor)
        self.attack_mul1 = torch.tensor(np.load(att_path_mul1)).type(torch.FloatTensor)
        self.attack_add = torch.tensor(np.load(att_path_add)).type(torch.FloatTensor)

        self.radiu = radiu
        if len(self.attack_add.shape) == 4:
            self.attack_add = self.attack_add[0]
            self.attack_mul = self.attack_mul[0]
            self.attack_mul1 = self.attack_mul1[0]

    def __call__(self, img):
        # img = img * self.attack + img + self.attack1
        # print(img.shape)
        noise = img / torch.sum(torch.pow(img, 2), dim=[0, 1, 2]) * self.attack_mul + img.sqrt() / torch.sum(torch.pow(img.sqrt(), 2), dim=[0, 1, 2]) * self.attack_mul1 + self.attack_add
        # print(self.radiu*255, torch.norm(noise)*255)
        noise = noise * self.radiu / torch.norm(noise)
        img = img + noise
        # img = img / torch.sum(torch.pow(img, 2), dim=[0, 1, 2]) * self.attack + self.attack1 + img
        img = torch.clamp(img, 0, 1)
        return img

class MulAdd(object):
    def __init__(self, att_path0, radiu, L2=True):
        import numpy as np
        import os
        self.L2 = L2
        att_path_mul = os.path.join(os.path.dirname(att_path0),
                     os.path.basename(att_path0).split('.')[0] + '_mul.npy')

        att_path_add = os.path.join(os.path.dirname(att_path0),
                     os.path.basename(att_path0).split('.')[0] + '_addi.npy')

        self.attack_mul = torch.tensor(np.load(att_path_mul)).type(torch.FloatTensor)
        self.attack_add = torch.tensor(np.load(att_path_add)).type(torch.FloatTensor)

        self.radiu = radiu
        if len(self.attack_add.shape) == 4:
            self.attack_add = self.attack_add[0]
            self.attack_mul = self.attack_mul[0]

    def __call__(self, img):
        # img = img * self.attack + img + self.attack1
        # print(img.shape)
        # noise = img / torch.sqrt(torch.sum(torch.pow(img, 2), dim=[0, 1, 2])) * self.attack_mul + self.attack_add
        noise = img / torch.sum(torch.pow(img, 2), dim=[0, 1, 2]) * self.attack_mul + self.attack_add
            # print(self.radiu*255, torch.norm(noise)*255)
        if self.L2:
            noise = noise * self.radiu / torch.norm(noise)
        else:
            noise = noise.clamp(min=-self.radiu, max=self.radiu)
        img = img + noise
        # img = img / torch.sum(torch.pow(img, 2), dim=[0, 1, 2]) * self.attack + self.attack1 + img
        img = torch.clamp(img, 0, 1)
        return img

class Filter(object):
    def __init__(self, size, mode=None):
        self.size = size
        self.mode = mode

    def __call__(self, img):  # img (3, 256, 128)
        img = img.permute(1, 2, 0)  # img (256, 128, 3)
        img = np.array(img)
        if self.mode == 'median':
            median = cv2.medianBlur(img, self.size)
            median = np.transpose(median, (2, 0, 1))
            median = torch.tensor(median)
        elif self.mode is None:
            median = img
        return median


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        # import pdb
        # pdb.set_trace()
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        import pdb
        pdb.set_trace()
        return x


class AvgPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(AvgPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            print(ih, iw)
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        # import pdb
        # pdb.set_trace()
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])

        x = x.contiguous().view(x.size()[:4] + (-1,)).mean(dim=-1)

        return x

class HighPassFilter(nn.Module):
    """
    Apply KB filter on a 4d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the g filter.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=3, stride=[1, 1], cuda=True):
        super(HighPassFilter, self).__init__()
        weight = torch.tensor([[[[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]], \
                                                       [[[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]], \
                                                       [[[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]]])
        if cuda:
            self.weight = torch.nn.Parameter(weight.cuda())
        else:
            self.weight = torch.nn.Parameter(weight)
        self.k = self.weight.shape[-2:] # kernel size
        self.groups = channels
        self.stride = stride

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        hyfilter = nn.Conv2d(self.groups, self.groups, 3, bias=False, groups=self.groups)
        # print(hyfilter.weight.shape)
        hyfilter.weight = self.weight
        ih, iw = input.size()[-2:]
        # print(ih, iw)
        if ih % self.stride[0] == 0:
            ph = max(self.k[0] - self.stride[0], 0)
        else:
            ph = max(self.k[0] - (ih % self.stride[0]), 0)
        if iw % self.stride[1] == 0:
            pw = max(self.k[1] - self.stride[1], 0)
        else:
            pw = max(self.k[1] - (iw % self.stride[1]), 0)
        pl = pw // 2
        pr = pw - pl
        pt = ph // 2
        pb = ph - pt
        padding = (pl, pr, pt, pb)
        # print(padding)
        input_pad = nn.functional.pad(input, padding, mode='reflect')
        return hyfilter(input_pad)

