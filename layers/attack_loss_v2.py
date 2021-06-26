from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from data.transforms.transforms import AvgPool2d
import math
import numbers
from data.transforms.transforms import AvgPool2d


def normalize(x, axis=1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if square:
        dist = dist.clamp(min=1e-12)
    else:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist



class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class TVLossTmp(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLossTmp,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = (x.size()[1]-1) * x.size()[2]
        count_w = x.size()[1] * (x.size()[2] - 1)
        h_tv = torch.pow((x[:, 1:, :]-x[:, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, 1:]-x[:, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


def map_loss_v3(atta_feat, feat, target, bin_num, margin=0):    # https://arxiv.org/pdf/1906.07589.pdf
    # import numpy as np
    # np.save('atta_feat.npy', atta_feat.data.cpu().numpy())
    # np.save('feat.npy', feat.data.cpu().numpy())
    # np.save('target.npy', target.data.cpu().numpy())
    N = atta_feat.size(0)
    atta_feat = normalize(atta_feat)
    feat = normalize(feat)
    dist_raw = euclidean_dist(atta_feat, feat)
    dist = dist_raw.clone()
    # bin_num = 20
    # bin_len = 2./(bin_num-1)
    bin_len = (2.+margin) / (bin_num - 1)
    is_pos = target.expand(N, N).eq(target.expand(N, N).t()).float()
    is_neg = target.expand(N, N).ne(target.expand(N, N).t()).float()
    total_true_indicator = torch.zeros(N).to('cuda')
    total_all_indicator = torch.zeros(N).to('cuda')
    AP = torch.zeros(N).to('cuda')

    # import pdb
    # pdb.set_trace()
    if margin is None:
        pass
    else:
        is_pos_index = target.expand(N, N).eq(target.expand(N, N).t())
        is_neg_index = target.expand(N, N).ne(target.expand(N, N).t())
        dist[is_pos_index] = dist_raw[is_pos_index] - margin/2
        dist[is_neg_index] = dist_raw[is_neg_index] + margin/2

    for i in range(1, bin_num+1):
        # bm = 1 - (i-1) * bin_len
        bm = (i-1) * bin_len - margin /2.
        indicator = (1 - torch.abs(dist - bm)/bin_len).clamp(min=0)
        true_indicator = is_pos * indicator
        all_indicator = indicator
        sum_true_indicator = torch.sum(true_indicator, 1)
        sum_all_indicator = torch.sum(all_indicator, 1)
        total_true_indicator = total_true_indicator + sum_true_indicator
        total_all_indicator = total_all_indicator + sum_all_indicator
        Pm = total_true_indicator / total_all_indicator.clamp(min=1e-12)
        rm = sum_true_indicator / 4
        ap_bin = Pm*rm
        AP = AP + ap_bin
        # import pdb
        # pdb.set_trace()
    final_AP = torch.sum(AP) / N
    return final_AP

class MapLoss(nn.Module):

    def __init__(self):
        super(MapLoss, self).__init__()
        # self.name = 'map'

    def forward(self, atta_feat, feat, target, bin_num, margin=0):
        loss = map_loss_v3(atta_feat, feat, target, bin_num, margin=margin)
        return loss


class ODFALoss(nn.Module):

    def __init__(self, use_gpu=True):
        super(ODFALoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, features_adv, features_x):
        '''
        Args:
            features_x: feature matrix with shape (feat_dim, ).
        '''
        assert features_adv.shape == features_x.shape
        # batch_size = features_x
        features_adv = features_adv.view(-1, 1)
        features_x = features_x.view(-1, 1)
        loss = torch.mm((features_adv / torch.norm(features_adv) + features_x / torch.norm(features_x)).t(),
                        (features_adv / torch.norm(features_adv) + features_x / torch.norm(features_x)))
        return loss


class MetricLoss(nn.Module):

    def __init__(self):
        super(MetricLoss, self).__init__()

    def forward(self, atta_feat, feat):
        loss = nn.MSELoss()
        ret = -loss(atta_feat, feat)
        return ret





# if __name__ == '__main__':
#     attafeat = torch.rand(100, 128)
#     feat = torch.rand(100, 128)
#
#     cor = pearson_distance(attafeat, attafeat)
#     print(cor)
