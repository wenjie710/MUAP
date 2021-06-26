# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .triplet_alignedreid import TripletLossAlignedReID


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if cfg.MODEL.NAME == 'mgn':
        def loss_func(scores, feats, target):
            total_loss = 0
            triplet_weight = cfg.MGN.WEIGHT
            for score in scores:
                # import pdb
                # pdb.set_trace()
                softmax_loss = xent(score, target)
                total_loss += softmax_loss
            for feat in feats:
                trip_loss = triplet(feat, target)[0]
                total_loss += trip_loss * triplet_weight
            return total_loss

    elif cfg.MODEL.NAME == 'pcb' or cfg.MODEL.NAME == 'pcbdense':
        def loss_func(scores, feats, target):
            total_loss = 0
            triplet_weight = cfg.PCB.WEIGHT
            for score in scores:
                # import pdb
                # pdb.set_trace()
                softmax_loss = xent(score, target)
                total_loss += softmax_loss
            for feat in feats:
                trip_loss = triplet(feat, target)[0]
                total_loss += trip_loss * triplet_weight
            return total_loss
    elif cfg.MODEL.NAME == 'aligned':
        triplet_aligned = TripletLossAlignedReID()

        def loss_func(score, feats, target):
            # total_loss = 0

            xent_loss = xent(score, target)
            # total_loss += xent_loss
            trip_loss = triplet_aligned(feats, target)
            total_loss = xent_loss + trip_loss
            return  total_loss
    elif cfg.MODEL.NAME == 'hacnn':
        def loss_func(scores, feats, target):
            softmax_loss = 0
            trip_loss = 0
            for score in scores:
                # import pdb
                # pdb.set_trace()
                softmax_loss += xent(score, target)
                softmax_loss = softmax_loss / 2
            for feat in feats:
                trip_loss += triplet(feat, target)[0]
                trip_loss = trip_loss / 2
            total_loss = softmax_loss + trip_loss
            return total_loss

    elif sampler == 'softmax':
        if cfg.MODEL.NAME == 'pcb':
            def loss_func(score, feat, target):
                import torch
                loss = torch.tensor([0]).float().to('cuda')
                for i in range(len(score)):
                    loss += F.cross_entropy(score[i], target)
                # import pdb
                # pdb.set_trace()
                return loss
        else:
            def loss_func(score, feat, target):
                return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + cluster(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]    # new add by luo, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    elif cfg.MODEL.NAME == 'googlenet':
        feat_dim = 1536
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, '
              'range_center,triplet_center, triplet_range_center '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0] # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]     # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, no label smooth

        else:
            print('expected METRIC_LOSS_TYPE with center should be center,'
                  ' range_center, triplet_center, triplet_range_center '
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion


def make_attack_loss(cfg):

    pass

    def loss_func(score, feat, target):
        pass

