# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing
from . transforms import AddAttack, MultiplyAttack
from .transforms import Filter, MA, MulAdd


def build_transforms(cfg, is_train=True, is_attack=False, attack_path=None, filter=False, filter_scale=5):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_attack:
        if filter:
            transform = T.Compose([
                    T.Resize(cfg.INPUT.SIZE_TEST),
                    T.ToTensor(),
                    AddAttack(attack_path),
                    Filter(filter_scale, mode='median'),
                    normalize_transform
                ])
        else:
            transform = T.Compose([
                    T.Resize(cfg.INPUT.SIZE_TEST),
                    T.ToTensor(),
                    AddAttack(attack_path),
                    normalize_transform
                ])

    else:
        if is_train:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                normalize_transform,
            ])
    return transform


def transforms_multiply_query(cfg, attack_path):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = T.Compose([
                    T.Resize(cfg.INPUT.SIZE_TEST),
                    T.ToTensor(),
                    MultiplyAttack(attack_path),
                    normalize_transform
                    ])
    return transform


def transforms_MA_query(cfg, attack_path, radiu, L2=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    transform = T.Compose([
                    T.Resize(cfg.INPUT.SIZE_TEST),
                    T.ToTensor(),
                    MulAdd(attack_path, radiu, L2=L2),
                    normalize_transform
                    ])
    return transform


def build_transforms_spec(cfg):
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor()])
    return transform


def build_transform_multiply(cfg):
    # import pdb
    # pdb.set_trace()
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor()
    ])
    return transform
