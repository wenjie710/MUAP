# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_query_transforms = build_transforms(cfg, is_train=False, is_attack=True, attack_path=cfg.ATTACK.ATTACK_PATH)
    val_gal_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_query_set = ImageDataset(dataset.query, val_query_transforms)
    val_query_loader = DataLoader(
        val_query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    val_gal_set = ImageDataset(dataset.gallery, val_gal_transforms)
    val_gal_loader = DataLoader(
        val_gal_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_query_loader, val_gal_loader, len(dataset.query), num_classes


def train_data_loader(cfg, train_attack=False):
    if cfg.DATASETS.NAMES == 'small_market':
        import torchvision.transforms as T
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor()]
        )
    else:
        train_transforms = build_transforms(cfg, is_train=True, is_attack=train_attack, attack_path=cfg.ATTACK.ATTACK_PATH)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        if cfg.DATASETS.NAMES == 'small_market':
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=train_collate_fn)
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )

    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    return train_loader, num_classes


def train_data_loader_new(cfg, train_attack=False, batch=64):  # easier to change batch
    train_transforms = build_transforms(cfg, is_train=True, is_attack=train_attack, attack_path=cfg.ATTACK.ATTACK_PATH)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=batch, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    return train_loader, num_classes


def val_data_loader(cfg):
    val_query_transforms = build_transforms(cfg, is_train=False, is_attack=cfg.ATTACK.IS_ATTACK, attack_path=cfg.ATTACK.ATTACK_PATH)
    val_gal_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    val_query_set = ImageDataset(dataset.query, val_query_transforms)
    val_query_loader = DataLoader(
        val_query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    val_gal_set = ImageDataset(dataset.gallery, val_gal_transforms)
    val_gal_loader = DataLoader(
        val_gal_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_query_loader, val_gal_loader, num_classes






# def make_data_loader(cfg):
#     train_transforms = build_transforms(cfg, is_train=True)
#     val_transforms = build_transforms(cfg, is_train=False)
#     num_workers = cfg.DATALOADER.NUM_WORKERS
#     if len(cfg.DATASETS.NAMES) == 1:
#         dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
#     else:
#         # TODO: add multi dataset to train
#         dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
#
#     num_classes = dataset.num_train_pids
#     train_set = ImageDataset(dataset.train, train_transforms)
#     if cfg.DATALOADER.SAMPLER == 'softmax':
#         train_loader = DataLoader(
#             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
#             collate_fn=train_collate_fn
#         )
#     else:
#         train_loader = DataLoader(
#             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
#             sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
#             # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
#             num_workers=num_workers, collate_fn=train_collate_fn
#         )
#
#     val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
#     val_loader = DataLoader(
#         val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
#         collate_fn=val_collate_fn
#     )
#
#     return train_loader, val_loader, len(dataset.query), num_classes