#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17 15:00
# @Author  : Hao Luo
# @File    : msmt17.py

import glob
import re
from collections import defaultdict
import os.path as osp
import random
from .bases import BaseImageDataset


def random_choose(train_set, num_id, num_img):
    random.seed(1)
    index_dic = defaultdict(list)
    num_imgs_per_id = int(num_img / num_id)
    assert num_img % num_id == 0
    for i, (_, pid, cam) in enumerate(train_set):
        index_dic[pid].append(i)
    pids = list(index_dic.keys())
    random.shuffle(pids)
    choose_pid = pids[:num_id]
    new_index_dic = defaultdict(list)
    for id in choose_pid:
        inds = index_dic[id]
        random.shuffle(inds)
        new_inds = inds[:num_imgs_per_id]
        new_index_dic[id] = new_inds
    new_train = []
    for key, value in new_index_dic.items():
        for ind in value:
            new_train.append(train_set[ind])
    return new_train

class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'msmt17'

    def __init__(self,root='/home/haoluo/data', verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        new_train = random_choose(train, 200, 800)
        self.train = new_train

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset
