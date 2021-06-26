import glob
import re
import os.path as osp

# from .bases import BaseImageDataset


class SmallMarket(object):


    dataset_dir = 'small_market'

    def __init__(self, root='/data/wenjie/', verbose=True, **kwargs):
        super(SmallMarket, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        if verbose:
            print("=> Market1501 loaded")

        self.train = train
        self.num_train_pids = 751

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            # if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
