# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .dukemtmcreid_attack import DukeMTMCreID_attack
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dataset_loader import ImageDataset
from .small_market import SmallMarket
from .market1501_attack import Market1501_attack
from .msmt17_attack import MSMT17_attack

__factory = {
    'market1501': Market1501,
    # 'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'dukemtmc_attack': DukeMTMCreID_attack,
    'msmt17': MSMT17,
    'msmt17_attack': MSMT17_attack,
    'small_market': SmallMarket,
    'market1501_attack': Market1501_attack,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
