import torch

import argparse
import os
import sys
import numpy as np


# from torch.backends import cudnn
from ignite.engine import Engine


sys.path.append('./')
from config import cfg

shuffle = torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=True)
from modeling import Baseline
from data.transforms import build_transforms
from data.datasets import init_dataset, ImageDataset
from data.collate_batch import train_collate_fn, val_collate_fn
from torch.utils.data import DataLoader
import torch.optim


def create_supervised_evaluator(model, metrics):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to('cuda')

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            # data = torch.clamp(data, 0, 1)
            data = data.to('cuda') if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine





def get_map(model, num_query, query_loader, gal_loader):

    from utils.reid_metric import R1_mAP
    from data.datasets.eval_reid import eval_func

    evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)})
    evaluator.run(query_loader)

    qf, q_pids, q_camids = evaluator.state.metrics['r1_mAP']
    evaluator.run(gal_loader)
    gf, g_pids, g_camids = evaluator.state.metrics['r1_mAP']
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP


def freeze_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    # parser.add_argument(
    #     '--resultpath', default='', help='the path to save results', type=str
    # )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    # cudnn.benchmark = True

    num_workers = cfg.DATALOADER.NUM_WORKERS

    
    #   #################     build duke model  ######################################################################
    result = []
    # model_name = ['shufflenet', 'googlenet', 'vgg', 'resnet', 'densenet',  'senet']
    # raw_map = np.array([66.9, 60.8, 66.3, 75.9, 73.4, 66.9]) * 0.01
    model_name = ['resnet', 'densenet', 'vgg',   'senet', 'shufflenet']
    # raw_map = np.array([75.9, 73.4, 66.3, 66.9, 66.9]) * 0.01
    rank1 = np.array([86.98, 85.23, 81.42, 81.33, 81.19]) * 0.01

    # print('testing model: shufflenet, dataset: Duke')



    duke_num_classes = 702


    duke = init_dataset('dukemtmc', root='/data/wenjie')
    duke_num_classes = duke.num_train_pids
    val_query_transforms = build_transforms(cfg, is_train=False, is_attack=True, attack_path=cfg.ATTACK.ATTACK_PATH)#, filter=True, filter_scale=5)
    val_gal_transforms = build_transforms(cfg, is_train=False)
    val_query_set = ImageDataset(duke.query, val_query_transforms)
    val_query_loader = DataLoader(
        val_query_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    duke_num_query = len(duke.query)

    val_gal_set = ImageDataset(duke.gallery, val_gal_transforms)
    val_gal_loader = DataLoader(
        val_gal_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    print('testing model: resnet50, dataset: Duke')
    duke_model = Baseline(duke_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                          cfg.TEST.NECK_FEAT, 'resnet50', cfg.MODEL.PRETRAIN_CHOICE)
    duke_model.load_state_dict(torch.load('models2attack/resnet-duke/resnet50_model_120.pth'), strict=True)
    freeze_norm(duke_model)
    duke_model.to('cuda')
    cmc, mAP = get_map(duke_model, duke_num_query, val_query_loader, val_gal_loader)
    result.append(cmc[0])
    print('resnet50', cmc[0])

    print('testing model: densenet121, dataset: Duke')
    duke_model = Baseline(duke_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                          cfg.TEST.NECK_FEAT, 'densenet121', cfg.MODEL.PRETRAIN_CHOICE)
    duke_model.load_state_dict(torch.load('models2attack/densenet-duke/densenet121_model_120.pth'))
    freeze_norm(duke_model)
    duke_model.to('cuda')
    cmc, mAP = get_map(duke_model, duke_num_query, val_query_loader, val_gal_loader)
    result.append(cmc[0])
    print('densenet121', cmc[0])

    print('testing model: vgg16, dataset: Duke')
    duke_model = Baseline(duke_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                            cfg.TEST.NECK_FEAT, 'vgg16', cfg.MODEL.PRETRAIN_CHOICE)
    duke_model.load_state_dict(torch.load('models2attack/vgg-duke/vgg16_model_120.pth'))
    freeze_norm(duke_model)
    duke_model.to('cuda')
    cmc, mAP = get_map(duke_model, duke_num_query, val_query_loader, val_gal_loader)
    result.append(cmc[0])
    print('vgg', cmc[0])



    print('testing model: senet, dataset: Duke')
    duke_model = Baseline(duke_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                               cfg.TEST.NECK_FEAT, 'senet154', cfg.MODEL.PRETRAIN_CHOICE)
    duke_model.load_state_dict(torch.load('models2attack/senet-duke/senet154_model_120.pth'))
    freeze_norm(duke_model)
    duke_model.to('cuda')
    cmc, mAP = get_map(duke_model, duke_num_query, val_query_loader, val_gal_loader)
    result.append(cmc[0])
    print('senet', cmc[0])

    print('testing model: shufflenet, dataset: Duke')
    duke_model = Baseline(duke_num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                          cfg.TEST.NECK_FEAT, 'shuffle', cfg.MODEL.PRETRAIN_CHOICE)
    duke_model.load_state_dict(torch.load('models2attack/shuffle-duke/shuffle_model_120.pth'))
    freeze_norm(duke_model)
    duke_model.to('cuda')
    cmc, mAP = get_map(duke_model, duke_num_query, val_query_loader, val_gal_loader)
    # print(cfg.ATTACK.ATTACK_PATH, 'test on shufflenet market', 100 * (66.9 - mAP * 100) / 66.9)
    result.append(cmc[0])
    print('shufflenet', cmc[0])


    result = np.array(result)
    drop_rate = (rank1 - result) / rank1


    # result1 = np.array(result)
    # result1 = np.transpose(result1, (1, 0))
    attack_path = cfg.ATTACK.ATTACK_PATH
    new_dir = os.path.dirname(attack_path)
    new_name = os.path.basename(attack_path).split('.')[0]+'_botdukemrdr.npy' #_medianfilter.npy'
    new_path = os.path.join(new_dir, new_name)
    np.save(new_path, drop_rate)


if __name__ == '__main__':
    main()
#
