import torch
import torchvision.transforms as T
import torch.optim
import argparse
import os
import sys
import cv2

sys.path.append('./')
from config import cfg
from torch.backends import cudnn
import numpy as np
from torch.autograd import Variable
from modeling import build_model

from data import train_data_loader
from utils.logger import setup_logger

from layers.attack_loss_v2 import MapLoss, TVLoss


def attack_update(att_img, grad, pre_sat, g, rate=0.8, base=False, i=10, radiu=10):

    norm = torch.sum(torch.abs(grad).view((grad.shape[0], -1)), dim=1).view(-1, 1, 1) + torch.tensor([[[1e-12]], [[1e-12]], [[1e-12]]])
    # norm = torch.max(torch.abs(grad).flatten())
    x_grad = grad / norm
    if torch.isnan(x_grad).any() or torch.isnan(g).any():
        import pdb
        pdb.set_trace()
    g = 0.4*g + x_grad
    att_img = att_img - 0.004*g.sign()
    radiu = radiu / 255.
    att_img = torch.clamp(att_img, -radiu, radiu)

    pre_sat = torch.div(torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
                    torch.tensor(att_img.flatten().size(), dtype=torch.float32))

    if not base:
        img_abs = torch.abs(att_img)
        img_sort = torch.sort(img_abs.flatten(), descending=True)[0]
        new_rate = max(pre_sat, rate)
        if pre_sat < rate and i > 0:
            img_median = img_sort[int((len(img_sort)*new_rate))]
            att_img = att_img * (radiu / (img_median + 1e-6))
            print('median', img_median)
            att_img = torch.clamp(att_img, -radiu, radiu)

    sat = torch.div(torch.sum(torch.eq(torch.abs(att_img), radiu), dtype=torch.float32),
                torch.tensor(att_img.flatten().size(), dtype=torch.float32))

    print('presat:', pre_sat, 'sat: ', sat)

    return att_img, sat, g


def train(cfg, weight, scale_rate, base, radiu):
    torch.manual_seed(1)
    attack_img = Variable(torch.rand(3, 256, 128), requires_grad=True)*1e-6
    normalize_transform = T.Normalize(mean=[0, 0, 0], std=cfg.INPUT.PIXEL_STD)
    g = torch.tensor([0.])
    pre_sat = 1.

    train_loader, num_classes = train_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.ATTACK.ATTACK_WEIGHT), strict=True)
    # model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()
    model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    loss_fn1 = MapLoss()
    loss_fn2 = TVLoss(TVLoss_weight=weight)
    train_loader, num_classes = train_data_loader(cfg)
    pre_loss = np.inf
    for i in range(cfg.ATTACK.MAX_ITER):   #
        print('epoch', i)
        j = 0
        loss = 0
        for images, labels in train_loader:
            j = j + 1
            attack_img = Variable(attack_img, requires_grad=True)
            images = images.to("cuda")
            target = labels.to("cuda")
            normed_atta_img = normalize_transform(attack_img)
            normed_atta_img = normed_atta_img.to('cuda')
            pertubated_images = torch.add(images, normed_atta_img)

            median_img = pertubated_images.to('cuda')
            score, feat = model(images)
      
            attack_score, attack_feat = model(median_img)
            tmp = (torch.argmax(attack_score, 1) == target).float().mean().data.cpu().numpy()
            print('acc: ', tmp)
           
            map_loss = loss_fn1(attack_feat, feat, target, 5)
            
            if base:
                total_loss = map_loss
                print('map_loss', map_loss)
            else:
                tvl_loss = loss_fn2(median_img)
                total_loss = tvl_loss + map_loss
                print('tvl_loss', tvl_loss, 'map_loss', map_loss)
            total_loss.backward()
            model.zero_grad()
            attack_grad = attack_img.grad.data
            attack_img, sat, g = attack_update(attack_img, attack_grad, pre_sat, g, scale_rate, base, i, radiu)
            pre_sat = sat

            loss = loss + total_loss
            avg_loss = loss/j
            print(avg_loss, pre_loss)
            if avg_loss < pre_loss:
                best_att = attack_img.data.numpy()
                np.save(cfg.ATTACK.ATTACK_PATH, best_att)
                print('saved')
                best_att = attack_img.data.numpy()
                npy_name = cfg.ATTACK.ATTACK_PATH
                vis_name = os.path.join(os.path.dirname(npy_name),
                                        os.path.basename(npy_name).split('.')[0] + '_vis.jpg')
                np.save(npy_name, best_att)
                vis = np.transpose(best_att, (1, 2, 0))
                vis = vis * 255
                vis = (10 + vis) * 10
                cv2.imwrite(vis_name, vis)

        if avg_loss < pre_loss:
            pre_loss = avg_loss

    if not os.path.exists(cfg.ATTACK.ATTACK_PATH):
        best_att = attack_img.data.numpy()
        np.save(cfg.ATTACK.ATTACK_PATH, best_att)
        print('saved')
        best_att = attack_img.data.numpy()
        npy_name = cfg.ATTACK.ATTACK_PATH
        vis_name = os.path.join(os.path.dirname(npy_name),
                                os.path.basename(npy_name).split('.')[0] + '_vis.jpg')
        np.save(npy_name, best_att)
        vis = np.transpose(best_att, (1, 2, 0))
        vis = vis * 255
        vis = (10 + vis) * 10
        cv2.imwrite(vis_name, vis)


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--weight", default=10., help="balance two losses", type=float,
    )
    parser.add_argument(
        "--scale_rate", default=0.8, help="balance two losses", type=float,
    )
    parser.add_argument(
        "--radiu", default=10, help="balance two losses", type=float,
    )
    parser.add_argument(
        "--base", action='store_true'
    )  # (default=False)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    weight = args.weight
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    print('base ? ', args.base)
    train(cfg, weight, args.scale_rate, args.base, args.radiu)


if __name__ == '__main__':
    main()
