import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim
import argparse
import os
import sys
sys.path.append('./')
from config import cfg
from torch.backends import cudnn
import numpy as np
from torch.autograd import Variable
from modeling import build_model
from data import train_data_loader
from utils.logger import setup_logger
import cv2


def attack_update(att_img, grad, g, radiu_exp=11):
    radiu = radiu_exp/255.
    norm = torch.sum(torch.abs(grad).view((grad.shape[0], -1)), dim=1).view(-1, 1, 1) + torch.tensor([[[1e-12]], [[1e-12]], [[1e-12]]])
    # norm = torch.max(torch.abs(grad).flatten())
    x_grad = grad / norm
    if torch.isnan(x_grad).any() or torch.isnan(g).any():
        import pdb
        pdb.set_trace()
    g = 0.4*g + x_grad
    att_img = att_img - 0.001*g.sign()
    att_img = att_img * min(1, radiu/(torch.norm(att_img, 2)+1e-6))

    return att_img, g


def least_likely(scores, chosen_class):
    return -torch.sum(torch.gather(scores, 1, chosen_class.view(-1, 1))) / scores.shape[0] # we hope it get large


def train(cfg, radiu_exp):
    attack_img = Variable(torch.rand(3, 256, 128), requires_grad=True)*1e-6
    normalize_transform = T.Normalize(mean=[0, 0, 0], std=cfg.INPUT.PIXEL_STD)
    g = torch.tensor([0.])

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
    train_loader, num_classes = train_data_loader(cfg)
    pre_loss = np.inf
    # cross_entropy =
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
            # attack_score, attack_feat = model(pertubated_images)
            attack_score, attack_feat = model(median_img)
            tmp = (torch.argmax(attack_score, 1) == target).float().mean().data.cpu().numpy()
            print('acc: ', tmp)
            chosen_class = torch.sort(score, dim=1)[1][:, 0]
            # total_loss = label_loss_v1(attack_score, target)
            # import pdb
            # pdb.set_trace()
            # total_loss = least_likely(attack_score, chosen_class)
            # total_loss = loss_fn3(feat, attack_feat)
            # total_loss = map_loss
            total_loss = F.cross_entropy(attack_score, chosen_class)
            total_loss.backward()
            model.zero_grad()
            attack_grad = attack_img.grad.data
            attack_img, g = attack_update(attack_img, attack_grad, g, radiu_exp=radiu_exp)
            # print('tv_loss: ', tvl_loss, 'map_loss: ', map_loss)
            print('total_loss', total_loss)
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
                vis = np.transpose(best_att[::-1, :, :], (1, 2, 0))
                vis = (vis*255 + 10)*10
                 # vis = ((vis - np.min(vis))/(np.max(vis) - np.min(vis)))*255
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
        vis = np.transpose(best_att[::-1, :, :], (1, 2, 0))
        vis = (vis*255 + 10) * 10
       #  vis = ((vis - np.min(vis))/(np.max(vis) - np.min(vis)))*255
        cv2.imwrite(vis_name, vis)



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    # parser.add_argument(
    #     "--weight", default=10., help="balance two losses", type=float,
    # )
    parser.add_argument(
        "--radiu_exp", default=2000., help="balance two losses", type=float,
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    radiu_exp = args.radiu_exp
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # weight = args.weight
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
    train(cfg, radiu_exp)


if __name__ == '__main__':
    main()
