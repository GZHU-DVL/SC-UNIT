"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils
from metrics.eval import calculate_metrics
from model.generator import Generator
from model.content_encoder import ContentEncoder
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="metric for Adversarial Image Translation of GP-UNIT")
        self.parser.add_argument("--numworkers", type = int, default=1,
                                help='the num of cpuworker')
        self.parser.add_argument("--content_encoder_path", type=str, default='./checkpoint/content_encoder.pt',
                                 help="path to the saved content encoder")
        self.parser.add_argument("--generator_path", type=str, default='./checkpoint/cat2dog.pt',
                                 help="path to the saved generator")
        self.parser.add_argument("--val_batch_size", type=int, default=8,
                                 help="path to the identity model")
        self.parser.add_argument("--src_num",type=int,nargs='+',default=[9999999],
                                 help="the number of pictures from sources_domain")
        self.parser.add_argument("--trg_num",type=int,nargs='+',default=[9999999],
                                 help="the number of pictures from target_domain")
        self.parser.add_argument("--source_paths", type=str, nargs='+',
                                 help="the paths to source domain")
        self.parser.add_argument("--target_paths", type=str, nargs='+',
                                 help="the paths to target domain")
        self.parser.add_argument('--val_img_dir', type=str, default='/data/binxin/dataset/afhq/val',
                                 help='Directory containing validation images')
        self.parser.add_argument('--eval_dir', type=str, default='/data/binxin/GP-UNIT-main/eval',
                                 help='Directory for saving metrics, i.e., FID and LPIPS')
        self.parser.add_argument('--train_img_dir',type=str,nargs='+')
        self.parser.add_argument('--img_size', type=int, default=256,
                                 help='Image resolution')
        self.parser.add_argument('--device', type=str, default='cuda',
                                 help="'cpu' for using cpu and 'GPU' for using GPU")
        self.parser.add_argument('--trg_domain', type=str, default='dog')
        self.parser.add_argument('--src_domain', type=str, default='cat')
        self.parser.add_argument('--num_outs_per_domain',type=int, default=10,
                                  help='Number of generated images per domain during sampling')
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

if __name__ == "__main__":
    parser = TrainOptions()
    args = parser.parse()
    device=args.device
    netEC=ContentEncoder()
    netEC.eval()
    netG=Generator()
    netG.eval()
    netEC.load_state_dict(torch.load(args.content_encoder_path,map_location=lambda storage,loc:storage))
    ckpt = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
    netG.load_state_dict(ckpt['g_ema'])

    netEC = netEC.to(device)
    netG = netG.to(device)
    calculate_metrics(netEC,netG,args,0,'reference')
    print('Load models successfully')


