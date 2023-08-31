"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import re
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils
os.environ["CUDA_VISIBLE_DEVICES"]="5"
@torch.no_grad()
def calculate_metrics(netEC, netG, args, step, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if args.device=='cuda' else 'cpu')

    lpips_dict = OrderedDict()                      ## create a dictionary with order
    src_domain = args.src_domain;
    trg_domain = args.trg_domain;

    if mode == 'reference':
        path_ref = args.target_paths
        loader_ref = get_eval_loader(root=path_ref,
                                     img_size=args.img_size,
                                     src_num=args.src_num,
                                     trg_num=args.trg_num,
                                     is_src=False,
                                     is_count=False,
                                     batch_size=args.val_batch_size,
                                     imagenet_normalize=False,
                                     drop_last=True,
                                     num_workers=args.numworkers)

    path_src = args.source_paths
    loader_src = get_eval_loader(root=path_src,
                                 img_size=args.img_size,
                                 src_num=args.src_num,
                                 trg_num=args.trg_num,
                                 is_src=True,
                                 is_count=False,
                                 batch_size=args.val_batch_size,
                                 imagenet_normalize=False,
                                 drop_last=True,
                                 num_workers=args.numworkers)
    con_enco=re.split('/',args.content_encoder_path)[-1]
    generator=re.split('/',args.generator_path)[-1]
    task = '%s2%s-use:%s_and_%s' % (src_domain, trg_domain,con_enco,generator)
    comptask='%s2%s_comp-use:%s_and_%s' % (src_domain, trg_domain,con_enco,generator)
    path_fake = os.path.join(args.eval_dir, task)
    path_comp=os.path.join(args.eval_dir,comptask)
    shutil.rmtree(path_fake, ignore_errors=True) ## remove a directory
    os.makedirs(path_fake)                       ## create a directory
    shutil.rmtree(path_comp, ignore_errors=True)  ## remove a directory
    os.makedirs(path_comp)  ## create a directory
    lpips_values = []
    print('Generating images and calculating LPIPS for %s...' % task)
  #  calculate_fid_for_all_tasks(args, trg_domain, src_domain, step=step, mode=mode, con_enco=con_enco,
   #                             generator=generator)
    for i, x_src in enumerate(tqdm(loader_src, total=len(loader_src))):
        N = x_src.size(0)                        ## N = batch
        x_src = x_src.to(device)## y_trg shape:[N,1] elements is trg_idx
       ## masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None ## i guess it is not important

        # generate 10 outputs from the same input
        group_of_images = []
        for j in range(args.num_outs_per_domain): ## default is 10
            try:
                x_ref = next(iter_ref).to(device)
            except:
                iter_ref = iter(loader_ref)   ##mode :reference,loader_ref = target
                x_ref = next(iter_ref).to(device)

            if x_ref.size(0) > N:
                x_ref = x_ref[:N]
            print(x_src.shape,x_ref.shape);
            x_cfeat=netEC(x_src.to(device),get_feature=True)
            x_fake,_ = netG(x_cfeat, x_ref.to(device))
            group_of_images.append(x_fake)

            # save generated images to calculate FID later

            for k in range(N):
                filename = os.path.join(
                    path_fake,
                    '%.4i_%.2i.png' % (i*args.val_batch_size+(k+1), j+1))
                compname=os.path.join(
                    path_comp,
                    '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
              #  print(torch.concat([x_src[k].unsqueeze(dim=0),x_ref[k].unsqueeze(dim=0),x_fake[k].unsqueeze(dim=0)],dim=0).shape)
                utils.save_image(x_fake[k],ncol=1,filename=filename);
                utils.save_image(torch.concat([x_src[k].unsqueeze(dim=0),x_ref[k].unsqueeze(dim=0),x_fake[k].unsqueeze(dim=0)],dim=0), ncol=1, filename=compname)

        lpips_value = calculate_lpips_given_images(group_of_images)
        lpips_values.append(lpips_value)

    # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
    lpips_mean = np.array(lpips_values).mean()
    lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

    # delete dataloaders
    del loader_src
    if mode == 'reference':
        del loader_ref
        del iter_ref

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s-use:%s_and_%s.json' % (step, mode,con_enco,generator))
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, trg_domain,src_domain, step=step, mode=mode, con_enco=con_enco,generator=generator)


def calculate_fid_for_all_tasks(args, otrg_domain,osrc_domain, step, mode,con_enco,generator):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    src_domain = osrc_domain
    trg_domain = otrg_domain

    task = '%s2%s-use:%s_and_%s' % (src_domain, trg_domain,con_enco,generator)
    path_real = args.train_img_dir
    path_fake = [os.path.join(args.eval_dir, task)]
    print(path_fake)
    print('Calculating FID for %s...' % task)
    fid_value = calculate_fid_given_paths(
        paths=[path_real, path_fake],
        img_size=args.img_size,
        batch_size=args.val_batch_size)
    fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s-use:%s_and_%s.json' % (step, mode,con_enco,generator))
    utils.save_json(fid_values, filename)

