from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac_video as endodac_video

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = opt.min_depth
    MAX_DEPTH = opt.max_depth

    if opt.ext_disp_to_eval is None:
        if not opt.model_type == 'depthanything':
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)

            print("-> Loading weights from {}".format(opt.load_weights_folder))
        else:
            print("Evaluating Depth Anything model")

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)

        frames_to_load = opt.frame_ids.copy()
        matching_ids = [0]
        if opt.use_future_frame:
            matching_ids.append(1)
        for idx in range(-1, -opt.num_frames, -1):
            matching_ids.append(idx)
            if idx not in frames_to_load:
                frames_to_load.append(idx)
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        img_ext = '.png' if opt.png else '.jpg'

        if opt.eval_split == 'endovis':
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            frames_to_load, 4, is_train=False, img_ext=img_ext)
        elif opt.eval_split == 'simcol':
            dataset = datasets.SimcolRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            frames_to_load, 4, is_train=False, img_ext=img_ext)
        elif opt.eval_split == 'hamlyn':
            dataset = datasets.HamlynRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            frames_to_load, 4, is_train=False, img_ext=img_ext)
        elif opt.eval_split == 'c3vd':
            dataset = datasets.C3VDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            frames_to_load, 4, is_train=False, img_ext=img_ext)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.model_type == 'endodac':
            depther = endodac_video.endodac_video_MATA(
                backbone_size = "small", r=opt.lora_rank, lora_type='dvlora',
                image_shape=(int(7/8*opt.height),int(7/8*opt.width)), pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token, num_frames=32)
            model_dict = depther.state_dict()
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda()
            depther.eval()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, allow_pickle=True, encoding='latin1')["data"]

    inference_times = []

    pred_disps = []
    
    errors = []
    ratios = []
    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            image_lists = [data["color", fid, 0].cuda().unsqueeze(dim=1) for fid in range(-opt.num_frames+1, 1)]
            image_lists = torch.cat(image_lists, dim=1)

            if opt.ext_disp_to_eval is None:
                time_start = time.time()
                output = depther(image_lists)
                inference_time = time.time() - time_start
                if opt.model_type == 'endodac' or opt.model_type == 'afsfm':
                    output_disp = output[("disp", 0)]
                pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, -1, 0].numpy()
                pred_disps.append(pred_disp)
                pred_disp = pred_disp[0]

            inference_times.append(inference_time)

            gt_depth = gt_depths[i]

            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1/pred_disp
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            
            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]
            
            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            error = compute_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)
            
        pred_disps = np.concatenate(pred_disps)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
