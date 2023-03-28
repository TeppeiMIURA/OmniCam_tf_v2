# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

import tensorflow as tf

from load_dataset_tf import load_json_full_dataset
from inference_tf import get_max_preds

import _init_paths
from config import cfg
from config import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    # load model
    model = tf.keras.models.load_model(cfg.TEST.MODEL_FILE)
    model.summary()

    if 2 <= cfg.GPUS:
        model = tf.keras.utils.multi_gpu_model(model, gpus=cfg.GPUS, cpu_merge=True)

    # load data
    val_json_file: str = f"{cfg.DATASET.ROOT}/{cfg.DATASET.TEST_SET}.json"

    (val_images, gt_targets) = load_json_full_dataset(cfg, val_json_file)

    print(f"val_images: {val_images.shape}")

    # do validation
    output_dir = f"{cfg.TEST.MODEL_FILE}/{cfg.DATASET.TEST_SET}"
    os.makedirs(output_dir, exist_ok=True)

    pred_hmaps = np.zeros_like(gt_targets["heatmaps"])
    pred_joints_2d = np.zeros_like(gt_targets["joints_2d"])
    pred_joints_3d = np.zeros_like(gt_targets["joints_3d"])

    for i, in_img in enumerate(val_images):
        in_img = np.expand_dims(in_img, 0)

        out_hmap = model.predict(in_img)
        out_hmap = np.transpose(out_hmap, (0, 3, 1, 2))

        h_hmap, w_hmap = out_hmap.shape[2:]
        pred_2d, maxval = get_max_preds(out_hmap)
        pred_2d = pred_2d / np.array([w_hmap, h_hmap])

        pred_hmaps[i] = np.squeeze(out_hmap).copy()
        pred_joints_2d[i] = np.squeeze(np.concatenate([pred_2d, maxval], 2)).copy()
        # no prediction for joints 3d

    # np.save(f"{output_dir}/val_images.npy", val_images)
    np.save(f"{output_dir}/gts.npy", gt_targets)

    pred_targets: dict = {"heatmaps": pred_hmaps, "joints_2d": pred_joints_2d, "joints_3d": pred_joints_3d}
    np.save(f"{output_dir}/preds.npy", pred_targets)

    print("finish validation!!")

    return

if __name__ == '__main__':
    main()
