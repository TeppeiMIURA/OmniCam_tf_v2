# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import matplotlib.pyplot as plt

import tensorflow as tf

from load_dataset_tf import load_json_train_dataset

import _init_paths
from config import cfg
from config import update_config

import models

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

    # philly
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
    with tf.device("/cpu:0"):
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )
    model.build(input_shape=(None, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0], 3))
    model.summary()

    if 2 <= cfg.GPUS:
        model = tf.keras.utils.multi_gpu_model(model, gpus=cfg.GPUS, cpu_merge=True)

    # load data
    train_json_file: str = f"{cfg.DATASET.ROOT}/{cfg.DATASET.TRAIN_SET}.json"
    # val_json_file: str = f"{cfg.DATASET.ROOT}/{cfg.DATASET.TEST_SET}.json"

    (train_images, train_targets) = load_json_train_dataset(cfg, train_json_file)
    # (val_images, val_targets) = load_json_train_dataset(cfg, val_json_file)

    print(f"train_images: {train_images.shape}, train_targets: {train_targets.shape}")
    # print(f"val_images: {val_images.shape}, val_targets: {val_targets.shape}")

    # do training
    cfg_name = os.path.basename(args.cfg).split(".")[0]
    output_dir = f"{cfg.OUTPUT_DIR}/{cfg.DATASET.DATASET}/{cfg.MODEL.NAME}/{cfg_name}"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )

    end_epochs = cfg.TRAIN.END_EPOCH
    begin_epochs = cfg.TRAIN.BEGIN_EPOCH
    history = model.fit(
        x=train_images,
        y=train_targets,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        epochs=end_epochs,
        shuffle=True,
        validation_data=None,
        initial_epoch=begin_epochs
    )

    # save trained model
    model.save(output_dir, include_optimizer=False)

    print("finish training!!")

    return

if __name__ == '__main__':
    main()
