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
import pprint

import tensorflow as tf
import torch
# import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, JointsVNectLoss, JointsMo2Cap2Loss
from core.function import validate
from utils.utils import create_logger

import dataset
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

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model = tf.keras.models.load_model(cfg.TEST.MODEL_FILE)
    else:
        model_state_file = os.path.join(
            final_output_dir, "final_model"
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model = tf.keras.models.load_model(model_state_file)

    if 2 <= cfg.GPUS:
        model = tf.keras.utils.multi_gpu_model(model, gpus=cfg.GPUS, cpu_merge=True)

    # define loss function (criterion) and optimizer
    if 'vnect' in cfg.MODEL.NAME:
	    criterion = JointsVNectLoss(
	        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
	    )
    elif 'mo2cap2' in cfg.MODEL.NAME:
	    criterion = JointsMo2Cap2Loss(
	        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT,
            train_target=cfg.TRAIN.MO2CAP2_TARGET
	    )
    else:   # model_resnet, model_base
	    criterion = JointsMSELoss(
	        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
	    )

    # Data loading code
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, None)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*cfg.GPUS,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
