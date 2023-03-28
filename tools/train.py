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
import pprint
import shutil

import tensorflow as tf
import torch
# import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss, JointsVNectLoss, JointsMo2Cap2Loss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
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

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    with tf.device("/cpu:0"):
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=True
        )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    if 2 <= cfg.GPUS:
        model = tf.keras.utils.multi_gpu_model(model, gpus=cfg.GPUS, cpu_merge=True)

    logger.info(model.summary())

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
    # train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, None)
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*cfg.GPUS,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*cfg.GPUS,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=cfg.TRAIN.LR_STEP,
        values=cfg.TRAIN.LR
    )

    best_perf = tf.Variable(0.0)
    best_model = False
    begin_epoch = tf.Variable(cfg.TRAIN.BEGIN_EPOCH)
    optimizer = get_optimizer(cfg, model, lr_scheduler)

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        best_perf=best_perf,
        begin_epoch=begin_epoch)
    ck_manager = tf.train.CheckpointManager(
        checkpoint, directory=final_output_dir, max_to_keep=1
    )

    status = checkpoint.restore(ck_manager.latest_checkpoint)
    if ck_manager.latest_checkpoint:
        logger.info("=> loaded checkpoint '{}'".format(ck_manager.latest_checkpoint))

    for epoch in range(begin_epoch.value(), cfg.TRAIN.END_EPOCH):
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir)

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        if perf_indicator >= best_perf.value():
            best_perf.assign(perf_indicator)
            model.save(os.path.join(final_output_dir, "best_model"))
            model.save_weights(os.path.join(final_output_dir, "best_weights"))

        begin_epoch.assign(epoch + 1)
        ck_manager.save()

    final_model_state_file = os.path.join(final_output_dir, "final_model")
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    model.save(final_model_state_file)
    model.save_weights(os.path.join(final_output_dir, "final_weights"))

if __name__ == '__main__':
    main()
