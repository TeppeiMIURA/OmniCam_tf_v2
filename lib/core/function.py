# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import tensorflow as tf

from core.evaluate import accuracy, accuracy_vnect, accuracy_mo2cap2
from core.inference import get_final_preds, get_final_preds_omni, get_final_preds_vnect, get_final_preds_mo2cap2
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    for i, (input, target_map, target_weight, target_dist, meta) in enumerate(train_loader):
        input = tf.constant(input.detach().cpu().numpy())

        target_map = tf.constant(
            np.transpose(
                target_map.detach().cpu().numpy(), (0, 2, 3, 1)))

        target_weight = tf.constant(
            target_weight.detach().cpu().numpy())

        target_dist = tf.constant(
            target_dist.detach().cpu().numpy())

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with tf.GradientTape() as tape:
            outputs = model(input, training=True)

            if 'vnect' in config.MODEL.NAME:
                output_map = outputs
                target_map = target_map
                target_weight = target_weight
            elif 'mo2cap2' in config.MODEL.NAME:
                output_map = outputs[0]
                output_dist = outputs[1]
                target_weight = target_weight
                target_map = target_map
                target_dist = target_dist
            else:   # model_resnet, model_base
                output_map = outputs
                target_map = target_map
                target_weight = target_weight

            if 'vnect' in config.MODEL.NAME:
                loss = criterion(output_map, target_map, target_weight)
            elif 'mo2cap2' in config.MODEL.NAME:
                loss = criterion(output_map, target_map, target_weight, output_dist, target_dist)
            else:   # model_resnet, model_base
                loss = criterion(output_map, target_map, target_weight)

        # compute gradient and do update step
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # measure accuracy and record loss
        losses.update(loss.numpy(), tf.shape(input)[0])
		
        if 'vnect' in config.MODEL.NAME:
	        _, avg_acc, cnt, pred = accuracy_vnect(output_map.numpy(),
	                                         target_map.numpy())
        elif 'mo2cap2' in config.MODEL.NAME:
            _, avg_acc, cnt, pred = accuracy_mo2cap2(output_map.numpy(),
	                                         target_map.numpy(),
                                             output_dist.numpy(),
	                                         target_dist.numpy())
        else:   # model_resnet, model_base
	        _, avg_acc, cnt, pred = accuracy(output_map.numpy(),
	                                         target_map.numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=float(input.numpy().shape[0])/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input.numpy(), meta, target_map.numpy(),
                pred * int(config.MODEL.IMAGE_SIZE[0]/config.MODEL.HEATMAP_SIZE[0]),
                output_map.numpy(), prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_preds_3d = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    end = time.time()
    for i, (input, target_map, target_weight, target_dist, meta) in enumerate(val_loader):
        input = tf.constant(input.detach().cpu().numpy())

        target_map = tf.constant(
            np.transpose(
                target_map.detach().cpu().numpy(), (0, 2, 3, 1)))

        target_weight = tf.constant(
            target_weight.detach().cpu().numpy())

        target_dist = tf.constant(
            target_dist.detach().cpu().numpy())

        # compute output
        outputs = model(input)

        if 'vnect' in config.MODEL.NAME:
            output_map = outputs
        elif 'mo2cap2' in config.MODEL.NAME:
            output_map = outputs[0]
            output_dist = outputs[1]
        else:   # model_resnet, model_base
            output_map = outputs
        # if isinstance(outputs, list):
        #     output = outputs[-1]
        # else:
        #     output = outputs

        # if config.TEST.FLIP_TEST:
        #     # this part is ugly, because pytorch has not supported negative index
        #     # input_flipped = model(input[:, :, :, ::-1])
        #     input_flipped = np.flip(input.cpu().numpy(), 3).copy()
        #     input_flipped = torch.from_numpy(input_flipped).cuda()
        #     outputs_flipped = model(input_flipped)

        #     if isinstance(outputs_flipped, list):
        #         output_flipped = outputs_flipped[-1]
        #     else:
        #         output_flipped = outputs_flipped

        #     output_flipped = flip_back(output_flipped.cpu().numpy(),
        #                                 val_dataset.flip_pairs)
        #     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


        #     # feature is not aligned, shift flipped heatmap for higher accuracy
        #     if config.TEST.SHIFT_HEATMAP:
        #         output_flipped[:, :, :, 1:] = \
        #             output_flipped.clone()[:, :, :, 0:-1]

        #     output = (output + output_flipped) * 0.5

        if 'vnect' in config.MODEL.NAME:
            target_map = target_map
            target_weight = target_weight
        elif 'mo2cap2' in config.MODEL.NAME:
            target_map = target_map
            target_weight = target_weight
            target_dist = target_dist
        else:   # model_resnet, model_base
            target_map = target_map
            target_weight = target_weight


        if 'vnect' in config.MODEL.NAME:
            loss = criterion(output_map, target_map, target_weight)
        elif 'mo2cap2' in config.MODEL.NAME:
            loss = criterion(output_map, target_map, target_weight, output_dist, target_dist)
        else:   # model_resnet, model_base
            loss = criterion(output_map, target_map, target_weight)


        num_images = tf.shape(input)[0]
        # measure accuracy and record loss
        losses.update(loss.numpy(), num_images)
        if 'vnect' in config.MODEL.NAME:
            _, avg_acc, cnt, pred = accuracy_vnect(output_map.numpy(),
                                                target_map.numpy())
        elif 'mo2cap2' in config.MODEL.NAME:
            _, avg_acc, cnt, pred = accuracy_mo2cap2(output_map.numpy(),
                                                target_map.numpy(),
                                                output_dist.numpy(),
                                                target_dist.numpy())
        else:   # model_resnet, model_base
            _, avg_acc, cnt, pred = accuracy(output_map.numpy(),
                                                target_map.numpy())

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()

        if 'vnect' in config.MODEL.NAME:
            preds, maxvals, preds_3d = get_final_preds_vnect(
                config, output_map.numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals			

            all_preds_3d[idx:idx + num_images, :, 0:3] = preds_3d[:, :, 0:3]
        elif 'mo2cap2' in config.MODEL.NAME:
            preds, maxvals, preds_3d = get_final_preds_mo2cap2(
                config, output_map.numpy(), output_dist.numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals			

            all_preds_3d[idx:idx + num_images, :, 0:3] = preds_3d[:, :, 0:3]
        elif "omni" == config.DATASET.DATASET:
            preds, maxvals = get_final_preds_omni(
                config, output_map.numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
        else:   # model_resnet, model_base
            preds, maxvals = get_final_preds(
                config, output_map.numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals

        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images

        if i % config.PRINT_FREQ == 0:
            msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            save_debug_images(config, input.numpy(), meta, target_map.numpy(),
                pred * (config.MODEL.IMAGE_SIZE[0]/config.MODEL.HEATMAP_SIZE[0]),
                output_map.numpy(), prefix)

    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, all_preds_3d, output_dir, all_boxes, image_path,
        filenames, imgnums
    )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
