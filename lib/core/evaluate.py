# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds, get_max_preds_vnect, get_max_preds_mo2cap2


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    output = np.transpose(output, [0, 3, 1, 2])
    target = np.transpose(target, [0, 3, 1, 2])

    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def accuracy_vnect(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y,z locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    output = np.transpose(output, [0, 3, 1, 2])
    target = np.transpose(target, [0, 3, 1, 2])

    idx = list(range(int(output.shape[1]/4)))

    norm = 1.0
    if hm_type == 'gaussian':
        pred_2d, _, pred_3d = get_max_preds_vnect(output)
        target_2d, _, target_3d = get_max_preds_vnect(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred_2d.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred_2d, target_2d, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred_2d

def accuracy_mo2cap2(output_hmap, target_hmap, output_dist, target_dist, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y,z locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    output_hmap = np.transpose(output_hmap, [0, 3, 1, 2])
    target_hmap = np.transpose(target_hmap, [0, 3, 1, 2])

    idx = list(range(output_hmap.shape[1]))

    norm = 1.0
    if hm_type == 'gaussian':
        pred_2d, _, pred_3d = get_max_preds_mo2cap2(output_hmap, output_dist)
        target_2d, _, target_3d = get_max_preds_mo2cap2(target_hmap, target_dist)
        h = output_hmap.shape[2]
        w = output_hmap.shape[3]
        norm = np.ones((pred_2d.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred_2d, target_2d, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc

    return acc, avg_acc, cnt, pred_2d
