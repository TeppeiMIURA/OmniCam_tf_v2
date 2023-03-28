# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_max_preds_vnect(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps and location-maps: numpy.ndarray([batch_size, num_joints * 4, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = int(batch_heatmaps.shape[0])
    num_joints = int(batch_heatmaps.shape[1] / 4)
    width = int(batch_heatmaps.shape[3])

    heatmaps_pred = batch_heatmaps[:, 0:num_joints]
    loc_maps_x_pred = batch_heatmaps[:, num_joints:num_joints*2]
    loc_maps_y_pred = batch_heatmaps[:, num_joints*2:num_joints*3]
    loc_maps_z_pred = batch_heatmaps[:, num_joints*3:num_joints*4]

    heatmaps_reshaped = heatmaps_pred.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds_2d = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds_2d[:,:,0] = (preds_2d[:,:,0]) % width
    preds_2d[:,:,1] = np.floor((preds_2d[:,:,1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds_2d *= pred_mask

    preds_3d = np.zeros((batch_size, num_joints, 3), dtype=np.float)
    for b in range(batch_size):
        for j in range(num_joints):
            h = int(preds_2d[b, j, 1])
            w = int(preds_2d[b, j, 0])

            preds_3d[b, j, 0] = loc_maps_x_pred[b, j, h, w]
            preds_3d[b, j, 1] = loc_maps_y_pred[b, j, h, w]
            preds_3d[b, j, 2] = loc_maps_z_pred[b, j, h, w]

    return preds_2d, maxvals, preds_3d


def cam2world(x, y, hmap_width, hmap_height):
    # reverse 2d to 3d for omnidirectional camera 736x368
    IN_IMAGE_WIDTH = 736
    IN_IMAGE_HEIGHT = 368
    x = np.clip((x+0.5)*(IN_IMAGE_WIDTH / hmap_width), 0, IN_IMAGE_WIDTH - 2)
    y = np.clip((y+0.5)*(IN_IMAGE_HEIGHT / hmap_height), 0, IN_IMAGE_HEIGHT - 2)
    pol = [-1.143953e+02, 0.000000e+00, 3.064861e-03, -5.460688e-06, 5.940485e-08]  # coeffient for 736x368
    xc = 183
    yc = 183

    c = 1.0
    d = 0.0
    e = 0.0
    invdet = 1 / (c-d*e)

    direction = np.ones((x.shape[0], x.shape[1], 3), dtype=np.float)

    lower_index = np.where(x < (int(IN_IMAGE_WIDTH/2)-1))
    upper_index = np.where(x >= (int(IN_IMAGE_WIDTH/2)-1))

    direction[lower_index] = np.array([1, -1, -1], dtype=np.float)
    direction[upper_index] = np.array([-1, -1, 1], dtype=np.float)
    x[upper_index] = x[upper_index] - (int(IN_IMAGE_WIDTH/2)-1)

    xp = invdet * ((x - xc) - d*(y - yc))
    yp = invdet * (-e*(x - xc) + c*(y - yc))

    r = np.sqrt(xp*xp + yp*yp)
    zp = np.full((xp.shape[0], xp.shape[1]), pol[0], dtype=np.float)
    r_i = np.ones_like(xp, dtype=np.float)

    for i_pol in pol[1:]:
        r_i = r_i * r
        zp = zp + (r_i * i_pol)

    xyz3d = np.stack([xp, yp, zp], axis=2)
    invnorm = np.ones_like(xyz3d, dtype=np.float) / np.linalg.norm(xyz3d, ord=2, axis=2, keepdims=True)
    xyz3d = direction * xyz3d * invnorm

    # convert x-y-z axis on camera coordinates to real world coordinates
    world_xyz3d = xyz3d[:, :, [0,2,1]]

    return world_xyz3d

def get_max_preds_mo2cap2(batch_heatmaps, batch_dist):
    '''
    get predictions from score maps
    heatmaps and location-maps: numpy.ndarray([batch_size, num_joints * 4, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    hmap_height = batch_heatmaps.shape[2]
    hmap_width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds_2d = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds_2d[:, :, 0] = (preds_2d[:, :, 0]) % hmap_width
    preds_2d[:, :, 1] = np.floor((preds_2d[:, :, 1]) / hmap_width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds_2d *= pred_mask

    preds_3d = np.zeros((batch_size, num_joints, 3), dtype=np.float)

    xyz3d = cam2world(preds_2d[:,:,0], preds_2d[:,:,1], hmap_width, hmap_height)

    preds_3d[:,:,:] = xyz3d[:,:,:] * batch_dist[:,:, np.newaxis]

    return preds_2d, maxvals, preds_3d

def get_final_preds(config, batch_heatmaps, center, scale):
    batch_heatmaps = np.transpose(batch_heatmaps, [0, 3, 1, 2])

    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()
    # Transform back

    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def get_final_preds_omni(config, batch_heatmaps, center, scale):
    batch_heatmaps = np.transpose(batch_heatmaps, [0, 3, 1, 2])

    coords, maxvals = get_max_preds(batch_heatmaps)

    preds = coords.copy()

    return preds, maxvals

def get_final_preds_vnect(config, batch_heatmaps, center, scale):
    batch_heatmaps = np.transpose(batch_heatmaps, [0, 3, 1, 2])

    preds_2d, maxvals, preds_3d = get_max_preds_vnect(batch_heatmaps)

    return preds_2d, maxvals, preds_3d

def get_final_preds_mo2cap2(config, batch_heatmaps, batch_dist,center, scale):
    batch_heatmaps = np.transpose(batch_heatmaps, [0, 3, 1, 2])

    preds_2d, maxvals, preds_3d = get_max_preds_mo2cap2(batch_heatmaps, batch_dist)

    return preds_2d, maxvals, preds_3d
