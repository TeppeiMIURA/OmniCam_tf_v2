from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def get_max_preds(hmaps:np.ndarray) -> (np.ndarray, np.ndarray):
    '''
    get predictions from score maps
    hmaps: numpy.ndarray([batch_size, num_joints, height, width])
    return numpy.ndarray([batch_size, num_joints, [width_ratio, height_ratio])
            numpy.ndarray([batch_size, num_joints, maxval)
    '''
    batch_size = hmaps.shape[0]
    num_joints = hmaps.shape[1]
    width = hmaps.shape[3]

    heatmaps_reshaped = hmaps.reshape((batch_size, num_joints, -1))
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
