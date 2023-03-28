# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class JointsMSELoss(tf.keras.Model):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.use_target_weight = use_target_weight

    def call(self, output, target, target_weight):
        loss = self.criterion(output, target)
        
        return loss


class JointsVNectLoss(tf.keras.Model):
    def __init__(self, use_target_weight):
        super(JointsVNectLoss, self).__init__()
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.use_target_weight = use_target_weight

    def call(self, output, target, target_weight):
        num_joints = int(tf.shape(output)[3] / 4)

        hmaps_pred = output[:,:,:, 0:num_joints]
        hmaps_gt = target[:,:,:, 0:num_joints]
        x_lmaps_pred = output[:,:,:, num_joints:num_joints*2]
        x_lmaps_gt = target[:,:,:, num_joints:num_joints*2]
        y_lmaps_pred = output[:,:,:, num_joints*2:num_joints*3]
        y_lmaps_gt = target[:,:,:, num_joints*2:num_joints*3]
        z_lmaps_pred = output[:,:,:, num_joints*3:num_joints*4]
        z_lmaps_gt = target[:,:,:, num_joints*3:num_joints*4]

        loss = 0
        loss += self.criterion(hmaps_pred, hmaps_gt)
        loss += self.criterion(tf.math.multiply(hmaps_gt, x_lmaps_pred), tf.math.multiply(hmaps_gt, x_lmaps_gt))
        loss += self.criterion(tf.math.multiply(hmaps_gt, y_lmaps_pred), tf.math.multiply(hmaps_gt, y_lmaps_gt))
        loss += self.criterion(tf.math.multiply(hmaps_gt, z_lmaps_pred), tf.math.multiply(hmaps_gt, z_lmaps_gt))

        return loss

class JointsMo2Cap2Loss(tf.keras.Model):
    def __init__(self, use_target_weight, train_target):
        super(JointsMo2Cap2Loss, self).__init__()
        self.criterion = tf.keras.losses.MeanSquaredError()
        self.use_target_weight = use_target_weight
        self.train_target = train_target

    def call(self, output_hmap, target_hmap, target_weight, output_dist, target_dist):
        # return loss
        loss = self.criterion(output_hmap, target_hmap)

        # loss for depth
        if 'depth' in self.train_target:
            loss += self.criterion(output_dist, target_dist)

        return loss


# class JointsOHKMMSELoss(nn.Module):
#     def __init__(self, use_target_weight, topk=8):
#         super(JointsOHKMMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='none')
#         self.use_target_weight = use_target_weight
#         self.topk = topk

#     def ohkm(self, loss):
#         ohkm_loss = 0.
#         for i in range(loss.size()[0]):
#             sub_loss = loss[i]
#             topk_val, topk_idx = torch.topk(
#                 sub_loss, k=self.topk, dim=0, sorted=False
#             )
#             tmp_loss = torch.gather(sub_loss, 0, topk_idx)
#             ohkm_loss += torch.sum(tmp_loss) / self.topk
#         ohkm_loss /= loss.size()[0]
#         return ohkm_loss

#     def forward(self, output, target, target_weight):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

#         loss = []
#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss.append(0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 ))
#             else:
#                 loss.append(
#                     0.5 * self.criterion(heatmap_pred, heatmap_gt)
#                 )

#         loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
#         loss = torch.cat(loss, dim=1)

#         return self.ohkm(loss)
