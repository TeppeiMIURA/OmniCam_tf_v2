# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json_tricks as json
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from dataset.JointsDatasetOmni import JointsDatasetOmni


logger = logging.getLogger(__name__)


class OMNIDataset(JointsDatasetOmni):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 14
        self.flip_pairs = []
        self.joints_order = [
            "Head", "Neck", "Spine1", "Pelvis",
            "L_Hand", "L_Wrist", "L_Elbow", "L_Shoulder",
            "R_Hand", "R_Wrist", "R_Elbow", "R_Shoulder",
            "L_Hip",
            "R_Hip"
        ]
        self.parent_ids = {
            "Head":"Neck", "Neck":"Neck", "Spine1":"Neck", "Pelvis":"Spine1",
            "L_Hand":"L_Wrist", "L_Wrist":"L_Elbow", "L_Elbow":"L_Shoulder", "L_Shoulder":"Neck",
            "R_Hand":"R_Wrist", "R_Wrist":"R_Elbow", "R_Elbow":"R_Shoulder", "R_Shoulder":"Neck",
            "L_Hip":"Pelvis",
            "R_Hip":"Pelvis"
        }

        self.upper_body_ids = ("Head", "Neck", "Spine1", "L_Hand", "L_Wrist", "L_Elbow", "L_Shoulder", "R_Hand", "R_Wrist", "R_Elbow", "R_Shoulder")
        self.lower_body_ids = ("Pelvis", "L_Hip", "R_Hip")

        self.image_width = cfg.MODEL.IMAGE_SIZE[0] 
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(
            self.root, self.image_set+'.json'
        )
        with open(file_name) as lists_file:
            lists = json.load(lists_file)
            input_list = lists['input']
            target_list = lists['target']

        c = np.array([self.image_width/2, self.image_height/2], dtype=np.float)
        s = np.array([1.0, 1.0], dtype=np.float)

        gt_db = []
        for input_img, target_file in zip(input_list, target_list):
            image_name = input_img

            with open(os.path.join(self.root, target_file)) as target_json:
                target = json.load(target_json)
                target_2d = target['joints2D']

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float)

            for i, j in enumerate(self.joints_order):
                joints_3d[i] = np.array([self.image_width*target_2d[j][0],
                                        self.image_height*target_2d[j][1],
                                        0.0])

            gt_db.append(
                {
                    'image': os.path.join(self.root, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, preds_3d, output_dir, *args, **kwargs):
        preds = preds[:, :, 0:2]

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'pred_2d': preds})

        # if 'test' in cfg.DATASET.TEST_SET:
        #     return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        test_file = os.path.join(cfg.DATASET.ROOT,
                                cfg.DATASET.TEST_SET+'.json')

        with open(test_file) as test_list:
            lists = json.load(test_list)
            target_list = lists['target']

        gt_2d = []
        gt_3d = []
        for target_file in target_list:
            with open(os.path.join(self.root, target_file)) as target_json:
                target = json.load(target_json)
                target_2d = target['joints2D']
                target_3d = target["joints3D"]

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones((self.num_joints, 3), dtype=np.float)

            joints_2d = np.zeros((self.num_joints, 2), dtype=np.float)
            joints_2d_vis = np.ones((self.num_joints, 2), dtype=np.float)

            for i, j in enumerate(self.joints_order):
                joints_3d[i] = np.array([target_3d[j][0],
                                        target_3d[j][1],
                                        target_3d[j][2]])
                joints_2d[i] = np.array([cfg.MODEL.HEATMAP_SIZE[0]*target_2d[j][0],
                                        cfg.MODEL.HEATMAP_SIZE[1]*target_2d[j][1]])
            gt_2d.append(joints_2d)
            gt_3d.append(joints_3d)

        gt_2d = np.array(gt_2d)
        gt_3d = np.array(gt_3d)
        if output_dir:
            gt_file = os.path.join(output_dir, 'gt.mat')
            savemat(gt_file, mdict={'gt_2d': gt_2d, 'gt_3d': gt_3d})

        pos_gt_src = np.transpose(gt_2d, [1, 2, 0])
        pos_pred_src = np.transpose(preds, [1, 2, 0])

        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = 0.1
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        jnt_visible = np.ones_like(scaled_uv_err, dtype=np.uint8)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), self.num_joints))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[self.joints_order.index("Head")]),
            ('Neck', PCKh[self.joints_order.index("Neck")]),
            ('Spine', PCKh[self.joints_order.index("Spine1")]),
            ('Pelvis', PCKh[self.joints_order.index("Pelvis")]),
            ('Shoulder', 0.5 * (PCKh[self.joints_order.index("L_Shoulder")] + PCKh[self.joints_order.index("R_Shoulder")])),
            ('Elbow', 0.5 * (PCKh[self.joints_order.index("L_Elbow")] + PCKh[self.joints_order.index("R_Elbow")])),
            ('Wrist', 0.5 * (PCKh[self.joints_order.index("L_Wrist")] + PCKh[self.joints_order.index("R_Wrist")])),
            ('Hand', 0.5 * (PCKh[self.joints_order.index("L_Hand")] + PCKh[self.joints_order.index("R_Hand")])),
            ('Hip', 0.5 * (PCKh[self.joints_order.index("L_Hip")] + PCKh[self.joints_order.index("R_Hip")])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
