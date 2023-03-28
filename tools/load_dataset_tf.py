from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import json

from tensorflow.python.keras.backend import dtype

def generate_hmap(cfg, joints:np.ndarray) -> np.ndarray:
    num_joints = joints.shape[0]
    target = np.zeros((num_joints,
                        cfg.MODEL.HEATMAP_SIZE[1],
                        cfg.MODEL.HEATMAP_SIZE[0]),
                        dtype=np.float32)

    tmp_size = cfg.MODEL.SIGMA * 3

    joints = joints * np.array(cfg.MODEL.IMAGE_SIZE)

    for joint_id in range(num_joints):
        feat_stride = np.array(cfg.MODEL.IMAGE_SIZE) / np.array(cfg.MODEL.HEATMAP_SIZE)
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * cfg.MODEL.SIGMA ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], cfg.MODEL.HEATMAP_SIZE[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], cfg.MODEL.HEATMAP_SIZE[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], cfg.MODEL.HEATMAP_SIZE[0])
        img_y = max(0, ul[1]), min(br[1], cfg.MODEL.HEATMAP_SIZE[1])

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target


def load_json_train_dataset(cfg, json_file:str) -> (np.ndarray, np.ndarray):
    with open(json_file, "r") as jf:
        df = json.load(jf)

        input_path = df["input"]
        target_path = df["target"]

    num_data = len(input_path)

    input_images = np.zeros((num_data,
                                cfg.MODEL.IMAGE_SIZE[1],
                                cfg.MODEL.IMAGE_SIZE[0],
                                3),
                                dtype=np.float32)
    target_hmaps = np.zeros((num_data,
                                cfg.MODEL.NUM_JOINTS,
                                cfg.MODEL.HEATMAP_SIZE[1],
                                cfg.MODEL.HEATMAP_SIZE[0]))

    for i, (input, target) in enumerate(zip(input_path, target_path)):
        input = f"{cfg.DATASET.ROOT}/{input}"
        target = f"{cfg.DATASET.ROOT}/{target}"

        image = Image.open(input)
        image = image.resize((cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]))
        input_images[i] = image.copy()

        with open(target, "r") as jf:
            json_data = json.load(jf)
            full_joints_2d = json_data["joints2D"]

        joints_2d = np.zeros((cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        for i_joint, joint in enumerate(["Head", "Neck", "Spine1", "Pelvis",
                                            "L_Hand", "L_Wrist", "L_Elbow", "L_Shoulder",
                                            "R_Hand", "R_Wrist", "R_Elbow", "R_Shoulder",
                                            "L_Hip", "R_Hip"]):
            joints_2d[i_joint] = full_joints_2d[joint]

        hmap = generate_hmap(cfg, joints_2d)
        target_hmaps[i] = hmap.copy()

    input_images = input_images / 255.0
    target_hmaps = np.transpose(target_hmaps, (0, 2, 3, 1))

    return (input_images, target_hmaps)


def load_json_full_dataset(cfg, json_file:str) -> (np.ndarray, dict):
    with open(json_file, "r") as jf:
        df = json.load(jf)

        input_path = df["input"]
        target_path = df["target"]

    num_data = len(input_path)

    input_images = np.zeros((num_data,
                                cfg.MODEL.IMAGE_SIZE[1],
                                cfg.MODEL.IMAGE_SIZE[0],
                                3),
                                dtype=np.float32)
    target_hmaps = np.zeros((num_data,
                                cfg.MODEL.NUM_JOINTS,
                                cfg.MODEL.HEATMAP_SIZE[1],
                                cfg.MODEL.HEATMAP_SIZE[0]),
                                dtype=np.float32)
    target_joints_2d = np.ones((num_data, cfg.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    target_joints_3d = np.ones((num_data, cfg.MODEL.NUM_JOINTS, 3), dtype=np.float32)

    for i, (input, target) in enumerate(zip(input_path, target_path)):
        input = f"{cfg.DATASET.ROOT}/{input}"
        target = f"{cfg.DATASET.ROOT}/{target}"

        image = Image.open(input)
        image = image.resize((cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]))
        input_images[i] = image.copy()

        with open(target, "r") as jf:
            json_data = json.load(jf)
            full_joints_2d = json_data["joints2D"]
            full_joints_3d = json_data["joints3D"]

        joints_2d = np.zeros((cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        for i_joint, joint in enumerate(["Head", "Neck", "Spine1", "Pelvis",
                                            "L_Hand", "L_Wrist", "L_Elbow", "L_Shoulder",
                                            "R_Hand", "R_Wrist", "R_Elbow", "R_Shoulder",
                                            "L_Hip", "R_Hip"]):
            joints_2d[i_joint] = full_joints_2d[joint]

        hmap = generate_hmap(cfg, joints_2d)

        joints_3d = np.zeros((cfg.MODEL.NUM_JOINTS, 3), dtype=np.float32)
        for i_joint, joint in enumerate(["Head", "Neck", "Spine1", "Pelvis",
                                            "L_Hand", "L_Wrist", "L_Elbow", "L_Shoulder",
                                            "R_Hand", "R_Wrist", "R_Elbow", "R_Shoulder",
                                            "L_Hip", "R_Hip"]):
            joints_3d[i_joint] = full_joints_3d[joint]

        target_hmaps[i] = hmap.copy()
        target_joints_2d[i, :,:2] = joints_2d.copy()
        target_joints_3d[i] = joints_3d.copy()

    input_images = input_images / 255.0
    full_targets: dict = {"heatmaps": target_hmaps, "joints_2d": target_joints_2d, "joints_3d": target_joints_3d}

    return (input_images, full_targets)