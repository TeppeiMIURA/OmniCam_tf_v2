# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

def get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 'same'
        output_padding = None
    elif deconv_kernel == 3:
        padding = 'same'
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 'valid'
        output_padding = None

    return deconv_kernel, padding, output_padding

def make_deconv_layer(num_layers, num_filters, num_kernels, deconv_with_bias):
    assert num_layers == len(num_filters), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    assert num_layers == len(num_kernels), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'

    layers = []
    for i in range(num_layers):
        kernel, padding, output_padding = \
            get_deconv_cfg(num_kernels[i], i)

        planes = num_filters[i]
        layers.append(
            tf.keras.layers.Conv2DTranspose(
                planes, kernel_size=kernel, strides=(2, 2),
                padding=padding, output_padding=output_padding,
                use_bias=deconv_with_bias
            )
        )
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.ReLU())

    return tf.keras.Sequential(layers)

def get_pose_net(cfg, is_train, **kwargs):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0], 3),
        include_top=False,
        weights="imagenet"
    )

    x = base_model.output
    x = make_deconv_layer(
        num_layers=cfg.MODEL.EXTRA.NUM_DECONV_LAYERS,
        num_filters=cfg.MODEL.EXTRA.NUM_DECONV_FILTERS,
        num_kernels=cfg.MODEL.EXTRA.NUM_DECONV_KERNELS,
        deconv_with_bias=cfg.MODEL.EXTRA.DECONV_WITH_BIAS
    )(x)
    prediction = tf.keras.layers.Conv2D(
        filters=cfg.MODEL.NUM_JOINTS,
        kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
        strides=(1, 1),
        padding='same',
        use_bias=False
    )(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=prediction)

    return model
