# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""alexnet backbone"""
import numpy as np
import mindspore.nn as nn
import mindspore.numpy  as  n_p
from mindspore.common.initializer import HeNormal
from mindspore import Parameter, Tensor, ops
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .config import config


class SiameseAlexNet(nn.Cell):
    """
    define  alexnet both used in train and eval

    if train = True
        Returns: loss
    else
        if the first image pair used in evel:
        Return: exemplar
        if the other image pair used in evel:
        Return: score map

    """
    def   __init__(self, train=True):
        super(SiameseAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, has_bias=True, stride=2, pad_mode='valid',
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu'),
                               bias_init=0)

        self.conv2 = nn.Conv2d(96, 256, 5, has_bias=True, stride=1, pad_mode='valid',
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu'),
                               bias_init=0)
        self.conv3 = nn.Conv2d(256, 384, 3, has_bias=True, stride=1, pad_mode='valid',
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu'),
                               bias_init=0)

        self.conv4 = nn.Conv2d(384, 384, 3, has_bias=True, stride=1, pad_mode='valid',
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu'),
                               bias_init=0)
        self.conv5 = nn.Conv2d(384, 256, 3, has_bias=True, stride=1, pad_mode='valid',
                               weight_init=HeNormal(mode='fan_out', nonlinearity='relu'),
                               bias_init=0)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(384)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu = nn.ReLU()
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.training = train
        self.conv2d_train = ops.Conv2D(out_channel=8, kernel_size=6, stride=1, pad_mode='valid')
        self.corr_bias = Parameter(n_p.zeros(1))
        self.cast = P.Cast()
        if train:
            gt_train, weight_train = self._create_gt_mask((config.train_response_sz,
                                                           config.train_response_sz))
            self.train_gt = Tensor(gt_train).astype(mstype.float32)
            self.train_weight = Tensor(weight_train).astype(mstype.float32)
            self.loss = nn.BCEWithLogitsLoss(reduction='sum', weight=self.train_weight)
        self.adjust_batchnormal = nn.BatchNorm2d(1)
        self.groups = config.train_batch_size
        self.feature_spilt = P.Split(axis=1, output_num=self.groups)
        self.kernel_split = P.Split(axis=0, output_num=self.groups)
        self.feature_spilt_evel = P.Split(axis=1, output_num=3)
        self.kernel_split_evel = P.Split(axis=0, output_num=3)
        self.Conv2D_1 = ops.Conv2D(out_channel=1, kernel_size=6, stride=1, pad_mode='valid')
        self.op_concat = P.Concat(axis=1)
        self.op_concat_exemplar = P.Concat(axis=0)
        self.seq = nn.SequentialCell([self.conv1,
                                      self.bn1,
                                      self.relu,
                                      self.max_pool2d_1,
                                      self.conv2,
                                      self.bn2,
                                      self.relu,
                                      self.max_pool2d_2,
                                      self.conv3,
                                      self.bn3,
                                      self.relu,
                                      self.conv4,
                                      self.bn4,
                                      self.relu,
                                      self.conv5,
                                      ])
    def construct(self, x, y):
        """network construct"""
        if self.training:
            x = n_p.squeeze(x)
            y = n_p.squeeze(y)
            exemplar = x
            instance = y
            exemplar = self.seq(exemplar)
            instance = self.seq(instance)
            nx, cx, h, w = instance.shape
            instance = instance.view(1, nx * cx, h, w)
            features = self.feature_spilt(instance)
            kernel = self.kernel_split(exemplar)
            outputs = ()
            for i in range(self.groups):
                outputs = outputs + ((self.Conv2D_1(self.cast(features[i], mstype.float32),
                                                    self.cast(kernel[i], mstype.float32))),)
            score_map = self.op_concat(outputs)
            score_map = n_p.transpose(score_map, (1, 0, 2, 3))
            score_map = score_map*1e-3+self.corr_bias
            score = self.loss(score_map, self.train_gt)/8

        else:
            exemplar = x
            instance = y
            if exemplar.size is not None and  instance.size == 1:
                exemplar = self.seq(exemplar)
                return exemplar
            instance = self.seq(instance)
            nx, cx, h, w = instance.shape
            instance = n_p.reshape(instance, [1, nx*cx, h, w])
            features = self.feature_spilt_evel(instance)
            kernel = self.kernel_split_evel(exemplar)
            outputs = ()
            outputs = outputs + (self.Conv2D_1(features[0], kernel[0]),)
            outputs = outputs + (self.Conv2D_1(features[1], kernel[1]),)
            outputs = outputs + (self.Conv2D_1(features[2], kernel[2]),)
            score_map = self.op_concat(outputs)
            score = n_p.transpose(score_map, (1, 0, 2, 3))
        return score

    def _create_gt_mask(self, shape):
        """crete label """
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.zeros((h, w))
        mask[dist <= config.radius / config.total_stride] = 1
        mask = mask[np.newaxis, :, :]
        weights = np.ones_like(mask)
        weights[mask == 1] = 0.5 / np.sum(mask == 1)
        weights[mask == 0] = 0.5 / np.sum(mask == 0)
        mask = np.repeat(mask, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return mask.astype(np.float32), weights.astype(np.float32)
