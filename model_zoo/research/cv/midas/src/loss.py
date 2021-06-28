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
"""loss."""
import mindspore.nn as nn
import mindspore.ops as ops


class Gradient_loss(nn.Cell):
    """Gradient_loss"""
    def __init__(self):
        super(Gradient_loss, self).__init__()
        self.ms_sum = ops.ReduceSum(keep_dims=False)
        self.abs = ops.Abs()

    def construct(self, prediction, target, mask):
        """Gradient_loss construct"""
        M = self.ms_sum(mask, (1, 2))
        diff = prediction - target
        diff = mask * diff
        grad_x = self.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = mask[:, :, 1:] * mask[:, :, :-1]
        grad_x = mask_x * grad_x

        grad_y = self.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = mask[:, 1:, :] * mask[:, :-1, :]
        grad_y = mask_y * grad_y

        image_loss = self.ms_sum(grad_x, (1, 2)) + self.ms_sum(grad_y, (1, 2))
        divisor = self.ms_sum(M)
        total = self.ms_sum(image_loss) / divisor
        return total


class ScaleAndShiftInvariantLoss(nn.Cell):
    """ScaleAndShiftInvariantLoss"""
    def __init__(self, alpha=0.5, scales=4):
        super(ScaleAndShiftInvariantLoss, self).__init__()
        self.ms_sum = ops.ReduceSum(keep_dims=False)
        self.zeroslike = ops.ZerosLike()
        self.select = ops.Select()
        self.ones = ops.OnesLike()
        self.reshape = ops.Reshape()
        self.alpha = alpha

        self.scales = scales
        self.loss = Gradient_loss()

    def construct(self, prediction, mask, target):
        """construct"""

        a_00 = self.ms_sum(mask * prediction * prediction, (1, 2))
        a_01 = self.ms_sum(mask * prediction, (1, 2))
        a_11 = self.ms_sum(mask, (1, 2))
        b_0 = self.ms_sum(mask * prediction * target, (1, 2))
        b_1 = self.ms_sum(mask * target, (1, 2))
        det = a_00 * a_11 - a_01 * a_01
        mask_det = det != 0
        input_y = self.zeroslike(det)
        input_z = self.ones(det)
        a_11 = self.select(mask_det, a_11, input_y)
        b_0 = self.select(mask_det, b_0, input_y)
        a_01 = self.select(mask_det, a_01, input_y)
        b_1 = self.select(mask_det, b_1, input_y)
        a_00 = self.select(mask_det, a_00, input_y)
        det = self.select(mask_det, det, input_z)
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
        scale = self.reshape(x_0, (-1, 1, 1))
        shift = self.reshape(x_1, (-1, 1, 1))
        prediction_ssi = scale * prediction + shift
        M = self.ms_sum(mask, (1, 2))
        res = prediction_ssi - target
        image_loss = self.ms_sum(mask * res * res, (1, 2))
        divisor = self.ms_sum(M)
        total = self.ms_sum(image_loss) / divisor
        for scale in range(self.scales):
            step = pow(2, scale)

            total += self.loss(prediction_ssi[:, ::step, ::step], target[:, ::step, ::step],
                               mask[:, ::step, ::step])
        return total * self.alpha
