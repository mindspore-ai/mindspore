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
"""
This test is used to monitor inversion attack method of MindArmour.
"""
import numpy as np

import mindspore.context as context
from mindspore.nn import Cell, MSELoss
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore import Tensor
from tests.mark_utils import arg_mark


class GradWrapWithLoss(Cell):
    def __init__(self, network):
        super(GradWrapWithLoss, self).__init__()
        self._grad_all = GradOperation(get_all=True, sens_param=False)
        self._network = network

    def construct(self, inputs, labels):
        gout = self._grad_all(self._network)(inputs, labels)
        return gout[0]


class AddNet(Cell):
    def __init__(self):
        super(AddNet, self).__init__()
        self._add = P.Add()

    def construct(self, inputs):
        out = self._add(inputs, inputs)
        return out


class InversionLoss(Cell):
    def __init__(self, network, weights):
        super(InversionLoss, self).__init__()
        self._network = network
        self._mse_loss = MSELoss()
        self._weights = weights
        self._get_shape = P.Shape()
        self._zeros = P.ZerosLike()
        self._device_target = context.get_context("device_target")

    def construct(self, input_data, target_features):
        output = self._network(input_data)
        loss_1 = self._mse_loss(output, target_features) / self._mse_loss(target_features, self._zeros(target_features))

        data_shape = self._get_shape(input_data)
        if self._device_target == 'CPU':
            split_op_1 = P.Split(2, data_shape[2])
            split_op_2 = P.Split(3, data_shape[3])
            data_split_1 = split_op_1(input_data)
            data_split_2 = split_op_2(input_data)
            loss_2 = 0
            for i in range(1, data_shape[2]):
                loss_2 += self._mse_loss(data_split_1[i], data_split_1[i - 1])
            for j in range(1, data_shape[3]):
                loss_2 += self._mse_loss(data_split_2[j], data_split_2[j - 1])
        else:
            data_copy_1 = self._zeros(input_data)
            data_copy_2 = self._zeros(input_data)
            data_copy_1[:, :, :(data_shape[2] - 1), :] = input_data[:, :, 1:, :]
            data_copy_2[:, :, :, :(data_shape[2] - 1)] = input_data[:, :, :, 1:]
            loss_2 = self._mse_loss(input_data, data_copy_1) + self._mse_loss(input_data, data_copy_2)
        loss_3 = self._mse_loss(input_data, self._zeros(input_data))
        loss = loss_1*self._weights[0] + loss_2*self._weights[1] + loss_3*self._weights[2]
        return loss


class ImageInversionAttack:
    def __init__(self, network, input_shape, loss_weights=(1, 0.2, 5)):
        self._network = network
        self._loss = InversionLoss(self._network, loss_weights)
        self._input_shape = input_shape

    def generate(self, target_features):
        target_features = target_features
        img_num = target_features.shape[0]
        test_input = np.random.random((img_num,) + self._input_shape).astype(np.float32)
        loss_net = self._loss
        loss_grad = GradWrapWithLoss(loss_net)
        x_grad = loss_grad(Tensor(test_input), Tensor(target_features)).asnumpy()
        return x_grad


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_loss_grad_graph():
    context.set_context(mode=context.GRAPH_MODE)
    net = AddNet()
    target_features = np.random.random((1, 32, 32)).astype(np.float32)
    inversion_attack = ImageInversionAttack(net, input_shape=(1, 32, 32))
    grads = inversion_attack.generate(target_features)
    assert np.any(grads != 0), 'grad result can not be all zeros'
