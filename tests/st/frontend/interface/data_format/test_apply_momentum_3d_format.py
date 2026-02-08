# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test data format."""
import numpy as np
import torch
from torch import nn
from torch.optim import SGD as TSGD
import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Momentum
from mindspore.nn import BCEWithLogitsLoss
from mindspore.nn import WithLossCell
from mindspore.nn import TrainOneStepCell
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.operations.nn_ops import Conv3D
from mindspore.nn.layer.normalization import BatchNorm3d
from mindspore.common import dtype as ms_dtype


class MyMomentum(Momentum):
    def __init__(self, params, learning_rate=0.1, momentum=0.0, weight_decay=0.0, loss_scale=1.0,
                 use_nesterov=False):
        super().__init__(params, learning_rate=learning_rate, momentum=momentum,
                         weight_decay=weight_decay, loss_scale=loss_scale,
                         use_nesterov=use_nesterov)


class MulNet(Cell):
    def __init__(self, input_shape, kernel_size=(1, 3, 3), dtype=np.float32):
        super().__init__(dtype)
        mul_np = np.full(input_shape, 0.5, dtype=dtype)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        weight_c = (input_shape[1], input_shape[1], *kernel_size)
        if dtype != np.float32:
            weight_c_np = np.ones(weight_c).astype(np.float32)
            self.flag = True
        else:
            weight_c_np = np.ones(weight_c).astype(dtype)
            self.flag = False
        self.weight_c = Parameter(Tensor(weight_c_np), name="weight_c")
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.conv = Conv3D(out_channel=input_shape[1], kernel_size=kernel_size, pad_mode='same')
        self.batchnorm = BatchNorm3d(num_features=input_shape[1], eps=1e-8, momentum=0.9)
        self.dtype = dtype

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        if self.flag:
            x = self.cast(x, ms_dtype.float32)
        x = self.conv(x, self.weight_c)
        x = self.batchnorm(x)
        return x.astype(self.dtype)


class MulNetTorch(nn.Module):
    def __init__(self, input_shape, kernel_size=(1, 3, 3), dtype=np.float32):
        super().__init__()
        mul_np = np.full(input_shape, 0.5, dtype=dtype)
        self.mul_weight = torch.nn.Parameter(torch.tensor(mul_np))
        conv3d_weight = (input_shape[1], input_shape[1], *kernel_size)
        conv3d_weight_np = np.ones(conv3d_weight).astype(dtype)
        weight_c = torch.nn.Parameter(torch.tensor(conv3d_weight_np))
        pad_along_d = max(1 * (kernel_size[0] - 1) + 1 - 1, 0)
        pad_along_height = max(1 * (kernel_size[1] - 1) + 1 - 1, 0)
        pad_along_width = max(1 * (kernel_size[2] - 1) + 1 - 1, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        pad_head = pad_along_d // 2
        pad_tail = pad_along_d - pad_head
        padding_torch = [pad_tail, pad_right, pad_bottom]
        self.conv3d = torch.nn.Conv3d(in_channels=input_shape[1],
                                      out_channels=input_shape[1],
                                      kernel_size=kernel_size, padding=padding_torch, bias=False)
        self.conv3d.register_parameter('weight', weight_c)
        self.bn3d = torch.nn.BatchNorm3d(num_features=input_shape[1], eps=1e-8, momentum=0.9)

    def forward(self, inputs):
        x = torch.mul(inputs, self.mul_weight)
        x = self.conv3d(x)
        x = self.bn3d(x)
        return x


class MomentumFactory:
    def __init__(self, input_shape, epoch, learning_rate=0.1, momentum=0.0,
                 weight_decay=0.0, loss_scale=1.0, use_nesterov=False, dtype=np.float32, loss=None):
        self.input_shape = input_shape
        self.input_np = np.random.randn(*input_shape).astype(dtype)
        self.label_np = np.random.randn(*input_shape).astype(dtype)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        self.use_nesterov = use_nesterov
        self.dtype = dtype
        self.loss = loss

    def forward_mindspore_impl(self):
        input_ms = Tensor(self.input_np)
        label = Tensor(self.label_np)
        net = MulNet(input_shape=self.input_shape, dtype=self.dtype)
        criterion = BCEWithLogitsLoss(reduction='mean')
        optimizer = MyMomentum(params=net.trainable_params(), learning_rate=self.learning_rate,
                             weight_decay=self.weight_decay,
                             momentum=self.momentum, loss_scale=self.loss_scale,
                             use_nesterov=self.use_nesterov)

        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()

        for _ in range(self.epoch):
            train_network(input_ms, label)
        output = net(input_ms)
        return output.asnumpy()

    def forward_pytorch_impl(self):
        input_t = torch.from_numpy(self.input_np.copy().astype(np.float32))
        label = torch.from_numpy(self.label_np.copy().astype(np.float32))

        net = MulNetTorch(input_shape=self.input_shape, dtype=np.float32)
        optimizer = TSGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum,
                         dampening=0.0, weight_decay=self.weight_decay, nesterov=self.use_nesterov)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        for _ in range(self.epoch):
            optimizer.zero_grad()
            loss = criterion(net(input_t), label)
            loss.backward()
            optimizer.step()

        output = net(input_t)
        return output.detach().numpy().astype(self.dtype)

    def forward_cmp(self):
        out_me = self.forward_mindspore_impl()
        out_torch = self.forward_pytorch_impl()
        np.allclose(out_torch, out_me, self.loss, self.loss)


def test_momentum_forward_input_3x8x4x12x32_lr_0001_momentum_00():
    """
    Feature: test 3d momentum.
    Description: test 3d momentum.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(enable_auto_mixed_precision=False)
    fact = MomentumFactory(input_shape=(3, 8, 4, 12, 32), epoch=1, learning_rate=0.001,
                           momentum=0.0, weight_decay=0.0, loss_scale=1.0, use_nesterov=False,
                           dtype=np.float32, loss=0.001)
    fact.forward_cmp()
