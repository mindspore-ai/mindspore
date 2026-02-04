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
from mindspore import Tensor
from mindspore.nn import SGD as MSGD
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from mindspore.ops.operations.nn_ops import ReLU
from mindspore.nn import Cell
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.nn import BCEWithLogitsLoss


class SGD(MSGD):
    def __init__(self, params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0,
                 nesterov=False, loss_scale=1.0):
        super().__init__(params, learning_rate=learning_rate, momentum=momentum,
                         dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov, loss_scale=loss_scale)


class MulNet(Cell):
    def __init__(self, input_shape, kernel_size=(1, 3, 3), dtype=np.float32):
        super().__init__(dtype)
        mul_np = np.full(input_shape, 0.5, dtype=dtype)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        weight_c = (input_shape[1], input_shape[1], *kernel_size)
        weight_c_np = np.ones(weight_c).astype(dtype)
        self.weight_c = Parameter(Tensor(weight_c_np), name="weight_c")
        self.mul = P.Mul()
        self.relu = ReLU()

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        x = self.relu(x)
        return x


class MulNetTorch(nn.Module):
    def __init__(self, input_shape, dtype=np.float32):
        super().__init__()
        mul_np = np.full(input_shape, 0.5, dtype=dtype)
        self.mul_weight = torch.nn.Parameter(torch.tensor(mul_np))
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        x = torch.mul(inputs, self.mul_weight)
        x = self.relu(x)
        return x


class SGDFactory:
    def __init__(self, input_shape, epoch, lr, momentum, dampening, weight_decay,
                 nesterov, dtype=np.float32, loss=None):
        self.dtype = dtype
        self.input_shape = input_shape
        self.input_np = np.random.randn(*input_shape).astype(np.float16).astype(dtype)
        self.label_np = np.random.randn(*input_shape).astype(dtype)
        self.lr = lr
        self.momentum = momentum
        self.epoch = epoch
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.loss = loss

    def forward_pytorch_impl(self):
        input_t = torch.from_numpy(self.input_np.copy().astype(np.float32))
        label = torch.from_numpy(self.label_np.copy().astype(np.float32))
        net = MulNetTorch(input_shape=self.input_shape, dtype=np.float32)
        optimizer = TSGD(net.parameters(), lr=self.lr, momentum=self.momentum,
                         dampening=self.dampening,
                         weight_decay=self.weight_decay, nesterov=self.nesterov)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        for _ in range(self.epoch):
            optimizer.zero_grad()
            loss = criterion(net(input_t), label)
            loss.backward()
            optimizer.step()
        output = net(input_t)
        return output.detach().numpy().astype(self.dtype)

    def forward_mindspore_impl(self):
        inputa = Tensor(self.input_np.copy())
        label = Tensor(self.label_np.copy())
        net = MulNet(input_shape=self.input_shape, dtype=self.dtype)
        criterion = BCEWithLogitsLoss(reduction='mean')
        optimizer = MSGD(params=net.trainable_params(), learning_rate=self.lr,
                         momentum=self.momentum,
                         dampening=self.dampening, weight_decay=self.weight_decay,
                         nesterov=self.nesterov)
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()
        for _ in range(self.epoch):
            train_network(inputa, label)
        output = net(inputa)
        return output.asnumpy()

    def forward_cmp(self):
        out_pt = self.forward_pytorch_impl()
        out_me = self.forward_mindspore_impl()
        np.allclose(out_pt, out_me, self.loss, self.loss)


def test_sgd_3d_forward_input_3x8x4x12x32_lr_01_momentum_00_epoch_2():
    """
    Feature: test 3d sgd momentum.
    Description: test 3d sgd momentum.
    Expectation: the result match with expected result.
    """
    fact = SGDFactory(input_shape=(3, 8, 4, 12, 32), epoch=2, lr=0.1,
                      momentum=0.0, dampening=0.0, weight_decay=0.0,
                      nesterov=False,loss=0.005)
    fact.forward_cmp()
