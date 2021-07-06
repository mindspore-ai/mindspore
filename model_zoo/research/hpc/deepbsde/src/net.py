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
"""Define the network structure of DeepBSDE"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops as P
from mindspore import Tensor, Parameter


class DeepBSDE(nn.Cell):
    """
    The network structure of DeepBSDE.

    Args:
        cfg: configure settings.
        bsde(Cell): equation function
    """
    def __init__(self, cfg, bsde):
        super(DeepBSDE, self).__init__()
        self.bsde = bsde
        self.delta_t = bsde.delta_t
        self.num_time_interval = bsde.num_time_interval
        self.dim = bsde.dim
        self.time_stamp = Tensor(np.arange(0, cfg.num_time_interval) * bsde.delta_t)
        self.y_init = Parameter(np.random.uniform(low=cfg.y_init_range[0],
                                                  high=cfg.y_init_range[1],
                                                  size=[1]).astype(np.float32))
        self.z_init = Parameter(np.random.uniform(low=-0.1, high=0.1, size=[1, cfg.dim]).astype(np.float32))

        self.subnet = nn.CellList([FeedForwardSubNet(cfg.dim, cfg.num_hiddens)
                                   for _ in range(bsde.num_time_interval-1)])
        self.generator = bsde.generator
        self.matmul = P.MatMul()
        self.sum = P.ReduceSum(keep_dims=True)

    def construct(self, dw, x):
        """repeat FeedForwardSubNet (num_time_interval - 1) times."""
        all_one_vec = P.Ones()((P.shape(dw)[0], 1), mstype.float32)
        y = all_one_vec * self.y_init
        z = self.matmul(all_one_vec, self.z_init)

        for t in range(0, self.num_time_interval - 1):
            y = y - self.delta_t * (self.generator(self.time_stamp[t], x[:, :, t], y, z)) + self.sum(z * dw[:, :, t], 1)
            z = self.subnet[t](x[:, :, t + 1]) / self.dim
        # terminal time
        y = y - self.delta_t * self.generator(self.time_stamp[-1], x[:, :, -2], y, z) + self.sum(z * dw[:, :, -1], 1)
        return y


class FeedForwardSubNet(nn.Cell):
    """
    Subnet to fit the spatial gradients at time t=tn

    Args:
        dim (int): dimension of the final output
        train (bool): True for train
        num_hidden list(int): number of hidden layers
    """
    def __init__(self, dim, num_hiddens):
        super(FeedForwardSubNet, self).__init__()
        self.dim = dim
        self.num_hiddens = num_hiddens
        bn_layers = [nn.BatchNorm1d(c, momentum=0.99, eps=1e-6, beta_init='normal', gamma_init='uniform')
                     for c in [dim] + num_hiddens + [dim]]
        self.bns = nn.CellList(bn_layers)
        dense_layers = [nn.Dense(dim, num_hiddens[0], has_bias=False, activation=None)]
        dense_layers = dense_layers + [nn.Dense(num_hiddens[i], num_hiddens[i + 1], has_bias=False, activation=None)
                                       for i in range(len(num_hiddens) - 1)]
        # final output should be gradient of size dim
        dense_layers.append(nn.Dense(num_hiddens[-1], dim, activation=None))
        self.denses = nn.CellList(dense_layers)
        self.relu = nn.ReLU()

    def construct(self, x):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        x = self.bns[0](x)
        hiddens_length = len(self.num_hiddens)
        for i in range(hiddens_length):
            x = self.denses[i](x)
            x = self.bns[i+1](x)
            x = self.relu(x)
        x = self.denses[hiddens_length](x)
        x = self.bns[hiddens_length + 1](x)
        return x


class WithLossCell(nn.Cell):
    """Loss function for DeepBSDE"""
    def __init__(self, net):
        super(WithLossCell, self).__init__()
        self.net = net
        self.terminal_condition = net.bsde.terminal_condition
        self.total_time = net.bsde.total_time
        self.sum = P.ReduceSum()
        self.delta_clip = 50.0
        self.selete = P.Select()

    def construct(self, dw, x):
        y_terminal = self.net(dw, x)
        delta = y_terminal - self.terminal_condition(self.total_time, x[:, :, -1])
        # use linear approximation outside the clipped range
        abs_delta = P.Abs()(delta)
        loss = self.sum(self.selete(abs_delta < self.delta_clip,
                                    P.Square()(delta),
                                    2 * self.delta_clip * abs_delta - self.delta_clip * self.delta_clip))
        return loss
