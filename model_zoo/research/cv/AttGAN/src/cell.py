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
"""Cell Definition"""

import numpy as np

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import nn, ops
from mindspore.common import initializer as init, set_seed
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)

set_seed(1)
np.random.seed(1)


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'KaimingUniform':
                cell.weight.set_data(init.initializer(init.HeUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, (nn.GroupNorm, nn.BatchNorm2d)):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class GenWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        _, g_loss, _, _, _, = self.network(img_a, att_a, att_a_, att_b, att_b_)
        return g_loss


class DisWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return discriminator loss
    """

    def __init__(self, network):
        super().__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        d_loss, _, _, _, _ = self.network(img_a, att_a, att_a_, att_b, att_b_)
        return d_loss


class TrainOneStepCellGen(nn.Cell):
    """Encapsulation class of AttGAN generator network training."""

    def __init__(self, generator, optimizer, sens=1.0):
        super().__init__()
        self.optimizer = optimizer
        self.generator = generator
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.weights = optimizer.parameters
        self.network = GenWithLossCell(generator)
        self.network.add_flags(defer_inline=True)

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()

        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        weights = self.weights
        _, loss, gf_loss, gc_loss, gr_loss = self.generator(img_a, att_a, att_a_, att_b, att_b_)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_a, att_a, att_a_, att_b, att_b_, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads)), gf_loss, gc_loss, gr_loss


class TrainOneStepCellDis(nn.Cell):
    """Encapsulation class of AttGAN discriminator network training."""

    def __init__(self, discriminator, optimizer, sens=1.0):
        super().__init__()
        self.optimizer = optimizer
        self.discriminator = discriminator
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DisWithLossCell(discriminator)
        self.network.add_flags(defer_inline=True)

        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()

        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, img_a, att_a, att_a_, att_b, att_b_):
        weights = self.weights
        loss, d_real_loss, d_fake_loss, dc_loss, df_gp = self.discriminator(img_a, att_a, att_a_, att_b, att_b_)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_a, att_a, att_a_, att_b, att_b_, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads)), d_real_loss, d_fake_loss, dc_loss, df_gp
