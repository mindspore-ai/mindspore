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
"""Train cell for StarGAN"""
import numpy as np
from mindspore import nn
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.common import initializer as init, set_seed
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
        elif isinstance(cell, nn.GroupNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class GeneratorWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(GeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_real, c_org, c_trg):
        _, G_Loss, _, _, _, = self.network(x_real, c_org, c_trg)
        return G_Loss


class DiscriminatorWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(DiscriminatorWithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, x_real, c_org, c_trg):
        D_Loss, _, _, _, _ = self.network(x_real, c_org, c_trg)
        return D_Loss


class TrainOneStepCellGen(nn.Cell):
    """Encapsulation class of StarGAN generator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph."""
    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOneStepCellGen, self).__init__()
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = GeneratorWithLossCell(G)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, img_real, c_org, c_trg):
        weights = self.weights
        fake_image, loss, G_fake_loss, G_fake_cls_loss, G_rec_loss = self.G(img_real, c_org, c_trg)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_real, c_org, c_trg, sens)
        grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads)), fake_image, loss, G_fake_loss, G_fake_cls_loss, G_rec_loss


class TrainOneStepCellDis(nn.Cell):
    """Encapsulation class of StarGAN Discriminator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph."""
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepCellDis, self).__init__()
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.network = DiscriminatorWithLossCell(D)
        self.network.add_flags(defer_inline=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_real, c_org, c_trg):
        weights = self.weights

        loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss = self.D(img_real, c_org, c_trg)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(img_real, c_org, c_trg, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)

        return F.depend(loss, self.optimizer(grads)), loss, D_real_loss, D_fake_loss, D_real_cls_loss, D_gp_loss
