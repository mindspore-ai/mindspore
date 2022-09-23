# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops.composite.clip_ops import get_square_sum


class LeNet(nn.Cell):
    """
    Implements lenet.
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 1
        weight1 = Tensor(np.ones([6, 3, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float16) * 0.01)
        self.conv1 = nn.Conv2d(3, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid', stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")

        self.reshape = P.Reshape()
        self.reshape1 = P.Reshape()

        self.fc1 = nn.Dense(400, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = P.Cast()(output, mstype.float16)
        output = self.conv2(output)
        output = P.Cast()(output, mstype.float32)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


_adam_opt = C.MultitypeFuncGraph("adam_opt")


@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Bool", "Bool")
def _fused_update_with_global_norm(opt, global_norm, beta1, beta2, eps, lr, weight_decay,
                                   param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by FusedAdamWeightDecay.
    """
    success = True
    if optim_filter:
        if decay_flags:
            next_param = opt(param, m, v, lr, beta1, beta2, eps, weight_decay, gradient, global_norm)
        else:
            next_param = opt(param, m, v, lr, beta1, beta2, eps, 0.0, gradient, global_norm)
        return F.depend(success, next_param)
    return success


def clone_state(parameter_tuple, prefix, init):
    new = []
    for old_param in parameter_tuple:
        new_state = Parameter(initializer(init, shape=old_param.shape, dtype=mstype.float32))
        new_state.param_info = old_param.param_info.clone()
        new_state.is_init = False
        new_state.name = prefix + '.' + new_state.name
        new.append(new_state)
    return ParameterTuple(new)


apply_global_norm = C.MultitypeFuncGraph("apply_global_norm")


@apply_global_norm.register("Tensor", "Tensor", "Tensor")
def _apply_global_norm(clip_norm, global_norm, grad):
    return grad * clip_norm / global_norm


class GlobalNorm(nn.Cell):
    """
    Calculate the global norm value of given tensors
    """

    def __init__(self):
        super(GlobalNorm, self).__init__()
        self.norm = nn.Norm()
        self.hyper_map = C.HyperMap()
        self.sqrt = P.Sqrt()

    def construct(self, grads):
        """Calculate global norm construct"""
        square_sum = self.hyper_map(get_square_sum, grads)
        global_norms = self.sqrt(F.addn(square_sum))
        return global_norms


class FusedAdamWeightDecayWithGlobalNorm(Optimizer):
    """
    Implements the gradient clipping by global norm for a AdamWeightDecay optimizer.
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(FusedAdamWeightDecayWithGlobalNorm, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = clone_state(self._parameters, prefix="adam_m", init='zeros')
        self.moments2 = clone_state(self._parameters, prefix="adam_v", init='zeros')
        self.norm = GlobalNorm()
        self.opt = P.FusedCastAdamWeightDecay()
        self.opt.add_prim_attr("primitive_target", "CPU")

    def construct(self, gradients):
        """construct with gradients"""
        global_norm = self.norm(gradients)
        lr = self.get_lr()
        optim_result = self.map_reverse(F.partial(_adam_opt, self.opt, global_norm,
                                                  self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                        self._parameters, self.moments1, self.moments2,
                                        gradients, self.decay_flags, self.optim_filter)
        return optim_result


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fused_cast_adam_weight_decay():
    '''
    Feature: FusedCastAdamWeightDecay
    Description: Test FusedCastAdamWeightDecay
    Expectation: Run lenet success
    '''
    context.set_context(mode=context.GRAPH_MODE)
    data = Tensor(np.ones([32, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    net.batch_size = 32
    learning_rate = 0.01
    optimizer = FusedAdamWeightDecayWithGlobalNorm(filter(lambda x: x.requires_grad, net.get_parameters()),
                                                   learning_rate)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    loss = []
    for _ in range(10):
        res = train_network(data, label)
        loss.append(res.asnumpy())
    assert np.all(loss[-1] < 0.1)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fused_cast_adam_weight_decay_with_memory_optimize():
    '''
    Feature: Integration of dynamic and static memory in the heterogeneous scene
    Description: Test FusedCastAdamWeightDecay
    Expectation: Run lenet success
    '''
    context.set_context(mode=context.GRAPH_MODE, memory_optimize_level="O1")
    data = Tensor(np.ones([32, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([32]).astype(np.int32))
    net = LeNet()
    net.batch_size = 32
    learning_rate = 0.01
    optimizer = FusedAdamWeightDecayWithGlobalNorm(filter(lambda x: x.requires_grad, net.get_parameters()),
                                                   learning_rate)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()
    loss = []
    for _ in range(10):
        res = train_network(data, label)
        loss.append(res.asnumpy())
    assert np.all(loss[-1] < 0.1)
