# Copyright 2020 Huawei Technologies Co., Ltd
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
train step wrap
"""
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Parameter, ParameterTuple


run_opt = C.MultitypeFuncGraph("run_opt")

# pylint: disable=unused-argument
@run_opt.register("Function", "Int", "Number", "Number",
                  "Tensor", "Tensor", "Tensor")
def tensor_run_opt(opt, iterator, learning_rate, momentum,
                   gradient, variable, moment):
    success = True
    new_weight = opt(gradient, moment, variable, learning_rate, momentum)
    success = F.depend(success, P.Assign()(variable, new_weight))
    return success


class OptimizerByMomentum(nn.Cell):
    """
    OptimizerByMomentum definition
    """
    # list of tensor
    def __init__(self, weights):
        super(OptimizerByMomentum, self).__init__()
        self.learning_rate = Parameter(0.1, name="learning_rate")
        self.momentum = Parameter(0.05, name="momentum")
        self.iter = Parameter(0, name="iter")

        self.weights = weights
        self.moments = weights.clone(prefix="moments", init='zeros')

        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum()

    def construct(self, grads):
        success = True
        weights = self.weights
        moments = self.moments
        success = self.hyper_map(
            F.partial(run_opt, self.opt, self.iter,
                      self.learning_rate, self.momentum), grads, weights, moments)
        # self.learning_rate = updata_lr(self.learning_rate, self.momentum)
        return success

class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """
    def __init__(self, network):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = OptimizerByMomentum(self.weights)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation('grad', get_by_list=True)

    def construct(self, x, label):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, label)
        return self.optimizer(grads)

class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """
    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


def train_step_with_loss_warp(network):
    return TrainStepWrap(NetWithLossClass(network))


class TrainStepWrap2(nn.Cell):
    """
    TrainStepWrap2 definition
    """
    def __init__(self, network, sens):
        super(TrainStepWrap2, self).__init__()
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.get_parameters())
        self.optimizer = OptimizerByMomentum(self.weights)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens

    def construct(self, x):
        weights = self.weights
        grads = self.grad(self.network, weights)(x, self.sens)
        return self.optimizer(grads)

def train_step_with_sens(network, sens):
    return TrainStepWrap2(network, sens)

class TrainStepWrapWithoutOpt(nn.Cell):
    """
    TrainStepWrapWithoutOpt definition
    """
    def __init__(self, network):
        super(TrainStepWrapWithoutOpt, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.grad = C.GradOperation('grad', get_by_list=True)

    def construct(self, x, label):
        grads = self.grad(self.network, self.weights)(x, label)
        return grads

def train_step_without_opt(network):
    return TrainStepWrapWithoutOpt(NetWithLossClass(network))
