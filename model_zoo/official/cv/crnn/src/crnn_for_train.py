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
"""Automatic differentiation with grad clip."""
import numpy as np
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn.cell import Cell
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
import mindspore.nn as nn
from mindspore.common.tensor import Tensor

compute_norm = C.MultitypeFuncGraph("compute_norm")


@compute_norm.register("Tensor")
def _compute_norm(grad):
    norm = nn.Norm()
    norm = norm(F.cast(grad, mstype.float32))
    ret = F.expand_dims(F.cast(norm, mstype.float32), 0)
    return ret


grad_div = C.MultitypeFuncGraph("grad_div")


@grad_div.register("Tensor", "Tensor")
def _grad_div(val, grad):
    div = P.RealDiv()
    mul = P.Mul()
    scale = div(10.0, val)
    ret = mul(grad, scale)
    return ret


class TrainOneStepCellWithGradClip(Cell):
    """
    Network training package class.

    Wraps the network with an optimizer. The resulting Cell be trained with input data and label.
    Backward graph with grad clip will be created in the construct function to do parameter updating.
    Different parallel modes are available to run the training.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.

    Inputs:
        - data (Tensor) - Tensor of shape :(N, ...).
        - label (Tensor) - Tensor of shape :(N, ...).

    Outputs:
        Tensor, a scalar Tensor with shape :math:`()`.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepCellWithGradClip, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.hyper_map = C.HyperMap()
        self.greater = P.Greater()
        self.select = P.Select()
        self.norm = nn.Norm(keep_dims=True)
        self.dtype = P.DType()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=0)
        self.ten = Tensor(np.array([10.0]).astype(np.float32))
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(data, label, sens)
        norm = self.hyper_map(F.partial(compute_norm), grads)
        norm = self.concat(norm)
        norm = self.norm(norm)
        cond = self.greater(norm, self.cast(self.ten, self.dtype(norm)))
        clip_val = self.select(cond, norm, self.cast(self.ten, self.dtype(norm)))
        grads = self.hyper_map(F.partial(grad_div, clip_val), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
