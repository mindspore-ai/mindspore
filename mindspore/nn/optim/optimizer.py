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
"""optimizer"""
from typing import Iterable

import numpy as np

import mindspore
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.nn.cell import Cell
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore._checkparam import ParamValidator as validator
from mindspore._checkparam import Rel
from mindspore.common.tensor import Tensor
from mindspore import log as logger


__all__ = ['Optimizer']


class Optimizer(Cell):
    """
    Base class for all optimizers.

    This class defines the API to add Ops to train a model.

    Note:
        This class defines the API to add Ops to train a model. Never use
        this class directly, but instead instantiate one of its subclasses.

    Args:
        learning_rate (float): A floating point value for the learning rate. Should be greater than 0.
        parameters (list): A list of parameter, which will be updated. The element in `parameters`
            should be class mindspore.Parameter.
        weight_decay (float): A floating point value for the weight decay. Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Default: 1.0. Should be greater than 0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default: lambda
            x: 'beta' not in x.name and 'gamma' not in x.name.

    Raises:
        ValueError: If the learning_rate is a Tensor, but the dims of tensor is greater than 1.
        TypeError: If the learning_rate is not any of the three types: float, Tensor, Iterable.
    """

    def __init__(self, learning_rate, parameters, weight_decay=0.0, loss_scale=1.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name):
        super(Optimizer, self).__init__()
        if isinstance(learning_rate, float):
            self.dynamic_lr = False
            self.gather = None
            self.assignadd = None
            self.global_step = None
            validator.check_number_range("learning rate", learning_rate, 0.0, float("inf"), Rel.INC_LEFT)
        else:
            self.dynamic_lr = True
            self.gather = P.GatherV2()
            self.assignadd = P.AssignAdd()
            self.global_step = Parameter(initializer(0, [1], mindspore.int32), name='global_step')
            if isinstance(learning_rate, Iterable):
                learning_rate = Tensor(np.array(list(learning_rate)).astype(np.float32))
            elif isinstance(learning_rate, Tensor):
                if learning_rate.dim() > 1:
                    raise ValueError("Learning rate should be a 0 or 1 dim `Tensor`,"
                                     f"but got {learning_rate.dim()}.")
                if learning_rate.dim() == 1 and learning_rate.size() < 2:
                    logger.warning("If want to use the dynamic learning rate, please make sure that the number "
                                   "of elements in the list, tuple or tensor passed is greater than 1.")
            else:
                raise TypeError("Learning rate should be float, Tensor or Iterable.")

        if isinstance(weight_decay, int):
            weight_decay = float(weight_decay)

        if not isinstance(weight_decay, float):
            raise TypeError("weight_decay should be a float number!")

        if isinstance(loss_scale, int):
            loss_scale = float(loss_scale)

        if not isinstance(loss_scale, float):
            raise TypeError("loss_scale should be a float number!")

        if loss_scale <= 0.0:
            raise ValueError("Loss scale should be greater than 0, but got {}".format(loss_scale))
        self.loss_scale = loss_scale

        if weight_decay < 0.0:
            raise ValueError("Weight decay should be equal or greater than 0, but got {}".format(weight_decay))

        self.learning_rate = Parameter(learning_rate, name="learning_rate")
        self.parameters = ParameterTuple(parameters)
        self.reciprocal_scale = 1.0 / loss_scale
        self.weight_decay = weight_decay * loss_scale
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)

        if not self.parameters:
            raise ValueError("optimizer got an empty parameter list.")

    def decay_weight(self, gradients):
        """
        Weight decay.

        An approach to reduce the overfitting of a deep learning neural network model.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape with
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after weight decay.
        """
        if self.weight_decay > 0:
            params = self.parameters
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags, params, gradients)

        return gradients

    def scale_grad(self, gradients):
        """
        Loss scale for mixed precision.

        An approach of mixed precision training to improve the speed and energy efficiency of training deep neural
        network.

        Args:
            gradients (tuple[Tensor]): The gradients of `self.parameters`, and have the same shape with
                `self.parameters`.

        Returns:
            tuple[Tensor], The gradients after loss scale.

        """
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)

        return gradients

    def get_lr(self):
        """
        Get the learning rate of current step.

        Returns:
            float, the learning rate of current step.
        """
        lr = self.learning_rate
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, 0)
            F.control_depend(lr, self.assignadd(self.global_step, 1))

        return lr

    def construct(self, *hyper_params):
        raise NotImplementedError


op_add = P.AddN()

apply_decay = C.MultitypeFuncGraph("apply_decay")


@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((gradient, weight * weight_decay))
    return gradient


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Number", "Tensor")
def tensor_grad_scale(scale, grad):
    """Get grad with scale."""
    if scale == 1.0:
        return grad
    cast_op = P.Cast()
    type_op = P.DType()
    return grad * cast_op(F.scalar_to_array(scale), type_op(grad))
