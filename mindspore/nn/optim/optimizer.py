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
import logging

import numpy as np

from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.nn.cell import Cell
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore._checkparam import ParamValidator as validator
from mindspore._checkparam import Rel
from mindspore.common.tensor import Tensor

logger = logging.getLogger('Optimizer')

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

    Raises:
        ValueError: If the learning_rate is a Tensor, but the dims of tensor is greater than 1.
        TypeError: If the learning_rate is not any of the three types: float, Tensor, Iterable.
    """

    def __init__(self, learning_rate, parameters):
        super(Optimizer, self).__init__()
        if isinstance(learning_rate, float):
            validator.check_number_range("learning rate", learning_rate, 0.0, float("inf"), Rel.INC_LEFT)
        elif isinstance(learning_rate, Iterable):
            learning_rate = Tensor(np.array(list(learning_rate)).astype(np.float32))
        elif isinstance(learning_rate, Tensor):
            if learning_rate.dim() > 1:
                raise ValueError("Learning rate should be a 0 or 1 dim `Tensor`,"
                                 f"but got {learning_rate.dim()}.")
        else:
            raise TypeError("Learning rate should be float, Tensor or Iterable.")

        if isinstance(learning_rate, Tensor) and learning_rate.dim() == 1 and learning_rate.size() < 2:
            logger.warning("If want to use the dynamic learning rate, please make sure that "
                           "the number of elements in the list, tuple or tensor passed is greater than 1.")
        self.learning_rate = Parameter(learning_rate, name="learning_rate")
        self.parameters = ParameterTuple(parameters)
        if not self.parameters:
            raise ValueError("optimizer got an empty parameter list.")

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
