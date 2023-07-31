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
"""grad accumulation"""
from __future__ import absolute_import
from __future__ import division

from mindspore.nn.cell import Cell
from mindspore.common import Parameter, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P


__all__ = ["GradientAccumulation", "gradient_accumulation_op", "gradient_clear_op"]


gradient_accumulation_op = C.MultitypeFuncGraph("gradient_accumulation_op")


@gradient_accumulation_op.register("Int64", "Tensor", "Tensor")
def cumulative_grad_process(accumulation_step, cumulative_grad, grad):
    """Apply gradient accumulation to cumulative grad."""
    P.AssignAdd()(cumulative_grad, grad / accumulation_step)
    return cumulative_grad


gradient_clear_op = C.MultitypeFuncGraph("gradient_clear_op")


@gradient_clear_op.register("Tensor")
def clear_grad(cumulative_grad):
    """Clear grad."""
    zero_grad = P.ZerosLike()(cumulative_grad)
    return F.assign(cumulative_grad, zero_grad)


class GradientAccumulation(Cell):
    """
    After accumulating the gradients of multiple steps, call to optimize its update.

    Args:
       max_accumulation_step (int): Steps to accumulate gradients.
       optimizer (Cell): Optimizer used.
    """
    def __init__(self, max_accumulation_step, optimizer):
        super(GradientAccumulation, self).__init__()
        self._max_accumulation_step = max_accumulation_step
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.hyper_map = C.HyperMap()
        self._grad_accumulation = self.weights.clone(prefix="grad_accumulation", init='zeros')
        self._accumulation_step = Parameter(Tensor(0, dtype=mstype.int32), name="accumulation_step")

    def construct(self, loss, grads):
        loss = F.depend(loss, self.hyper_map(F.partial(gradient_accumulation_op, self._max_accumulation_step),
                                             self._grad_accumulation, grads))
        self._accumulation_step += 1

        if self._accumulation_step >= self._max_accumulation_step:
            loss = F.depend(loss, self.optimizer(self._grad_accumulation))
            F.assign(self._accumulation_step, 0)

        if self._accumulation_step == 0:
            loss = F.depend(loss, self.hyper_map(F.partial(gradient_clear_op), self._grad_accumulation))

        return loss
