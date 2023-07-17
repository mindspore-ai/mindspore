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

"""Implementation for internal polymorphism `bitwise and` operations."""
from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner

# bitwise_and is a metagraph object which will generate function according to input type
# using ".register" decorator
bitwise_and = base.MultitypeFuncGraph("bitwise_and", True)

bitwise_and.set_need_raise()


@bitwise_and.register("Number", "Number")
def _bitwise_and_scalar(x, y):
    """Returns x & y where x and y are all scalars."""
    return inner.bit_and(x, y)


@bitwise_and.register("Tensor", "Tensor")
def _bitwise_and_tensor(x, y):
    """Returns x & y where x is Number and y is Tensor."""
    return F.bitwise_and(x, y)


@bitwise_and.register("Tensor", "Number")
def _tensor_bitwise_and_scalar(x, y):
    """Returns x & y where x is Tensor and y is Number."""
    return F.bitwise_and(x, y)


@bitwise_and.register("Number", "Tensor")
def _scalar_bitwise_and_tensor(x, y):
    """Returns x & y where x and y are all tensors."""
    return F.bitwise_and(x, y)
