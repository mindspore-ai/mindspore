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

"""Implementation for internal polymorphism `bitwise or` operations."""
from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner

# bitwise_or is a metagraph object which will generate function according to input type
# using ".register" decorator
bitwise_or = base.MultitypeFuncGraph("bitwise_or", True)

bitwise_or.set_need_raise()


@bitwise_or.register("Number", "Number")
def _bitwise_or_scalar(x, y):
    """Returns x | y where x and y are all scalars."""
    return inner.bit_or(x, y)


@bitwise_or.register("Tensor", "Tensor")
def _bitwise_or_tensor(x, y):
    """Returns x | y where x is Number and y is Tensor."""
    return F.bitwise_or(x, y)


@bitwise_or.register("Tensor", "Number")
def _tensor_bitwise_or_scalar(x, y):
    """Returns x | y where x is Tensor and y is Number."""
    return F.bitwise_or(x, y)


@bitwise_or.register("Number", "Tensor")
def _scalar_bitwise_or_tensor(x, y):
    """Returns x | y where x and y are all tensors."""
    return F.bitwise_or(x, y)
