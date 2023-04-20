# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Implementation for internal polymorphism `logical not` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner

# logical_not is a metagraph object which will generate function according to input type
# using ".register" decorator
logical_not = base.MultitypeFuncGraph("logical_not", True)


@logical_not.register("Number")
def _logical_not_scala(x):
    """
    Return logical not operation result of x.

    Args:
       x(Number): Number.

    Returns:
       bool, Return logical not operation result of x.
   """
    return F.bool_not(x.__bool__())


@logical_not.register("String")
def _logical_not_string(x):
    """
    Return logical not operation result of x.

    Args:
       x(String): String.

    Returns:
       bool, Return logical not operation result of x.
   """
    return inner.string_not(x)


@logical_not.register("None")
def _logical_not_none(x):
    """
    Return logical not operation result of none.

    Args:
       x(None): None.

    Returns:
       bool, return True.
   """
    return True


@logical_not.register("Tensor")
def _logical_not_tensor(x):
    """
    Return logical not operation result of x.
    Args:
       x(Tensor): Tensor.
    Returns:
       Tensor, Return logical not operation result of x.
   """
    if F.isconstant(x):
        return F.bool_not(x.__bool__())
    return F.logical_not(x.__bool__())


@logical_not.register("Tuple")
def _logical_not_tuple(x):
    """
    Return logical not operation result of a tuple object.

    Args:
       x(Tuple): The input tuple.

    Returns:
       bool, Return logical not operation result of x.
   """
    return x.__len__() == 0


@logical_not.register("List")
def _logical_not_list(x):
    """
    Return logical not operation result of a list object.

    Args:
       x(List): The input tuple.

    Returns:
       bool, Return logical not operation result of x.
   """
    return x.__len__() == 0
