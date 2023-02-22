# Copyright 2023 Huawei Technologies Co., Ltd
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

"""Defines other operators with functional form."""

from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim


def partial(func, *args):
    """
    Makes a partial function instance. Partial function can be used to derived specialized
    functions from general functions by fixing the value of certain arguments.

    Args:
        func (FunctionType): The incoming function.
        args (Tensor): The arguments of the incoming function.

    Returns:
        FunctionType, partial function bound with arguments.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> def show_input(x, y, z):
        ...     return x, y, z
        >>> partial_show_input = ops.partial(show_input, Tensor(1))
        >>> output1 = partial_show_input(Tensor(2), Tensor(3))
        >>> print(output1)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64,
         value= 3))
        >>> output2 = partial_show_input(Tensor(3), Tensor(4))
        >>> print(output2)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64,
         value= 4))
    """
    partial_ = _get_cache_prim(P.Partial)()
    return partial_(func, *args)


def depend(value, expr):
    """
    depend is used for processing dependency operations.

    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = depend(y, a)
                                --->        b = B(y)

    Args:
        value (Tensor): The real value to return for depend operator.
        expr (Expression): The expression to execute with no outputs.

    Returns:
        Tensor, the value passed by last operator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.softmax = ops.Softmax()
        ...
        ...     def construct(self, x, y):
        ...         mul = x * y
        ...         y = ops.depend(y, mul)
        ...         ret = self.softmax(y)
        ...         return ret
        ...
        >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
        [[0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]]
    """
    depend_ = _get_cache_prim(P.Depend)()
    return depend_(value, expr)


__all__ = [
    'depend',
    'partial'
]
__all__.sort()
