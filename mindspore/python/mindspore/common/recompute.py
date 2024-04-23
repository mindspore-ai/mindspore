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

from mindspore import log as logger
from mindspore.nn.cell import Cell, _RecomputeCell
from mindspore import context


def _check_validation(block):
    if not isinstance(block, Cell):
        raise TypeError("Recompute function now only support block which inherited from Cell!")
    if context.get_context("mode") != context.PYNATIVE_MODE:
        raise AssertionError("Recompute function now only support pynative mode, you can use \
                             Cell.recompute() in graph mode.")
    if block.construct.__code__.co_name == "staging_specialize":
        logger.warning('Block\'s construct method decorated by @jit that recompute \
                        function will not come into effect.')
    return True


def recompute(block, *args, **kwargs):
    r"""
    This function is used to reduce memory, when run block, rather than
    storing the intermediate activation computed in forward pass, we will recompute it in backward pass.

    Note:
     - Recompute function only support block which inherited from Cell object.
     - This function interface now only support pynative mode. you can use Cell.recompute interface
       in graph mode.
     - When use recompute function, block object should not decorated by @jit.

    Args:
        block (Cell): Block to be recompute.
        args(tuple): Inputs for block object to run forward pass.
        kwargs(dict): Optional input for recompute function.

    Returns: Same as return type of block.

    Raises:
        TypeError: If `block` is not Cell object.
        AssertionError: If execute mode is not PYNATIVE_MODE.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor, recompute
        >>> class MyCell(nn.Cell):
        ...     def __init__(self):
        ...         super(MyCell, self).__init__(auto_prefix=False)
        ...         self.conv = nn.Conv2d(2, 2, 2, has_bias=False, weight_init='ones')
        ...         self.relu = ops.ReLU()
        ...
        ...     def construct(self, x):
        ...         y = recompute(self.conv, x)
        ...         return self.relu(y)
        >>> inputs = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
        >>> my_net = MyCell()
        >>> grad = ops.grad(my_net)(inputs)
        >>> print(grad)
        [[[[2. 4.]
           [4. 8.]]
          [[2. 4.]
           [4. 8.]]]
         [[[2. 4.]
           [4. 8.]]
          [[2. 4.]
           [4. 8.]]]]
    """

    _check_validation(block)
    return _RecomputeCell(block)(*args, **kwargs)


__all__ = [
    'recompute',
]
__all__.sort()
