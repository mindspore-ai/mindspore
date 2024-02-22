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

Math Operators with better performance

"""

from mindspore.ops import auto_generate as P
from mindspore.ops.auto_generate.gen_ops_prim import add_ext_op, sub_ext_op


# define Primitive global variables


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    r"""
    The result is the sum of the input and a batch matrix-matrix product of matrices in batch1 and batch2.
    The formula is defined as follows:

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        input (Tensor): The input Tensor. When batch1 is a :math:`(C, W, T)` Tensor and batch2 is a
            :math:`(C, T, H)` Tensor, input must be broadcastable with :math:`(C, W, H)` Tensor.
        batch1 (Tensor): :math:`batch1` in the above formula. Must be 3-D Tensor, dtype is same as input.
        batch2 (Tensor): :math:`batch2` in the above formula. Must be 3-D Tensor, dtype is same as input.
        beta (Union[float, int], optional): multiplier for input. Default: ``1`` .
        alpha (Union[float, int], optional): multiplier for :math:`batch1 @ batch2`. Default: ``1`` .
            Arguments beta and alpha must be integers when inputs of type not FloatTensor, otherwise they should
            be a real number.

    Returns:
        Tensor, has the same dtype as input, shape will be :math:`(C, W, H)`.

    Raises:
        TypeError: The type of `input`, `batch1`, `batch2` is not Tensor.
        TypeError: The types of `input`, `batch1`, `batch2` are different.
        TypeError: For inputs of type FloatTensor or DoubleTensor, \
                    arguments beta and alpha not be real numbers, otherwise not be integers.
        TypeError: For Baddbmm, attributes alpha and beta are not real numbers
        ValueError: If `batch1` and `batch2` are not 3-D tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
        >>> batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
        >>> batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
        >>> output = ops.baddbmm(input, batch1, batch2)
        >>> print(output)
        [[[5. 5. 5.]
          [5. 5. 5.]
          [5. 5. 5.]]]
    """
    return P.baddbmm(input, batch1, batch2, beta, alpha)


def add(input, other, alpha=1):
    r"""
    Adds scaled other value to input Tensor.

    .. math::

        out_{i} = input_{i} + alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha cannot be bool type at the same time,
          [True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)] are all considered bool type.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.
        - Alpha is a scaling factor applied to `other`.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        alpha (number.Number): A scaling factor applied to `other`.

    Returns:
        Tensor, the shape is the same as the one of the input `input`, `other` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops.extend import sub
        >>> # case 1: x, y and alpha are all Tensor.
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> alpha = Tensor(2.0, mindspore.float32)
        >>> output = add(x, y, alpha)
        >>> print(output)
        [9. 12. 15.]
        >>> # case 2: x is a scalar, y is a Tensor and alpha is a scalar
        >>> x = Tensor(1, mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> alpha = 0.5
        >>> output = add(x, y, alpha)
        >>> print(output)
        [3. 3.5 4.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return add_ext_op(input, other, alpha)


def sub(input, other, alpha=1):
    r"""
    Subtracts scaled other value from input Tensor.

    .. math::

        out_{i} = input_{i} - alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha cannot be bool type at the same time,
          [True, Tensor(True, bool\_), Tensor(np.array([True]), bool\_)] are all considered bool type.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.
        - Alpha is a scaling factor applied to `other`.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        alpha (number.Number): A scaling factor applied to `other`.

    Returns:
        Tensor, the shape is the same as the one of the input `input`, `other` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.ops.extend import sub
        >>> # case 1: x, y and alpha are all Tensor.
        >>> x = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> y = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> alpha = Tensor(2.0, mindspore.float32)
        >>> output = sub(x, y, alpha)
        >>> print(output)
        [2. 1. 0.]
        >>> # case 2: x is a Tensor, y is a scalar and alpha is a scalar
        >>> x = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> y = Tensor(1, mindspore.int32)
        >>> alpha = 0.5
        >>> output = sub(x, y, alpha)
        >>> print(output)
        [3.5 4.5 5.5]
        >>> # the data type of x is float32, the data type of y is int32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return sub_ext_op(input, other, alpha)


__all__ = ['baddbmm', 'add', 'sub']
