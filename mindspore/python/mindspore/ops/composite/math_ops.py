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
"""math Operations."""
import mindspore.ops as ops
from mindspore.ops import functional as F
from mindspore.ops.function.math_func import cummin
from mindspore.ops._primitive_cache import _get_cache_prim


def matmul(x1, x2, dtype=None):
    """
    Returns the matrix product of two arrays.

    Note:
        Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x1 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.
        x2 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the matrix product of the inputs. This is a scalar only
        when both `x1`, `x2` are 1-d vectors.

    Raises:
        ValueError: If the last dimension of `x1` is not the same size as the
            second-to-last dimension of `x2`, or if a scalar value is passed in.
        ValueError: If the shape of `x1` and `x2` could not broadcast together.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> # case 1 : Reasonable application of broadcast mechanism
        >>> x1 = Tensor(np.arange(2*3*4).reshape(2, 3, 4), mindspore.float32)
        >>> x2 = Tensor(np.arange(4*5).reshape(4, 5), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [[[  70.   76.   82.   88.   94.]
        [ 190.  212.  234.  256.  278.]
        [ 310.  348.  386.  424.  462.]]
        [[ 430.  484.  538.  592.  646.]
        [ 550.  620.  690.  760.  830.]
        [ 670.  756.  842.  928. 1014.]]]
        >>> print(output.shape)
        (2, 3, 5)
        >>> # case 2 : the rank of `x1` is 1
        >>> x1 = Tensor(np.ones([1, 2]), mindspore.float32)
        >>> x2 = Tensor(np.ones([2,]), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [2.]
        >>> print(output.shape)
        (1,)
    """
    res = F.matmul(x1, x2)
    if dtype is not None:
        res = res.astype(dtype)
    return res


def mm(input, mat2):
    r"""
    Returns the matrix product of two arrays.
    If `input` is a :math:`(n \times m)` Tensor, `mat2` is a
    :math:`(m \times p)` Tensor, `out` will be a :math:`(n \times p)` Tensor.

    Note:
        - This function cannot support broadcasting.
          Refer to :func:`mindspore.ops.matmul` instead if you need a broadcastable function.
        - On Ascend, float64 doesn't be supported.

    Args:
        input (Tensor): The first matrix of matrix multiplication.
            The last dimension of `input` must be the same size as the first dimension of `mat2`.
        mat2 (Tensor): The second matrix of matrix multiplication.
            The last dimension of `input` must be the same size as the first dimension of `mat2`.

    Returns:
        Tensor or scalar, the matrix product of the inputs.

    Raises:
        ValueError: If the last dimension of `input` is not the same size as the
            second-to-last dimension of `mat2`.
        ValueError: If `input` or `mat2` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x1 = ms.Tensor(np.random.rand(2, 3), ms.float32)
        >>> x2 = ms.Tensor(np.random.rand(3, 4), ms.float32)
        >>> out = ops.mm(x1, x2)
        >>> print(out.shape)
        (2, 4)
    """
    _matmul = _get_cache_prim(ops.MatMul)()
    out = _matmul(input, mat2)
    return out
