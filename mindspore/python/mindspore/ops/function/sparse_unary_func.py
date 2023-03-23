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

"""sparse unary function api"""

from mindspore.common import CSRTensor, COOTensor
from mindspore.ops.composite.multitype_ops._constexpr_utils import raise_type_error
from mindspore.ops.function import math_func, nn_func


def csr_cos(x: CSRTensor) -> CSRTensor:
    """
    Computes cosine of input element-wise.

    .. math::
        out_i = cos(x_i)

    .. warning::
        If use Float64, there may be a problem of missing precision.

    Args:
        x (CSRTensor): Input CSRTensor.

    Returns:
        CSRTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64,
    complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_cos(x)
        >>> print(output.values)
        [ 0.5403023  -0.41614684]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_cos')
    return CSRTensor(x.indptr, x.indices, math_func.cos(x.values), x.shape)


def coo_cos(x: COOTensor) -> COOTensor:
    """
    Computes cosine of input element-wise.

    .. math::
        out_i = cos(x_i)

    .. warning::
        If use Float64, there may be a problem of missing precision.

    Args:
        x (COOTensor): Input COOTensor.

    Returns:
        COOTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64,
            complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_cos(x)
        >>> print(output.values)
        [ 0.5403023  -0.41614684]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_cos')
    return COOTensor(x.indices, math_func.cos(x.values), x.shape)


def csr_tan(x: CSRTensor) -> CSRTensor:
    """
    Computes tangent of `x` element-wise.

    .. math::

        out_i = tan(x_i)

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_tan(x)
        >>> print(output.values)
        [-1.5574077 -2.1850398]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_tan')
    return CSRTensor(x.indptr, x.indices, math_func.tan(x.values), x.shape)


def coo_tan(x: COOTensor) -> COOTensor:
    """
    Computes tangent of `x` element-wise.

    .. math::

        out_i = tan(x_i)

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_tan(x)
        >>> print(output.values)
        [-1.5574077 -2.1850398]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_tan')
    return COOTensor(x.indices, math_func.tan(x.values), x.shape)


def csr_exp(x: CSRTensor) -> CSRTensor:
    """
    Returns csr_exponential of a CSRTensor element-wise.

    .. math::

        out_i = e^{x_i}

    Args:
        x (CSRTensor): The input CSRTensor, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_exp(x)
        >>> print(output.values)
        [0.36787948 7.3890557 ]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_exp')
    return CSRTensor(x.indptr, x.indices, math_func.exp(x.values), x.shape)


def coo_exp(x: COOTensor) -> COOTensor:
    """
    Returns coo_exponential of a COOTensor element-wise.

    .. math::

        out_i = e^{x_i}

    Args:
        x (COOTensor): The input COOTensor, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_exp(x)
        >>> print(output.values)
        [0.36787948 7.3890557 ]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_exp')
    return COOTensor(x.indices, math_func.exp(x.values), x.shape)


def csr_inv(x: CSRTensor) -> CSRTensor:
    r"""
    Computes Reciprocal of input CSRTensor element-wise.

    .. math::
        out_i = \frac{1}{x_{i} }

    Args:
        x (CSRTensor): Input CSRTensor. Must be one of the following types: float16, float32 or int32.

    Returns:
        CSRTensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not one of float16, float32, int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_inv(x)
        >>> print(output.values)
        [-1.   0.5]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_inv')
    return CSRTensor(x.indptr, x.indices, math_func.inv(x.values), x.shape)


def coo_inv(x: COOTensor) -> COOTensor:
    r"""
    Computes Reciprocal of input COOTensor element-wise.

    .. math::
        out_i = \frac{1}{x_{i} }

    Args:
        x (COOTensor): Input COOTensor. Must be one of the following types: float16, float32 or int32.

    Returns:
        COOTensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not one of float16, float32, int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_inv(x)
        >>> print(output.values)
        [-1.   0.5]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_inv')
    return COOTensor(x.indices, math_func.inv(x.values), x.shape)


def csr_relu(x: CSRTensor) -> CSRTensor:
    """
    Computes ReLU (Rectified Linear Unit activation function) of input csr_tensors element-wise.

    It returns max(x, 0) element-wise. Specially, the neurons with the negative output
    will be suppressed and the active neurons will stay the same.

    .. math::

        ReLU(x) = (x)^+ = max(0, x)

    Note:
        In general, this operator is more commonly used. The difference from `ReLuV2` is that the `ReLuV2` will
        output one more Mask.

    Args:
        x (CSRTensor): Input CSRTensor.

    Returns:
        CSRTensor of shape :math:`(N, *)`, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is not a number.
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_relu(x)
        >>> print(output.values)
        [0. 2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_relu')
    return CSRTensor(x.indptr, x.indices, nn_func.relu(x.values), x.shape)


def coo_relu(x: COOTensor) -> COOTensor:
    """
    Computes ReLU (Rectified Linear Unit activation function) of input coo_tensors element-wise.

    It returns max(x, 0) element-wise. Specially, the neurons with the negative output
    will be suppressed and the active neurons will stay the same.

    .. math::

        ReLU(x) = (x)^+ = max(0, x)

    Note:
        In general, this operator is more commonly used. The difference from `ReLuV2` is that the `ReLuV2` will
        output one more Mask.

    Args:
        x (COOTensor): Input COOTensor.

    Returns:
        COOTensor of shape :math:`(N, *)`, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is not a number.
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_relu(x)
        >>> print(output.values)
        [0. 2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_relu')
    return COOTensor(x.indices, nn_func.relu(x.values), x.shape)


def csr_expm1(x: CSRTensor) -> CSRTensor:
    """
    Returns exponential then minus 1 of a CSRTensor element-wise.

    .. math::

        out_i = e^{x_i} - 1

    Args:
        x (CSRTensor): The input CSRTensor with a dtype of float16 or float32, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_expm1(x)
        >>> print(output.values)
        [-0.63212055  6.389056  ]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_expm1')
    return CSRTensor(x.indptr, x.indices, math_func.expm1(x.values), x.shape)


def coo_expm1(x: COOTensor) -> COOTensor:
    """
    Returns exponential then minus 1 of a COOTensor element-wise.

    .. math::

        out_i = e^{x_i} - 1

    Args:
        x (COOTensor): The input COOTensor with a dtype of float16 or float32, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_expm1(x)
        >>> print(output.values)
        [-0.63212055  6.389056  ]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_expm1')
    return COOTensor(x.indices, math_func.expm1(x.values), x.shape)


def csr_isfinite(x: CSRTensor) -> CSRTensor:
    r"""
    Determines which elements are finite for each position.

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True\  \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_isfinite(x)
        >>> print(output.values)
        [ True  True]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_isfinite')
    return CSRTensor(x.indptr, x.indices, math_func.isfinite(x.values), x.shape)


def coo_isfinite(x: COOTensor) -> COOTensor:
    r"""
    Determines which elements are finite for each position.

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True\  \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_isfinite(x)
        >>> print(output.values)
        [ True  True]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_isfinite')
    return COOTensor(x.indices, math_func.isfinite(x.values), x.shape)


def csr_asin(x: CSRTensor) -> CSRTensor:
    """
    Computes arcsine of input csr_tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Args:
        x (CSRTensor): Input CSRTensor.

    Returns:
        CSRTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16, float32, float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_asin(x)
        >>> print(output.values)
        [-1.5707964        nan]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_asin')
    return CSRTensor(x.indptr, x.indices, math_func.asin(x.values), x.shape)


def coo_asin(x: COOTensor) -> COOTensor:
    """
    Computes arcsine of input coo_tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Args:
        x (COOTensor): Input COOTensor.

    Returns:
        COOTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16, float32, float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_asin(x)
        >>> print(output.values)
        [-1.5707964        nan]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_asin')
    return COOTensor(x.indices, math_func.asin(x.values), x.shape)


def csr_sqrt(x: CSRTensor) -> CSRTensor:
    r"""
    Returns sqrt of a CSRTensor element-wise.

    .. math::

        out_{i} = \sqrt{x_{i}}

    Args:
        x (CSRTensor): The input CSRTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_sqrt(x)
        >>> print(output.values)
        [      nan 1.4142135]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_sqrt')
    return CSRTensor(x.indptr, x.indices, math_func.sqrt(x.values), x.shape)


def coo_sqrt(x: COOTensor) -> COOTensor:
    r"""
    Returns sqrt of a COOTensor element-wise.

    .. math::

        out_{i} = \sqrt{x_{i}}

    Args:
        x (COOTensor): The input COOTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_sqrt(x)
        >>> print(output.values)
        [      nan 1.4142135]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_sqrt')
    return COOTensor(x.indices, math_func.sqrt(x.values), x.shape)


def csr_log(x: CSRTensor) -> CSRTensor:
    """
    Returns the natural logarithm of a CSRTensor element-wise.

    .. math::
        y_i = log_e(x_i)

    .. warning::
        If the input value of operator Log is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affacted.

    Args:
        x (CSRTensor): The value must be greater than 0.

    Returns:
        CSRTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64 on GPU and CPU.
        TypeError: If dtype of `x` is not float16 or float32 on Ascend.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_log(x)
        >>> print(output.values)
        [       nan 0.69314575]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_log')
    return CSRTensor(x.indptr, x.indices, math_func.log(x.values), x.shape)


def coo_log(x: COOTensor) -> COOTensor:
    """
    Returns the natural logarithm of a COOTensor element-wise.

    .. math::
        y_i = log_e(x_i)

    .. warning::
        If the input value of operator Log is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affacted.

    Args:
        x (COOTensor): The value must be greater than 0.

    Returns:
        COOTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64 on GPU and CPU.
        TypeError: If dtype of `x` is not float16 or float32 on Ascend.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_log(x)
        >>> print(output.values)
        [       nan 0.69314575]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_log')
    return COOTensor(x.indices, math_func.log(x.values), x.shape)


def csr_isnan(x: CSRTensor) -> CSRTensor:
    r"""
    Determines which elements are NaN for each position.

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Nan} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Nan}
        \end{cases}

    where :math:`Nan` means not a number.

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_isnan(x)
        >>> print(output.values)
        [False False]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_isnan')
    return CSRTensor(x.indptr, x.indices, math_func.isnan(x.values), x.shape)


def coo_isnan(x: COOTensor) -> COOTensor:
    r"""
    Determines which elements are NaN for each position.

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Nan} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Nan}
        \end{cases}

    where :math:`Nan` means not a number.

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_isnan(x)
        >>> print(output.values)
        [False False]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_isnan')
    return COOTensor(x.indices, math_func.isnan(x.values), x.shape)


def csr_acos(x: CSRTensor) -> CSRTensor:
    """
    Computes arccosine of input csr_tensors element-wise.

    .. math::

        out_i = cos^{-1}(x_i)

    Args:
        x (CSRTensor): Input CSRTensor.

    Returns:
        CSRTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64,
    complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_acos(x)
        >>> print(output.values)
        [3.1415927       nan]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_acos')
    return CSRTensor(x.indptr, x.indices, math_func.acos(x.values), x.shape)


def coo_acos(x: COOTensor) -> COOTensor:
    """
    Computes arccosine of input coo_tensors element-wise.

    .. math::

        out_i = cos^{-1}(x_i)

    Args:
        x (COOTensor): Input COOTensor.

    Returns:
        COOTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_acos(x)
        >>> print(output.values)
        [3.1415927       nan]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_acos')
    return COOTensor(x.indices, math_func.acos(x.values), x.shape)


def csr_floor(x: CSRTensor) -> CSRTensor:
    r"""
    Rounds a CSRTensor down to the closest integer element-wise.

    .. math::

        out_i = \lfloor x_i \rfloor

    Args:
        x (CSRTensor): The input CSRTensor, its rank must be in [0, 7] inclusive
            and data type must be float16, float32 or float64.

    Returns:
        CSRTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_floor(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_floor')
    return CSRTensor(x.indptr, x.indices, math_func.floor(x.values), x.shape)


def coo_floor(x: COOTensor) -> COOTensor:
    r"""
    Rounds a COOTensor down to the closest integer element-wise.

    .. math::

        out_i = \lfloor x_i \rfloor

    Args:
        x (COOTensor): The input COOTensor, its rank must be in [0, 7] inclusive
            and data type must be float16, float32 or float64.

    Returns:
        COOTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_floor(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_floor')
    return COOTensor(x.indices, math_func.floor(x.values), x.shape)


def csr_atan(x: CSRTensor) -> CSRTensor:
    """
    Computes the trigonometric inverse tangent of the input element-wise.

    .. math::

        out_i = tan^{-1}(x_i)

    Args:
        x (CSRTensor): The data type should be one of the following types: float16, float32.

    Returns:
        A CSRTensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_atan(x)
        >>> print(output.values)
        [-0.7853982  1.1071488]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_atan')
    return CSRTensor(x.indptr, x.indices, math_func.atan(x.values), x.shape)


def coo_atan(x: COOTensor) -> COOTensor:
    """
    Computes the trigonometric inverse tangent of the input element-wise.

    .. math::

        out_i = tan^{-1}(x_i)

    Args:
        x (COOTensor): The data type should be one of the following types: float16, float32.

    Returns:
        A COOTensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_atan(x)
        >>> print(output.values)
        [-0.7853982  1.1071488]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_atan')
    return COOTensor(x.indices, math_func.atan(x.values), x.shape)


def csr_square(x: CSRTensor) -> CSRTensor:
    """
    Returns square of a CSRTensor element-wise.

    .. math::

        out_{i} = (x_{i})^2

    Args:
        x (CSRTensor): The input CSRTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_square(x)
        >>> print(output.values)
        [1. 4.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_square')
    return CSRTensor(x.indptr, x.indices, math_func.square(x.values), x.shape)


def coo_square(x: COOTensor) -> COOTensor:
    """
    Returns square of a COOTensor element-wise.

    .. math::

        out_{i} = (x_{i})^2

    Args:
        x (COOTensor): The input COOTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_square(x)
        >>> print(output.values)
        [1. 4.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_square')
    return COOTensor(x.indices, math_func.square(x.values), x.shape)


def csr_relu6(x: CSRTensor) -> CSRTensor:
    r"""
    Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input csr_tensors element-wise.

    .. math::

        \text{ReLU6}(x) = \min(\max(0,x), 6)

    It returns :math:`\min(\max(0,x), 6)` element-wise.

    Args:
        x (CSRTensor): Input CSRTensor, with float16 or float32 data type.

    Returns:
        CSRTensor, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_relu6(x)
        >>> print(output.values)
        [0. 2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_relu6')
    return CSRTensor(x.indptr, x.indices, nn_func.relu6(x.values), x.shape)


def coo_relu6(x: COOTensor) -> COOTensor:
    r"""
    Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input coo_tensors element-wise.

    .. math::

        \text{ReLU6}(x) = \min(\max(0,x), 6)

    It returns :math:`\min(\max(0,x), 6)` element-wise.

    Args:
        x (COOTensor): Input COOTensor, with float16 or float32 data type.

    Returns:
        COOTensor, with the same dtype and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_relu6(x)
        >>> print(output.values)
        [0. 2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_relu6')
    return COOTensor(x.indices, nn_func.relu6(x.values), x.shape)


def csr_sinh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh(x_i)

    Args:
        x (CSRTensor): The input CSRTensor of hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_sinh(x)
        >>> print(output.values)
        [-1.1752012  3.6268604]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_sinh')
    return CSRTensor(x.indptr, x.indices, math_func.sinh(x.values), x.shape)


def coo_sinh(x: COOTensor) -> COOTensor:
    r"""
    Computes hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh(x_i)

    Args:
        x (COOTensor): The input COOTensor of hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_sinh(x)
        >>> print(output.values)
        [-1.1752012  3.6268604]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_sinh')
    return COOTensor(x.indices, math_func.sinh(x.values), x.shape)


def csr_ceil(x: CSRTensor) -> CSRTensor:
    r"""
    Rounds a CSRTensor up to the closest integer element-wise.

    .. math::

        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    Args:
        x (CSRTensor): The input CSRTensor with a dtype of float16 or float32, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_ceil(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_ceil')
    return CSRTensor(x.indptr, x.indices, math_func.ceil(x.values), x.shape)


def coo_ceil(x: COOTensor) -> COOTensor:
    r"""
    Rounds a COOTensor up to the closest integer element-wise.

    .. math::

        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    Args:
        x (COOTensor): The input COOTensor with a dtype of float16 or float32.

    Returns:
        COOTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_ceil(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_ceil')
    return COOTensor(x.indices, math_func.ceil(x.values), x.shape)


def csr_cosh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes hyperbolic cosine of input element-wise.

    .. math::

        out_i = \cosh(x_i)

    Args:
        x (CSRTensor): The input CSRTensor of hyperbolic cosine function, its rank must be in [0, 7] inclusive
            and data type must be float16, float32, float64, complex64 or complex128.

    Returns:
        CSRTensor, has the same shape as `x`.

    Raises:
        TypeError: If the dtype of `x` is not one of the following types:
                   float16, float32, float64, complex64, complex128.
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_cosh(x)
        >>> print(output.values)
        [1.5430807 3.7621956]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_cosh')
    return CSRTensor(x.indptr, x.indices, math_func.cosh(x.values), x.shape)


def coo_cosh(x: COOTensor) -> COOTensor:
    r"""
    Computes hyperbolic cosine of input element-wise.

    .. math::

        out_i = \cosh(x_i)

    Args:
        x (COOTensor): The input COOTensor of hyperbolic cosine function, its rank must be in [0, 7] inclusive
            and data type must be float16, float32, float64, complex64 or complex128.

    Returns:
        COOTensor, has the same shape as `x`.

    Raises:
        TypeError: If the dtype of `x` is not one of the following types:
                   float16, float32, float64, complex64, complex128.
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_cosh(x)
        >>> print(output.values)
        [1.5430807 3.7621956]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_cosh')
    return COOTensor(x.indices, math_func.cosh(x.values), x.shape)


def csr_softsign(x: CSRTensor) -> CSRTensor:
    r"""
    Softsign activation function.

    The function is shown as follows:

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Args:
        x (CSRTensor): Input CSRTensor, with float16 or float32 data type.

    Returns:
        CSRTensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_softsign(x)
        >>> print(output.values)
        [-0.5        0.6666667]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_softsign')
    return CSRTensor(x.indptr, x.indices, nn_func.softsign(x.values), x.shape)


def coo_softsign(x: COOTensor) -> COOTensor:
    r"""
    Softsign activation function.

    The function is shown as follows:

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Args:
        x (COOTensor): Input COOTensor, with float16 or float32 data type.

    Returns:
        COOTensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_softsign(x)
        >>> print(output.values)
        [-0.5        0.6666667]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_softsign')
    return COOTensor(x.indices, nn_func.softsign(x.values), x.shape)


def csr_log1p(x: CSRTensor) -> CSRTensor:
    """
    Returns the natural logarithm of one plus the input CSRTensor element-wise.

    .. math::
        out_i = {log_e}(x_i + 1)

    Args:
        x (CSRTensor): The input CSRTensor. With float16 or float32 data type.
            The value must be greater than -1.

    Returns:
        CSRTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_log1p(x)
        >>> print(output.values)
        [     -inf 1.0986123]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_log1p')
    return CSRTensor(x.indptr, x.indices, math_func.log1p(x.values), x.shape)


def coo_log1p(x: COOTensor) -> COOTensor:
    """
    Returns the natural logarithm of one plus the input COOTensor element-wise.

    .. math::
        out_i = {log_e}(x_i + 1)

    Args:
        x (COOTensor): The input COOTensor. With float16 or float32 data type.

    Returns:
        COOTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_log1p(x)
        >>> print(output.values)
        [     -inf 1.0986123]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_log1p')
    return COOTensor(x.indices, math_func.log1p(x.values), x.shape)


def csr_round(x: CSRTensor) -> CSRTensor:
    r"""
    Returns half to even of a CSRTensor element-wise.

    .. math::

        out_i \\approx x_i

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape and type as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_round(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_round')
    return CSRTensor(x.indptr, x.indices, math_func.round(x.values), x.shape)


def coo_round(x: COOTensor) -> COOTensor:
    """
    Returns half to even of a COOTensor element-wise.

    .. math::

        out_i \approx x_i

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape and type as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_round(x)
        >>> print(output.values)
        [-1.  2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_round')
    return COOTensor(x.indices, math_func.round(x.values), x.shape)


def csr_tanh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::

        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input CSRTensor.

    Args:
        x (CSRTensor): Input CSRTensor, with float16 or float32 data type.

    Returns:
        CSRTensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_tanh(x)
        >>> print(output.values)
        [-0.7615942  0.9640276]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_tanh')
    return CSRTensor(x.indptr, x.indices, math_func.tanh(x.values), x.shape)


def coo_tanh(x: COOTensor) -> COOTensor:
    r"""
    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::

        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input COOTensor.

    Args:
        x (COOTensor): Input COOTensor, with float16 or float32 data type.

    Returns:
        COOTensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_tanh(x)
        >>> print(output.values)
        [-0.7615942  0.9640276]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_tanh')
    return COOTensor(x.indices, math_func.tanh(x.values), x.shape)


def csr_asinh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes inverse hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh^{-1}(input_i)

    Args:
        x (CSRTensor): The input CSRTensor of inverse hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_asinh(x)
        >>> print(output.values)
        [-0.8813736  1.4436355]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_asinh')
    return CSRTensor(x.indptr, x.indices, math_func.asinh(x.values), x.shape)


def coo_asinh(x: COOTensor) -> COOTensor:
    r"""
    Computes inverse hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh^{-1}(input_i)

    Args:
        x (COOTensor): The input COOTensor of inverse hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_asinh(x)
        >>> print(output.values)
        [-0.8813736  1.4436355]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_asinh')
    return COOTensor(x.indices, math_func.asinh(x.values), x.shape)


def csr_neg(x: CSRTensor) -> CSRTensor:
    """
    Returns a CSRTensor with csr_negative values of the input CSRTensor element-wise.

    .. math::

        out_{i} = - x_{i}

    Args:
        x (CSRTensor): The input CSRTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and dtype as input.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_neg(x)
        >>> print(output.values)
        [ 1. -2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_neg')
    return CSRTensor(x.indptr, x.indices, math_func.neg(x.values), x.shape)


def coo_neg(x: COOTensor) -> COOTensor:
    """
    Returns a COOTensor with coo_negative values of the input COOTensor element-wise.

    .. math::

        out_{i} = - x_{i}

    Args:
        x (COOTensor): The input COOTensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and dtype as input.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_neg(x)
        >>> print(output.values)
        [ 1. -2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_neg')
    return COOTensor(x.indices, math_func.neg(x.values), x.shape)


def csr_acosh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes inverse hyperbolic cosine of the inputs element-wise.

    .. math::

        out_i = \cosh^{-1}(input_i)

    .. warning::
        Given an input CSRTensor x, the function computes inverse hyperbolic cosine of every element.
        Input range is [1, inf].

    Args:
        x (CSRTensor): The input CSRTensor of inverse hyperbolic cosine function, its rank must be in [0, 7] inclusive.

    Returns:
        CSRTensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_acosh(x)
        >>> print(output.values)
        [     nan 1.316958]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_acosh')
    return CSRTensor(x.indptr, x.indices, math_func.acosh(x.values), x.shape)


def coo_acosh(x: COOTensor) -> COOTensor:
    r"""
    Computes inverse hyperbolic cosine of the inputs element-wise.

    .. math::

        out_i = \cosh^{-1}(input_i)

    .. warning::
        Given an input COOTensor x, the function computes inverse hyperbolic cosine of every element.
        Input range is [1, inf].

    Args:
        x (COOTensor): The input COOTensor of inverse hyperbolic cosine function, its rank must be in [0, 7] inclusive.

    Returns:
        COOTensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_acosh(x)
        >>> print(output.values)
        [     nan 1.316958]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_acosh')
    return COOTensor(x.indices, math_func.acosh(x.values), x.shape)


def csr_isinf(x: CSRTensor) -> CSRTensor:
    r"""
    Determines which elements are inf or -inf for each position.

    .. math::

        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    where :math:`Inf` means not a number.

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_isinf(x)
        >>> print(output.values)
        [False False]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_isinf')
    return CSRTensor(x.indptr, x.indices, math_func.isinf(x.values), x.shape)


def coo_isinf(x: COOTensor) -> COOTensor:
    r"""
    Determines which elements are inf or -inf for each position.

    .. math::

        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    where :math:`Inf` means not a number.

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_isinf(x)
        >>> print(output.values)
        [False False]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_isinf')
    return COOTensor(x.indices, math_func.isinf(x.values), x.shape)


def csr_atanh(x: CSRTensor) -> CSRTensor:
    r"""
    Computes inverse hyperbolic tangent of the input element-wise.

    .. math::

        out_i = \\tanh^{-1}(x_{i})

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        x (CSRTensor): Input CSRTensor.
            The data type should be one of the following types: float16, float32.

    Returns:
        A CSRTensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_atanh(x)
        >>> print(output.values)
        [-inf  nan]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_atanh')
    return CSRTensor(x.indptr, x.indices, math_func.atanh(x.values), x.shape)


def coo_atanh(x: COOTensor) -> COOTensor:
    """
    Computes inverse hyperbolic tangent of the input element-wise.

    .. math::

        out_i = \tanh^{-1}(x_{i})

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        x (COOTensor): Input COOTensor.
            The data type should be one of the following types: float16, float32.

    Returns:
        A COOTensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_atanh(x)
        >>> print(output.values)
        [-inf  nan]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_atanh')
    return COOTensor(x.indices, math_func.atanh(x.values), x.shape)


def csr_sigmoid(x: CSRTensor) -> CSRTensor:
    r"""
    Sigmoid activation function.

    Computes Sigmoid of input element-wise. The Sigmoid function is defined as:

    .. math::

        \text{csr_sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    where :math:`x_i` is an element of the x.

    Args:
        x (CSRTensor): Input CSRTensor, the data type is float16, float32, float64, complex64 or complex128.

    Returns:
        CSRTensor, with the same type and shape as the x.

    Raises:
        TypeError: If dtype of `x` is not float16, float32, float64, complex64 or complex128.
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_sigmoid(x)
        >>> print(output.values)
        [0.26894143 0.8807971 ]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_sigmoid')
    return CSRTensor(x.indptr, x.indices, nn_func.sigmoid(x.values), x.shape)


def coo_sigmoid(x: COOTensor) -> COOTensor:
    r"""
    Sigmoid activation function.

    Computes Sigmoid of input element-wise. The Sigmoid function is defined as:

    .. math::

        \text{coo_sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    where :math:`x_i` is an element of the x.

    Args:
        x (COOTensor): Input COOTensor, the data type is float16, float32, float64, complex64 or complex128.

    Returns:
        COOTensor, with the same type and shape as the x.

    Raises:
        TypeError: If dtype of `x` is not float16, float32, float64, complex64 or complex128.
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_sigmoid(x)
        >>> print(output.values)
        [0.26894143 0.8807971 ]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_sigmoid')
    return COOTensor(x.indices, nn_func.sigmoid(x.values), x.shape)


def csr_abs(x: CSRTensor) -> CSRTensor:
    """
    Returns csr_absolute value of a CSRTensor element-wise.

    .. math::

        out_i = |x_i|

    Args:
        x (CSRTensor): The input CSRTensor.

    Returns:
        CSRTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_abs(x)
        >>> print(output.values)
        [1. 2.]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_abs')
    return CSRTensor(x.indptr, x.indices, math_func.abs(x.values), x.shape)


def coo_abs(x: COOTensor) -> COOTensor:
    """
    Returns coo_absolute value of a COOTensor element-wise.

    .. math::

        out_i = |x_i|

    Args:
        x (COOTensor): The input COOTensor.

    Returns:
        COOTensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_abs(x)
        >>> print(output.values)
        [1. 2.]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_abs')
    return COOTensor(x.indices, math_func.abs(x.values), x.shape)


def csr_sin(x: CSRTensor) -> CSRTensor:
    """
    Computes sine of the input element-wise.

    .. math::

        out_i = sin(x_i)

    Args:
        x (CSRTensor): Input CSRTensor.

    Returns:
        CSRTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a CSRTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64,
    complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indptr = Tensor([0, 1, 2, 2], dtype=mstype.int32)
        >>> indices = Tensor([3, 0], dtype=mstype.int32)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = CSRTensor(indptr, indices, values, shape)
        >>> output = ops.csr_sin(x)
        >>> print(output.values)
        [-0.84147096  0.9092974 ]
    """
    if not isinstance(x, CSRTensor):
        raise_type_error('Expects CSRTensor for csr_sin')
    return CSRTensor(x.indptr, x.indices, math_func.sin(x.values), x.shape)


def coo_sin(x: COOTensor) -> COOTensor:
    """
    Computes sine of the input element-wise.

    .. math::

        out_i = sin(x_i)

    Args:
        x (COOTensor): Input COOTensor.

    Returns:
        COOTensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a COOTensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64,
            complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
        >>> values = Tensor([-1, 2], dtype=mstype.float32)
        >>> shape = (3, 4)
        >>> x = COOTensor(indices, values, shape)
        >>> output = ops.coo_sin(x)
        >>> print(output.values)
        [-0.84147096  0.9092974 ]
    """
    if not isinstance(x, COOTensor):
        raise_type_error('Expects COOTensor for coo_sin')
    return COOTensor(x.indices, math_func.sin(x.values), x.shape)


__all__ = ["csr_cos", "csr_tan", "csr_exp", "csr_inv", "csr_relu", "csr_expm1", "csr_isfinite",
           "csr_asin", "csr_sqrt", "csr_log", "csr_isnan", "csr_acos", "csr_floor", "csr_atan",
           "csr_square", "csr_relu6", "csr_sinh", "csr_ceil", "csr_cosh", "csr_softsign",
           "csr_log1p", "csr_round", "csr_tanh", "csr_asinh", "csr_neg", "csr_acosh", "csr_isinf",
           "csr_atanh", "csr_sigmoid", "csr_abs", "csr_sin", "coo_cos", "coo_tan", "coo_exp",
           "coo_inv", "coo_relu", "coo_expm1", "coo_isfinite", "coo_asin", "coo_sqrt", "coo_log",
           "coo_isnan", "coo_acos", "coo_floor", "coo_atan", "coo_square", "coo_relu6", "coo_sinh",
           "coo_ceil", "coo_cosh", "coo_softsign", "coo_log1p", "coo_round", "coo_tanh",
           "coo_asinh", "coo_neg", "coo_acosh", "coo_isinf", "coo_atanh", "coo_sigmoid", "coo_abs",
           "coo_sin"]
__all__.sort()
