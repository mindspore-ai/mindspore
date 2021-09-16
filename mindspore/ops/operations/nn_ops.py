# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Operators for nn."""

import math
import operator
from functools import reduce, partial
import numpy as np
from mindspore import log as logger
from mindspore._checkparam import _check_3d_int_or_tuple
from ... import context
from .. import signature as sig
from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ...common import dtype as mstype
from ...common._decorator import deprecated
from ..primitive import Primitive, PrimitiveWithInfer, PrimitiveWithCheck, prim_attr_register


def _check_positive_int_or_tuple(arg_name, arg_value, prim_name, allow_four=False, ret_four=False):
    """
    Checks whether an argument is a positive int or tuple with 2 or 4(when allow_four is True) positive int elements.
    """

    def _raise_message():
        raise ValueError(f"For '{prim_name}' attr '{arg_name}' should be an positive int number or a tuple of two "
                         f"{'or four ' if allow_four else ''}positive int numbers, but got {arg_value}")

    def _get_return_value():
        if isinstance(arg_value, int):
            ret = (1, 1, arg_value, arg_value) if ret_four else (arg_value, arg_value)
        elif len(arg_value) == 2:
            ret = (1, 1, arg_value[0], arg_value[1]) if ret_four else arg_value
        elif len(arg_value) == 4:
            if not allow_four:
                _raise_message()
            ret = arg_value if ret_four else (arg_value[2], arg_value[3])
        else:
            _raise_message()
        return ret

    validator.check_value_type(arg_name, arg_value, (int, tuple), prim_name)
    ret_value = _get_return_value()
    for item in ret_value:
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
            continue
        _raise_message()
    return ret_value


def _check_shape(arg_name, arg_value, prim_name):
    """
    Checks whether an shape dims is a positive int elements.
    """

    def _raise_message():
        raise ValueError(f"For '{prim_name}' attr '{arg_name}' dims elements should be positive int numbers, "
                         f"but got {arg_value}")

    validator.check_value_type(arg_name, arg_value, (list, tuple), prim_name)
    for item in arg_value:
        if isinstance(item, int) and item > 0:
            continue
        _raise_message()
    return arg_value


def _update_attr_by_format(arg_value, arg_format):
    """
    If the format is NHWC, should modify the strides or dilation shape.
    """
    ret = arg_value
    if len(arg_value) == 4 and arg_format == "NHWC":
        ret = arg_value[1:] + (1,)

    return ret


class Flatten(PrimitiveWithInfer):
    r"""
    Flattens a tensor without changing its batch size on the 0-th axis.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, \ldots)` to be flattened, where :math:`N` is batch size.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimension.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If length of shape of `input_x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> flatten = ops.Flatten()
        >>> output = flatten(input_x)
        >>> print(output.shape)
        (1, 24)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x):
        validator.check_int(len(input_x), 1, Rel.GE, 'input_x rank', self.name)
        prod = 1 if len(input_x) == 1 else reduce(operator.mul, input_x[1:])
        return input_x[0], prod

    def infer_dtype(self, input_x):
        validator.check_subclass("input_x", input_x, mstype.tensor, self.name)
        return input_x


class AdaptiveAvgPool2D(PrimitiveWithInfer):
    r"""
    AdaptiveAvgPool2D operation.

    This operator applies a 2D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input planes.

    Args:
        output_size (Union[int, tuple]): The target output size is H x W.
            ouput_size can be a tuple, or a single H for H x H, and H and W can be int or None
            which means the output size is the same as the input.

    Inputs:
        - **input_x** (Tensor) - The input of AdaptiveAvgPool2D, which is a 3D or 4D tensor,
          with float16, float32, float64 data type.

    Outputs:
        Tensor, with the same type as the `input_x`.

        Shape of the output is `input_x_shape[:len(input_x_shape) - len(out_shape)] + out_shape`.

        If `output_size` contains `None`:

        - `out_shape = input_x_shape[-2] + output_size[1]`: If `output_size` is `(None, w)`
        - `out_shape = output_size[0] + input_x_shape[-1]`: If `output_size` is `(h, None)`
        - `out_shape = input_x_shape[-2:]: If output_size` is `(None, None)`

        If `output_size` does not contain `None`:

        - `out_shape = (h, h)`: If `output_size` is `h`
        - `out_shape = (h, w)`: If `output_size` is `(h, w)`

    Raises:
        ValueError: If `output_size` is a tuple and if `output_size` length is not 2.
        TypeError: If `input_x` is not a tensor.
        TypeError: If dtype of `input_x` is not float16, float32, float64.
        ValueError: If `input_x` dimension is less than or equal to output_size dimension.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> # case 1: output_size=(None, 2)
        >>> input_x = Tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), mindspore.float32)
        >>> adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D((None, 2))
        >>> output = adaptive_avg_pool_2d(input_x)
        >>> print(output)
        [[[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]
         [[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]
         [[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]]
        >>> # case 2: output_size=2
        >>> adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D(2)
        >>> output = adaptive_avg_pool_2d(input_x)
        >>> print(output)
        [[[3. 4.]
          [6. 7.]]
         [[3. 4.]
          [6. 7.]]
         [[3. 4.]
          [6. 7.]]]
        >>> # case 3: output_size=(1, 2)
        >>> adaptive_avg_pool_2d = ops.AdaptiveAvgPool2D((1, 2))
        >>> output = adaptive_avg_pool_2d(input_x)
        >>> print(output)
        [[[4.5 5.5]]
         [[4.5 5.5]]
         [[4.5 5.5]]]
    """

    @prim_attr_register
    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool2D."""
        validator.check_value_type("output_size", output_size, [int, tuple], self.name)
        if isinstance(output_size, tuple):
            validator.check_int(len(output_size), 2, Rel.EQ, 'length of output_size', self.name)
        self.output_size = (output_size, output_size) if isinstance(self.output_size, int) else output_size

    def infer_shape(self, x_shape):
        if len(x_shape) <= len(self.output_size):
            raise ValueError("input_x {} dimension should be larger than output_size {} "
                             "dimension".format(x_shape, self.output_size))
        validator.check_int(len(x_shape), 5, Rel.LT, 'input_x_dimensions', self.name)
        for input_x_dimension in x_shape:
            validator.check_int(input_x_dimension, 0, Rel.GT, 'input_x dimension', self.name)
        zipped = zip(self.output_size, x_shape[-len(self.output_size):])
        out_size = [i if i is not None else j for i, j in zipped]
        for item in out_size:
            validator.check_value_type("item of output_size", item, [int], self.name)
        self.add_prim_attr('output_size', out_size)
        output_shape = x_shape[:len(x_shape) - len(out_size)] + out_size
        return output_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, [mstype.float16, mstype.float32, mstype.float64],
                                           self.name)
        return x_dtype


class Softmax(Primitive):
    r"""
    Softmax operation.

    Applies the Softmax operation to the input tensor on the specified axis.
    Supposes a slice in the given aixs :math:`x`, then for each element :math:`x_i`,
    the Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)},

    where :math:`N` is the length of the tensor.

    Args:
        axis (Union[int, tuple]): The axis to perform the Softmax operation. Default: -1.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the logits.

    Raises:
        TypeError: If `axis` is neither an int nor a tuple.
        TypeError: If dtype of `logits` is neither float16 nor float32.
        ValueError: If `axis` is a tuple whose length is less than 1.
        ValueError: If `axis` is a tuple whose elements are not all in range [-len(logits.shape), len(logits.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> softmax = ops.Softmax()
        >>> output = softmax(logits)
        >>> print(output)
        [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        """Initialize Softmax."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type("axis", axis, [int, tuple], self.name)
        if isinstance(axis, int):
            self.add_prim_attr('axis', (axis,))
        for item in self.axis:
            validator.check_value_type("item of axis", item, [int], self.name)


class LogSoftmax(Primitive):
    r"""
    Log Softmax activation function.

    Applies the Log Softmax function to the input tensor on the specified axis.
    Supposes a slice in the given aixs, :math:`x` for each element :math:`x_i`,
    the Log Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    where :math:`N` is the length of the Tensor.

    Args:
        axis (int): The axis to perform the Log softmax operation. Default: -1.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the logits.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If dtype of `logits` is neither float16 nor float32.
        ValueError: If `axis` is not in range [-len(logits.shape), len(logits.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> log_softmax = ops.LogSoftmax()
        >>> output = log_softmax(logits)
        >>> print(output)
        [-4.4519143 -3.4519143 -2.4519143 -1.4519144 -0.4519144]
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        """Initialize LogSoftmax."""
        validator.check_value_type("axis", axis, [int], self.name)


class Softplus(Primitive):
    r"""
    Softplus activation function.

    Softplus is a smooth approximation to the ReLU function.
    It can be used to constrain the output of a machine to always be positive.
    The function is shown as follows:

    .. math::

        \text{output} = \log(1 + \exp(\text{x})),

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If the dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``  ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> softplus = ops.Softplus()
        >>> output = softplus(input_x)
        >>> print(output)
        [1.3132615 2.126928  3.0485873 4.01815   5.0067153]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Softplus"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class Softsign(PrimitiveWithInfer):
    r"""
    Softsign activation function.

    The function is shown as follows:

    .. math::

        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([0, -1, 2, 30, -30]), mindspore.float32)
        >>> softsign = ops.Softsign()
        >>> output = softsign(input_x)
        >>> print(output)
        [ 0.        -0.5         0.6666667  0.9677419 -0.9677419]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Softsign"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid('input_x', input_x, [mstype.float16, mstype.float32], self.name)
        return input_x


class ReLU(Primitive):
    r"""
    Computes ReLU (Rectified Linear Unit) of input tensors element-wise.

    It returns :math:`\max(x,\  0)` element-wise.

    Note:
        In general, this operator is more commonly used. The difference from `ReLuV2` is that the operator will
        output one more Mask.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with number data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is not number.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> relu = ops.ReLU()
        >>> output = relu(input_x)
        >>> print(output)
        [[0. 4. 0.]
         [2. 0. 9.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ReLU"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class Mish(PrimitiveWithInfer):
    r"""
    Computes MISH(A Self Regularized Non-Monotonic Neural Activation Function) of input tensors element-wise.

    The function is shown as follows:

    .. math::

        \text{output} = x * \tan(\log(1 + \exp(\text{x})))

    See more details in `A Self Regularized Non-Monotonic Neural Activation Function
    <https://arxiv.org/abs/1908.08681>`_.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Supported Platforms:
        ``Ascend``

    Raise:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> mish = ops.Mish()
        >>> output = mish(x)
        >>> print(output)
        [[-0.30273438  3.9974136 -0.015625]
         [ 1.9439697  -0.02929688 8.999999]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Mish"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid('x', x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class SeLU(PrimitiveWithInfer):
    r"""
    Computes SeLU (scaled exponential Linear Unit) of input tensors element-wise.

    The activation function is defined as:

    .. math::
        E_{i} =
        scale *
        \begin{cases}
        x_{i}, &\text{if } x_{i} \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`alpha` and :math:`scale` are pre-defined constants(:math:`alpha=1.67326324`
    and :math:`scale=1.05070098`).

    See more details in `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Supported Platforms:
        ``Ascend``

    Raise:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> selu = ops.SeLU()
        >>> output = selu(input_x)
        >>> print(output)
        [[-1.1113307 4.202804 -1.7575096]
        [ 2.101402 -1.7462534 9.456309 ]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SeLU"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensor_dtype_valid('x', x_dtype, valid_dtypes, self.name)
        return x_dtype


class ReLU6(PrimitiveWithCheck):
    r"""
    Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input tensors element-wise.

    .. math::

        \text{ReLU6}(x) = \min(\max(0,x), 6)

    It returns :math:`\min(\max(0,x), 6)` element-wise.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> relu6 = ops.ReLU6()
        >>> result = relu6(input_x)
        >>> print(result)
        [[0. 4. 0.]
         [2. 0. 6.]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ReLU6"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def check_shape(self, input_x):
        pass

    def check_dtype(self, input_x):
        validator.check_tensor_dtype_valid('input_x', input_x, (mstype.float16, mstype.float32), self.name)


class ReLUV2(Primitive):
    r"""
    Computes ReLU (Rectified Linear Unit) of input tensors element-wise.

    It returns :math:`\max(x,\  0)` element-wise.

    Note:
        The difference from `ReLu` is that the operator will output one more Mask,
        and the kernel of the operator is different from `ReLu`.

    Inputs:
        - **input_x** (Tensor) - The input tensor must be a 4-D tensor.

    Outputs:
        - **output** (Tensor) - Has the same type and shape as the `input_x`.
        - **mask** (Tensor) - A tensor whose data type must be uint8.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        ValueError: If shape of `input_x` is not 4-D.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]]]), mindspore.float32)
        >>> relu_v2 = ops.ReLUV2()
        >>> output, mask= relu_v2(input_x)
        >>> print(output)
        [[[[1. 0.]
           [0. 4.]]
          [[0. 6.]
           [7. 0.]]]]
        >>> print(mask)
        [[[[[1 0]
            [2 0]]
           [[2 0]
            [1 0]]]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ReLUV2"""
        self.init_prim_io_names(inputs=['x'], outputs=['output', 'mask'])


class Elu(PrimitiveWithInfer):
    r"""
    Computes exponential linear:

    .. math::

        \text{ELU}(x)= \left\{
        \begin{array}{align}
            \alpha(e^{x}  - 1) & \text{if } x \le 0\\
            x & \text{if } x \gt 0\\
        \end{array}\right.

    The data type of input tensor must be float.

    Args:
        alpha (float): The coefficient of negative factor whose type is float,
            only support '1.0' currently. Default: 1.0.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, has the same shape and data type as `input_x`.

    Raises:
        TypeError: If `alpha` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        ValueError: If `alpha` is not equal to 1.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> elu = ops.Elu()
        >>> output = elu(input_x)
        >>> print(output)
        [[-0.63212055  4.         -0.99966455]
         [ 2.         -0.99326205  9.        ]]
    """

    @prim_attr_register
    def __init__(self, alpha=1.0):
        """Initialize Elu"""
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_number("alpha", alpha, 1.0, Rel.EQ, self.name)

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid('input_x', input_x, mstype.float_type, self.name)
        return input_x


class HSwish(PrimitiveWithInfer):
    r"""
    Hard swish activation function.

    Applies hswish-type activation element-wise. The input is a Tensor with any valid shape.

    Hard swish is defined as:

    .. math::

        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> hswish = ops.HSwish()
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> result = hswish(input_x)
        >>> print(result)
        [-0.3333  -0.3333  0  1.666  0.6665]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize HSwish."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, xshape):
        return xshape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, (mstype.float16, mstype.float32), self.name)
        return x_dtype


class Sigmoid(PrimitiveWithInfer):
    r"""
    Sigmoid activation function.

    Computes Sigmoid of input element-wise. The Sigmoid function is defined as:

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the input_x.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> sigmoid = ops.Sigmoid()
        >>> output = sigmoid(input_x)
        >>> print(output)
        [0.7310586  0.880797   0.95257413 0.98201376 0.9933072 ]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Sigmoid."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, (mstype.float16, mstype.float32), self.name)
        return input_x


class HSigmoid(Primitive):
    r"""
    Hard sigmoid activation function.

    Applies hard sigmoid activation element-wise. The input is a Tensor with any valid shape.

    Hard sigmoid is defined as:

    .. math::

        \text{hsigmoid}(x_{i}) = max(0, min(1, \frac{x_{i} + 3}{6})),

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> hsigmoid = ops.HSigmoid()
        >>> input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> result = hsigmoid(input_x)
        >>> print(result)
        [0.3333 0.1666 0.5    0.8335 0.6665]
    """
    @prim_attr_register
    def __init__(self):
        """Initialize HSigmoid."""
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])


class Tanh(PrimitiveWithInfer):
    r"""
    Tanh activation function.

    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::

        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> tanh = ops.Tanh()
        >>> output = tanh(input_x)
        >>> print(output)
        [0.7615941 0.9640276 0.9950547 0.9993293 0.9999092]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, mstype.float_type, self.name)
        return input_x


class FusedBatchNorm(Primitive):
    r"""
    The FusedBatchNorm interface is deprecated, please use the BatchNorm interface.
    """

    def __init__(self, mode=0, epsilon=1e-5, momentum=0.1):
        raise TypeError("The FusedBatchNorm interface is deprecated, please use the BatchNorm interface.")


class FusedBatchNormEx(PrimitiveWithCheck):
    r"""
    The FusedBatchNormEx interface is deprecated, please use the BatchNorm interface.
    """

    def __init__(self, mode=0, epsilon=1e-5, momentum=0.1, data_format="NCHW"):
        raise TypeError("FusedBatchnormEx interface is deprecated, please use BatchNorm interface.")


class InstanceNorm(PrimitiveWithInfer):
    r"""
    Instance Normalization over a 4D input.

    This operator applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with
    additional channel dimension) as described in the paper `Instance Normalization: The Missing Ingredient for
    Fast Stylization <https://arxiv.org/abs/1607.08022>`_. It rescales and recenters the feature using a mini-batch
    of data and the learned parameters which can be described in the following formula.

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        epsilon (float): A small value added for numerical stability. Default: 1e-5.
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = momentum * running\_mean + (1 - momentum) * current\_mean`).
            Momentum value must be [0, 1]. Default: 0.1.

    Inputs:
        - **input_x** (Tensor) - The input of InstanceNorm, Tensor of shape :math:`(N, C)`,
          data type: float16 or float32.
        - **gamma** (Parameter) - Scale, Tensor of shape :math:`(C,)`,
          data type: float32.
        - **beta** (Parameter) - Bias, Tensor of shape :math:`(C,)`,
          data type: float32.
        - **mean** (Parameter) - Mean value, Tensor of shape :math:`(C,)`, data type: float32.
        - **variance** (Parameter) - Variance value, Tensor of shape :math:`(C,)`, data type: float32.

    Outputs:
        Tuple of 3 Tensors, the normalized input, the updated parameters.

        - **output_x** (Tensor) - The output of InstanceNorm, same type and shape as the `input_x`.
        - **updated_moving_mean** (Tensor) - Updated mean value, Tensor of shape :math:`(NC,)`, data type: float32.
        - **updated_moving_variance** (Tensor) - Updated variance value, Tensor of shape :math:`(NC,)`,
          data type: float32.

    Supported Platforms:
        ``GPU``

    Raise:
        TypeError: If `epsilon` or `momentum` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        TypeError: If dtype of `gamma`, `beta` or `mean` is not float32.
        ValueError: If `epsilon` is not in the range of [0, 1).
        ValueError: If `momentum` is not in the range of [0, 1].

    Examples:
        >>> class InstanceNormNet(nn.Cell):
        >>>     def __init__(self):
        >>>         super(InstanceNormNet, self).__init__()
        >>>         self.instance_norm = ops.InstanceNorm()
        >>>         self.gamma = Parameter(Tensor(np.ones([64]), mindspore.float32), name="gamma")
        >>>         self.beta = Parameter(Tensor(np.ones([64]), mindspore.float32), name="beta")
        >>>         self.mean = Parameter(Tensor(np.ones([64]), mindspore.float32), name="mean")
        >>>         self.variance = Parameter(Tensor(np.ones([64]), mindspore.float32), name="variance")
        >>>
        >>>     def construct(self, input_x):
        >>>         out = self.instance_norm(input_x, self.gamma, self.beta, self.mean, self.variance)
        >>>         return out
        >>>
        >>> input_x = Tensor(np.ones([128, 64, 32, 64]), mindspore.float32)
        >>> net = InstanceNormNet()
        >>> output = net(input_x)
        >>> result = output[0].shape
        >>> print(result)
        (128, 64, 32, 64)
    """
    __mindspore_signature__ = (
        sig.make_sig('input_x', dtype=sig.sig_dtype.T2),
        sig.make_sig('gamma', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('beta', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('mean', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('variance', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
    )

    @prim_attr_register
    def __init__(self, epsilon=1e-5, momentum=0.1):
        """Initialize InstanceNorm."""
        self.init_prim_io_names(inputs=['x', 'gamma', 'beta', 'mean', 'variance'],
                                outputs=['y', 'save_mean', 'save_variance'])
        self.epsilon = validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        self.momentum = validator.check_float_range(momentum, 0, 1, Rel.INC_BOTH, 'momentum', self.name)
        self._update_parameter = True

    def infer_shape(self, input_x, gamma, beta, mean, variance):
        input_shape_norm = input_x
        validator.check_equal_int(len(gamma), 1, "gamma rank", self.name)
        validator.check("gamma shape", gamma, "beta shape", beta, Rel.EQ, self.name)
        validator.check("gamma shape[0]", gamma[0], "input channel", input_shape_norm[1], Rel.EQ, self.name)
        validator.check_equal_int(len(mean), 1, "mean rank", self.name)

        validator.check("mean shape", mean, "variance shape", variance, Rel.EQ, self.name)
        validator.check("mean shape", mean, "gamma shape", gamma, Rel.EQ, self.name)
        save_mean_shape = gamma
        save_mean_shape[0] = save_mean_shape[0] * input_shape_norm[0]
        return input_x, save_mean_shape, save_mean_shape

    def infer_dtype(self, input_x, gamma, beta, mean, variance):
        validator.check_tensor_dtype_valid("input_x", input_x, [mstype.float16, mstype.float32], self.name)
        args = {"gamma": gamma, "beta": beta}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float32], self.name)
        args_moving = {"mean": mean, "variance": variance}
        valid_dtypes = [mstype.tensor_type(mstype.float32)]
        validator.check_types_same_and_valid(args_moving, valid_dtypes, self.name)
        return input_x, gamma, gamma


class BNTrainingReduce(PrimitiveWithInfer):
    """
    For the BatchNorm operation this operator updates the moving averages for training and is used in conjunction with
    BNTrainingUpdate.

    Inputs:
        - **x** (Tensor) - A 4-D Tensor with float16 or float32 data type. Tensor of shape :math:`(N, C, A, B)`.

    Outputs:
        - **sum** (Tensor) - A 1-D Tensor with float32 data type. Tensor of shape :math:`(C,)`.
        - **square_sum** (Tensor) - A 1-D Tensor with float16 or float32 data type. Tensor of shape :math:`(C,)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.ones([128, 3, 32, 3]), mindspore.float32)
        >>> bn_training_reduce = ops.BNTrainingReduce()
        >>> output = bn_training_reduce(x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value=
        [ 1.22880000e+04, 1.22880000e+04, 1.22880000e+04]), Tensor(shape=[3], dtype=Float32, value=
        [ 1.22880000e+04, 1.22880000e+04, 1.22880000e+04]))
    """

    @prim_attr_register
    def __init__(self):
        """Initialize BNTrainingReduce."""
        self.init_prim_io_names(inputs=['x'], outputs=['sum', 'square_sum'])

    def infer_shape(self, x_shape):
        validator.check_equal_int(len(x_shape), 4, "x rank", self.name)
        return [x_shape[1]], [x_shape[1]]

    def infer_dtype(self, x_type):
        validator.check_tensor_dtype_valid("x", x_type, [mstype.float16, mstype.float32], self.name)
        return x_type, x_type


class BNTrainingUpdate(PrimitiveWithInfer):
    """
    For the BatchNorm operation, this operator updates the moving averages for training and is used in conjunction with
    BNTrainingReduce. Where the moving averages is a method of analyzing data points by creating a series of averages
    of different subsets of the entire data set.

    .. warning::
        For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.

    Args:
        isRef (bool): If a ref. Default: True. Ref indicates whether to enable the output multiplexing input address.
        epsilon (float): A small value added to variance avoid dividing by zero. Default: 1e-5.
        factor (float): A weight for updating the mean and variance. Default: 0.1.

    Inputs:
        - **input_x** (Tensor) - A 4-D Tensor with float16 or float32 data type. Tensor of shape :math:`(N, C, A, B)`.
        - **sum** (Tensor) - A 1-D Tensor with float16 or float32 data type for the output of operator BNTrainingReduce.
          Tensor of shape :math:`(C,)`.
        - **square_sum** (Tensor) - A 1-D Tensor with float16 or float32 data type for the output of operator
          BNTrainingReduce. Tensor of shape :math:`(C,)`.
        - **scale** (Tensor) - A 1-D Tensor with float16 or float32, for the scaling factor.
          Tensor of shape :math:`(C,)`.
        - **offset** (Tensor) - A 1-D Tensor with float16 or float32, for the scaling offset.
          Tensor of shape :math:`(C,)`.
        - **mean** (Tensor) - A 1-D Tensor with float16 or float32, for the scaling mean. Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - A 1-D Tensor with float16 or float32, for the update variance.
          Tensor of shape :math:`(C,)`.

    Outputs:
        - **y** (Tensor) - Tensor, has the same shape and data type as `input_x`.
        - **mean** (Tensor) - Tensor for the updated mean, with float32 data type.
          Has the same shape as `variance`.
        - **variance** (Tensor) - Tensor for the updated variance, with float32 data type.
          Has the same shape as `variance`.
        - **batch_mean** (Tensor) - Tensor for the mean of `input_x`, with float32 data type.
          Has the same shape as `variance`.
        - **batch_variance** (Tensor) - Tensor for the mean of `variance`, with float32 data type.
          Has the same shape as `variance`.

    Raises:
        TypeError: If `isRef` is not a bool.
        TypeError: If dtype of `epsilon` or `factor` is not float.
        TypeError: If `input_x`, `sum`, `square_sum`, `scale`, `offset`, `mean` or `variance` is not a Tensor.
        TypeError: If dtype of `input_x`, `sum`, `square_sum`, `scale`, `offset`, `mean` or `variance` is neither
                   float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.ones([1, 2, 2, 2]), mindspore.float32)
        >>> sum_val = Tensor(np.ones([2]), mindspore.float32)
        >>> square_sum = Tensor(np.ones([2]), mindspore.float32)
        >>> scale = Tensor(np.ones([2]), mindspore.float32)
        >>> offset = Tensor(np.ones([2]), mindspore.float32)
        >>> mean = Tensor(np.ones([2]), mindspore.float32)
        >>> variance = Tensor(np.ones([2]), mindspore.float32)
        >>> bn_training_update = ops.BNTrainingUpdate()
        >>> output = bn_training_update(input_x, sum_val, square_sum, scale, offset, mean, variance)
        >>> print(output)
        (Tensor(shape=[1, 2, 2, 2], dtype=Float32, value=
        [[[[ 2.73200464e+00,  2.73200464e+00],
           [ 2.73200464e+00,  2.73200464e+00]],
          [[ 2.73200464e+00,  2.73200464e+00],
           [ 2.73200464e+00,  2.73200464e+00]]]]), Tensor(shape=[2], dtype=Float32, value= [9.24999952e-01,
        9.24999952e-01]), Tensor(shape=[2], dtype=Float32, value= [ 9.24999952e-01, 9.24999952e-01]),
        Tensor(shape=[2], dtype=Float32, value= [ 2.50000000e-01, 2.50000000e-01]), Tensor(shape=[2], dtype=Float32,
        value= [ 1.87500000e-01, 1.87500000e-01]))
    """

    @prim_attr_register
    def __init__(self, isRef=True, epsilon=1e-5, factor=0.1):
        """Initialize BNTrainingUpdate."""
        self.init_prim_io_names(inputs=['x', 'sum', 'square_sum', 'scale', 'b', 'mean', 'variance'],
                                outputs=['y', 'running_mean', 'running_variance', 'save_mean', 'save_inv_variance'])
        validator.check_value_type("isRef", isRef, [bool], self.name)
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_value_type("factor", factor, [float], self.name)
        self.epsilon = validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', 'BNTrainingUpdate')
        self.factor = validator.check_float_range(factor, 0, 1, Rel.INC_BOTH, 'factor', 'BNTrainingUpdate')

    def infer_shape(self, x, sum, square_sum, scale, b, mean, variance):
        validator.check_equal_int(len(x), 4, "x rank", self.name)
        validator.check_equal_int(len(sum), 1, "sum rank", self.name)
        validator.check_equal_int(len(square_sum), 1, "square_sum rank", self.name)
        validator.check_equal_int(len(scale), 1, "scale rank", self.name)
        validator.check_equal_int(len(b), 1, "b rank", self.name)
        validator.check_equal_int(len(mean), 1, "mean rank", self.name)
        validator.check_equal_int(len(variance), 1, "variance rank", self.name)
        validator.check("sum shape", sum[0], "x_shape[1]", x[1], Rel.EQ, self.name)
        validator.check("square_sum shape", square_sum, "sum", sum, Rel.EQ, self.name)
        validator.check("scale shape", scale[0], "x_shape[1]", x[1], Rel.EQ, self.name)
        validator.check("offset shape", b[0], "x_shape[1]", x[1], Rel.EQ, self.name)
        validator.check("mean shape", mean[0], "x_shape[1]", x[1], Rel.EQ, self.name)
        validator.check("variance shape", variance[0], "x_shape[1]", x[1], Rel.EQ, self.name)
        return x, variance, variance, variance, variance

    def infer_dtype(self, x, sum, square_sum, scale, b, mean, variance):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x", "sum", "square_sum", "scale", "b", "mean", "variance"),
                  (x, sum, square_sum, scale, b, mean, variance)))
        return x, variance, variance, variance, variance


class BatchNorm(PrimitiveWithInfer):
    r"""
    Batch Normalization for input data and updated parameters.

    Batch Normalization is widely used in convolutional neural networks. This operation
    applies Batch Normalization over inputs to avoid internal covariate shift as described
    in the paper `Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    features using a mini-batch of data and the learned parameters can be described
    in the following formula,

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon, :math:`mean` is the mean of x,
    :math:`variance` is the variance of x.

    .. warning::
        - If the operation is used for inference, and outputs "reserve_space_1" and "reserve_space_2" are available,
          then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
        - For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.

    Args:
        is_training (bool): If `is_training` is True, `mean` and `variance` are computed during training.
            If `is_training` is False, they're loaded from checkpoint during inference. Default: False.
        epsilon (float): A small value added for numerical stability. Default: 1e-5.
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1]. Default: 0.1.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: "NCHW".

    Inputs:
        If `is_training` is False, inputs are Tensors.

        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Tensor) - Tensor of shape :math:`(C,)`, with float16 or float32 data type.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.

        If `is_training` is True, `scale`, `bias`, `mean` and `variance` are Parameters.

        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Parameter) - Parameter of shape :math:`(C,)`, with float16 or float32 data type.
        - **bias** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.
        - **mean** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.
        - **variance** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.

    Outputs:
        Tuple of 5 Tensors, the normalized inputs and the updated parameters.

        - **output_x** (Tensor) - The same type and shape as the input_x. The shape is :math:`(N, C)`.
        - **updated_scale** (Tensor) - Tensor of shape :math:`(C,)`.
        - **updated_bias** (Tensor) - Tensor of shape :math:`(C,)`.
        - **reserve_space_1** (Tensor) - Tensor of shape :math:`(C,)`.
        - **reserve_space_2** (Tensor) - Tensor of shape :math:`(C,)`.

    Raises:
        TypeError: If `is_training` is not a bool.
        TypeError: If dtype of `epsilon` or `momentum` is not float.
        TypeError: If `data_format` is not a str.
        TypeError: If `input_x`, `scale`, `bias`, `mean` or `variance` is not a Tensor.
        TypeError: If dtype of `input_x`, `scale` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.ones([2, 2]), mindspore.float32)
        >>> scale = Tensor(np.ones([2]), mindspore.float32)
        >>> bias = Tensor(np.ones([2]), mindspore.float32)
        >>> mean = Tensor(np.ones([2]), mindspore.float32)
        >>> variance = Tensor(np.ones([2]), mindspore.float32)
        >>> batch_norm = ops.BatchNorm()
        >>> output = batch_norm(input_x, scale, bias, mean, variance)
        >>> print(output[0])
        [[1. 1.]
         [1. 1.]]
    """

    __mindspore_signature__ = (
        sig.make_sig('input_x', dtype=sig.sig_dtype.T1),
        sig.make_sig('scale', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T2),
        sig.make_sig('bias', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T2),
        sig.make_sig('mean', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T3),
        sig.make_sig('variance', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T3)
    )

    @prim_attr_register
    def __init__(self, is_training=False, epsilon=1e-5, momentum=0.1, data_format="NCHW"):
        """Initialize BatchNorm."""
        if is_training is False:
            self.set_signatures(tuple())
        validator.check_value_type('is_training', is_training, (bool,), self.name)
        validator.check_float_range(epsilon, 0, 1, Rel.INC_RIGHT, 'epsilon', self.name)
        validator.check_float_range(momentum, 0, 1, Rel.INC_BOTH, 'momentum', self.name)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the format is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)
        self.init_prim_io_names(inputs=['x', 'scale', 'offset', 'mean', 'variance'],
                                outputs=['y', 'batch_mean', 'batch_variance', 'reserve_space_1', 'reserve_space_2'])

    def infer_shape(self, input_x, scale, bias, mean, variance):
        input_x_channel = input_x[-1] if self.format == "NHWC" else input_x[1]
        validator.check_equal_int(len(scale), 1, "scale rank", self.name)
        validator.check("scale shape", scale, "bias shape", bias, Rel.EQ, self.name)
        validator.check("scale shape[0]", scale[0], "input_x channel", input_x_channel, Rel.EQ, self.name)
        if not self.is_training:
            validator.check_equal_int(len(mean), 1, "mean rank", self.name)
            validator.check("mean shape", mean, "variance shape", variance, Rel.EQ, self.name)
            validator.check("mean shape", mean, "scale shape", scale, Rel.EQ, self.name)
        return input_x, scale, scale, scale, scale

    def infer_dtype(self, input_x, scale, bias, mean, variance):
        validator.check_tensor_dtype_valid("input_x", input_x, [mstype.float16, mstype.float32], self.name)
        args = {"scale": scale, "bias": bias, "mean": mean, "variance": variance}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        return input_x, mstype.float32, mstype.float32, mstype.float32, mstype.float32


class Conv2D(Primitive):
    r"""
    2D convolution layer.

    Applies a 2D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, H_{in}, W_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`H` is height, :math:`W` is width, :math:`X_i` is
    the :math:`i^{th}` input value and :math:`b_i` indicates the deviation value of the :math:`i^{th}` input value.
    For each batch of shape :math:`(C_{in}, H_{in}, W_{in})`, the formula is defined as:

    .. math::

        out_j = \sum_{i=0}^{C_{in} - 1} ccor(W_{ij}, X_i) + b_j,

    where :math:`ccor` is the cross correlation operator, :math:`C_{in}` is the input channel number, :math:`j` ranges
    from :math:`0` to :math:`C_{out} - 1`, :math:`W_{ij}` corresponds to the :math:`i`-th channel of the :math:`j`-th
    filter and :math:`out_{j}` corresponds to the :math:`j`-th channel of the output. :math:`W_{ij}` is a slice
    of kernel and it has shape :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}` and :math:`\text{kernel_size[1]}` are the height and width of the
    convolution kernel. The full kernel has shape
    :math:`(C_{out}, C_{in} // \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where group is the group number to split the input in the channel dimension.

    If the 'pad_mode' is set to be "valid", the output height and width will be
    :math:`\left \lfloor{1 + \frac{H_{in} + \text{padding[0]} + \text{padding[1]} - \text{kernel_size[0]} -
    (\text{kernel_size[0]} - 1) \times (\text{dilation[0]} - 1) }{\text{stride[0]}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + \text{padding[2]} + \text{padding[3]} - \text{kernel_size[1]} -
    (\text{kernel_size[1]} - 1) \times (\text{dilation[1]} - 1) }{\text{stride[1]}}} \right \rfloor` respectively.
    Where :math:`dialtion` is Spacing between kernel elements, :math:`stride` is The step length of each step,
    :math:`padding` is zero-padding added to both sides of the input.


    The first introduction can be found in paper `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_. More detailed introduction can be found here:
    http://cs231n.github.io/convolutional-networks/.

    Args:
        out_channel (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both the height and the width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input `x`. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `pad`
              must be 0.

            - valid: Adopts the way of discarding. The possible largest height and width of output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `pad` must be 0.

            - pad: Implicit paddings on both sides of the input `x`. The number of `pad` will be padded to the input
              Tensor borders. `pad` must be greater than or equal to 0.
        pad (Union(int, tuple[int])): Implicit paddings on both sides of the input `x`. If `pad` is one integer,
                    the paddings of top, bottom, left and right are the same, equal to pad. If `pad` is a tuple
                    with four integers, the paddings of top, bottom, left and right will be equal to pad[0],
                    pad[1], pad[2], and pad[3] accordingly. Default: 0.
        stride (Union(int, tuple[int])): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        dilation (Union(int, tuple[int])): The data type is int or a tuple of 2 integers. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the height and width of the
                                      input `x`. Default: 1.
        group (int): Splits input into groups. Default: 1.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Default: "NCHW".

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **weight** (Tensor) - Set size of kernel is :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
          then the shape is :math:`(C_{out}, C_{in}, \text{kernel_size[0]}, \text{kernel_size[1]})`.

    Outputs:
        Tensor, the value that applied 2D convolution. The shape is :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int nor a tuple.
        TypeError: If `out_channel` or `group` is not an int.
        ValueError: If `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` it not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
        >>> weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
        >>> conv2d = ops.Conv2D(out_channel=32, kernel_size=3)
        >>> output = conv2d(x, weight)
        >>> print(output.shape)
        (10, 32, 30, 30)
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCHW"):
        """Initialize Conv2D"""
        self.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
        self.kernel_size = _check_positive_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_positive_int_or_tuple('stride', stride, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('stride', self.stride)
        self.dilation = _check_positive_int_or_tuple('dilation', dilation, self.name, allow_four=True, ret_four=True)
        self.add_prim_attr('dilation', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.padding = pad
        self.pad_mode = validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.name)

        if pad_mode != 'pad' and pad != (0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be zero when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' and 'pad_mode' is {pad_mode}.")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)

        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the format is {self.format} "
                             f"and platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.group = validator.check_positive_int(group, 'group', self.name)
        self.add_prim_attr('groups', self.group)


class DepthwiseConv2dNative(PrimitiveWithInfer):
    r"""
    Returns the depth-wise convolution value for the input.

    Applies depthwise conv2d for the input, which will generate more channels with channel_multiplier.
    Given an input tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})` where :math:`N` is the batch size,
    :math:`C` is the channels, :math:`H` is height, :math:`W` is width and a filter tensor with kernel size
    :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`, where :math:`\text{kernel_size[0]}` indicates the
    kernel_size of height, :math:`\text{kernel_size[1]}` indicates the kernel_size of width, containing
    :math:`C_{in} * \text{channel_multiplier}` convolutional filters of depth 1;
    it applies different filters to each input channel (channel_multiplier channels
    for each input channel has the default value 1), then concatenates the results together. The output has
    :math:`C_{in} * \text{channel_multiplier}` channels.

    Args:
        channel_multiplier (int): The multiplier for the original output convolution. Its value must be greater than 0.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both the height and the width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        mode (int): Modes for different convolutions. 0 Math convolution, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 3.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input `x`. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible. Otherwise, the
              last extra padding will be done from the bottom and the right side. If this mode is set, `pad`
              must be 0.

            - valid: Adopts the way of discarding. The possible largest height and width of output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `pad` must be 0.

            - pad: Implicit paddings on both sides of the input `x`. The number of `pad` will be padded to the input
              Tensor borders. `pad` must be greater than or equal to 0.
        pad (Union[int, tuple[int]]): Implicit paddings on both sides of the input `x`. If `pad` is one integer,
                    the paddings of top, bottom, left and right are the same, equal to pad. If `pad` is a tuple
                    with four integers, the paddings of top, bottom, left and right will be equal to pad[0],
                    pad[1], pad[2], and pad[3] accordingly. Default: 0.
        stride (Union(int, tuple[int])): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        dilation (Union(int, tuple[int])): The data type is int or a tuple of 2 integers. Specifies the dilation rate
                                      to use for dilated convolution. If set to be :math:`k > 1`, there will
                                      be :math:`k - 1` pixels skipped for each sampling location. Its value must
                                      be greater or equal to 1 and bounded by the height and width of the
                                      input `x`. Default: 1.
        group (int): Splits input into groups. Default: 1.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        - **weight** (Tensor) - Set the size of kernel as :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
          then the shape is :math:`(K, C_{in}, \text{kernel_size[0]}, \text{kernel_size[1]})`, `K` must be 1.

    Outputs:
        Tensor of shape :math:`(N, C_{in} * \text{channel_multiplier}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int nor a tuple.
        TypeError: If `channel_multiplier` or `group` is not an int.
        ValueError: If `stride` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of the following:'same', 'valid' or 'pad'.
        ValueError: If `pad_mode` it not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
        >>> weight = Tensor(np.ones([1, 32, 3, 3]), mindspore.float32)
        >>> depthwise_conv2d = ops.DepthwiseConv2dNative(channel_multiplier=3, kernel_size=(3, 3))
        >>> output = depthwise_conv2d(x, weight)
        >>> print(output.shape)
        (10, 96, 30, 30)
    """

    @prim_attr_register
    def __init__(self,
                 channel_multiplier,
                 kernel_size,
                 mode=3,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1):
        """Initialize DepthwiseConv2dNative"""
        logger.warning("WARN_DEPRECATED: The usage of DepthwiseConv2dNative is deprecated."
                       " Please use nn.Conv2D.")
        self.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
        self.kernel_size = _check_positive_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_positive_int_or_tuple('stride', stride, self.name)
        if self.stride[0] != self.stride[1]:
            raise ValueError("The height and width of stride should be equal,"
                             f"but got height:{self.stride[0]},  width:{self.stride[1]}")
        self.add_prim_attr('stride', (1, 1, self.stride[0], self.stride[1]))

        self.dilation = _check_positive_int_or_tuple('dilation', dilation, self.name)
        if self.dilation[0] != self.dilation[1]:
            raise ValueError("The height and width of dilation should be equal,"
                             f"but got height:{self.dilation[0]},  width:{self.dilation[1]}")
        self.add_prim_attr('dilation', (1, 1, self.dilation[0], self.dilation[1]))
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.padding = pad
        self.pad_mode = validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.name)
        if pad_mode != 'pad' and pad != (0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be zero when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {pad} and 'pad_mode' is {pad_mode}")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)
        self.mode = validator.check_equal_int(mode, 3, "mode", self.name)
        self.add_prim_attr('data_format', "NCHW")
        self.channel_multiplier = validator.check_positive_int(channel_multiplier, "channel_multiplier", self.name)
        self.group = validator.check_positive_int(group, "group", self.name)
        self.add_prim_attr('offset_a', 0)

    def infer_shape(self, x_shape, w_shape, b_shape=None):
        validator.check_equal_int(len(w_shape), 4, "weight rank", self.name)
        validator.check_equal_int(len(x_shape), 4, "x rank", self.name)
        validator.check("x_shape[1]", x_shape[1], "w_shape[1]", w_shape[1], Rel.EQ, self.name)
        validator.check('kernel_size', self.kernel_size, 'w_shape[2:4]', tuple(w_shape[2:4]), Rel.EQ, self.name)

        kernel_size_n, _, kernel_size_h, kernel_size_w = w_shape
        _, _, stride_h, stride_w = self.stride
        _, _, dilation_h, dilation_w = self.dilation
        if kernel_size_n != 1:
            raise ValueError(f"For '{self.name}', the batch of 'weight' should be 1, but got {kernel_size_n}")
        if self.pad_mode == "valid":
            h_out = math.ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.pad_mode == "same":
            h_out = math.ceil(x_shape[2] / stride_h)
            w_out = math.ceil(x_shape[3] / stride_w)

            pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
        elif self.pad_mode == 'pad':
            pad_top, pad_bottom, pad_left, pad_right = self.padding

            h_out = 1 + (x_shape[2] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) \
                    / stride_h
            w_out = 1 + (x_shape[3] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) \
                    / stride_w
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)

        self.pad_list = (pad_top, pad_bottom, pad_left, pad_right)
        self.add_prim_attr('pad_list', self.pad_list)

        out_channel = self.channel_multiplier * x_shape[1]
        out_shape = [x_shape[0], out_channel, h_out, w_out]
        return out_shape

    def infer_dtype(self, x_dtype, w_dtype, b_dtype=None):
        args = {'x': x_dtype, 'w': w_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        if x_dtype.element_type() == mstype.int8:
            return mstype.tensor_type(mstype.int32)
        return x_dtype


class _Pool(PrimitiveWithInfer):
    r"""
    Performs max/avg pooling operation.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of the kernel, that must be a tuple
           of two `int` for height and width. Default: 1.
        strides (Union[int, tuple[int]]): The stride of the window, that must be
            a tuple of two `int` for height and width. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: "NCHW".
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        """Initialize _Pool."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.name)
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.is_maxpoolwithargmax = (self.name == "MaxPoolWithArgmax")
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the format is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        if not self.is_maxpoolwithargmax:
            self.add_prim_attr('data_format', self.format)

        self.kernel_size = _check_positive_int_or_tuple(
            "kernel_size", kernel_size, self.name, allow_four=False, ret_four=True)
        if self.is_maxpoolwithargmax:
            self.kernel_size = (1, self.kernel_size[-2], self.kernel_size[-1], 1)
        self.add_prim_attr("kernel_size", self.kernel_size)

        self.strides = _check_positive_int_or_tuple("strides", strides, self.name, allow_four=False, ret_four=True)
        if self.is_maxpoolwithargmax:
            self.strides = (1, self.strides[-2], self.strides[-1], 1)
        self.add_prim_attr("strides", self.strides)

    def infer_shape(self, x_shape):
        x_shape_norm = x_shape if self.format == "NCHW" else [x_shape[0], x_shape[3], x_shape[1], x_shape[2]]
        validator.check_equal_int(len(x_shape_norm), 4, "x rank", self.name)
        batch, channel, input_h, input_w = x_shape_norm
        if self.is_maxpoolwithargmax:
            _, kernel_h, kernel_w, _ = self.kernel_size
            _, stride_h, stride_w, _ = self.strides
        else:
            _, _, kernel_h, kernel_w = self.kernel_size
            _, _, stride_h, stride_w = self.strides

        if self.pad_mode == "VALID":
            out_h = math.ceil((input_h - (kernel_h - 1)) / stride_h)
            out_w = math.ceil((input_w - (kernel_w - 1)) / stride_w)
        elif self.pad_mode == "SAME":
            out_h = math.ceil(input_h / stride_h)
            out_w = math.ceil(input_w / stride_w)
        out_shape = [batch, channel, out_h, out_w] if self.format == "NCHW" else [batch, out_h, out_w, channel]

        for shape_value in out_shape:
            if shape_value <= 0:
                raise ValueError(f"For '{self.name}', the each element of the output shape must be larger than 0, "
                                 f"but got output shape: {out_shape}. The input shape: {x_shape}, "
                                 f"kernel size: {self.kernel_size}, strides: {self.strides}."
                                 f"Please check the official api documents for "
                                 f"more information about the output.")
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input", x_dtype, mstype.tensor, self.name)
        return x_dtype


class MaxPool(_Pool):
    r"""
    Max pooling operation.

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents height and width of the kernel, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str) : The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `input` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.arange(1 * 3 * 3 * 4).reshape((1, 3, 3, 4)), mindspore.float32)
        >>> maxpool_op = ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)
        >>> output = maxpool_op(x)
        >>> print(output)
        [[[[ 5.  6.  7.]
           [ 9. 10. 11.]]
          [[17. 18. 19.]
           [21. 22. 23.]]
          [[29. 30. 31.]
           [33. 34. 35.]]]]
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        """Initialize MaxPool."""
        super(MaxPool, self).__init__(kernel_size, strides, pad_mode, data_format)


class MaxPoolWithArgmax(_Pool):
    r"""
    Performs max pooling on the input Tensor and returns both max values and indices.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value and arg
            value, is an int number that represents height and width of the kernel, or a tuple of
            two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          Data type must be float16 or float32.

    Outputs:
        Tuple of 2 Tensors, representing the maxpool result and where the max values are generated.

        - **output** (Tensor) -  Maxpooling result, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.
          It has the same data type as `x`.
        - **mask** (Tensor) -  Max values' index represented by the mask. Data type is int32.

    Raises:
        TypeError: If the data type of `x` is neither float16 nor float32.
        TypeError: If `kernel_size` or `strides` is neither an int nor a tuple.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.arange(1 * 3 * 3 * 4).reshape((1, 3, 3, 4)), mindspore.float32)
        >>> maxpool_arg_op = ops.MaxPoolWithArgmax(pad_mode="VALID", kernel_size=2, strides=1)
        >>> output_tensor, argmax = maxpool_arg_op(x)
        >>> print(output_tensor)
        [[[[ 5.  6.  7.]
           [ 9. 10. 11.]]
          [[17. 18. 19.]
           [21. 22. 23.]]
          [[29. 30. 31.]
           [33. 34. 35.]]]]
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        """Initialize MaxPoolWithArgmax."""
        super(MaxPoolWithArgmax, self).__init__(kernel_size, strides, pad_mode, data_format)

    def infer_shape(self, x_shape):
        out_shape = _Pool.infer_shape(self, x_shape)
        return out_shape, out_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, (mstype.float16, mstype.float32), self.name)
        argmax_dtype = mstype.int32
        return x_dtype, argmax_dtype


class MaxPool3D(PrimitiveWithInfer):
    r"""
    3D max pooling operation.

    Applies a 3D max pooling over an input Tensor which can be regarded as a composition of 3D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})`, MaxPool outputs
    regional maximum in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            is an int number that represents depth, height and width of the kernel, or a tuple
            of three int numbers that represent depth, height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of "pad" will
              be padded to the input Tensor borders. "pad" must be greater than or equal to 0.

        pad_list (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings
            of head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six
            integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
            pad[3], pad[4] and pad[5] correspondingly.
        ceil_mode (bool): Whether to use ceil instead of floor to calculate output shape. Only effective in "pad" mode.
            When "pad_mode" is "pad" and "ceil_mode" is "None", "ceil_mode" will be set as "False". Default: None.
        data_format (str) : The optional value for data format. Currently only support 'NCDHW'. Default: 'NCDHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
          Data type must be float16 or float32.

    Outputs:
        Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`. Has the data type with `x`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither an int not a tuple.
        TypeError: If `pad_mode` or `data_format` is not a string.
        ValueError: If numbers in `kernel_size` or `strides` are not positive.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad_mode` is 'same' or 'valid', 'ceil_mode' is not None.
        ValueError: If `kernel_size` or `strides` is a tuple whose length is not equal to 3.
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), mindspore.float32)
        >>> max_pool3d = ops.MaxPool3D(kernel_size=2, strides=1, pad_mode="valid")
        >>> output = max_pool3d(x)
        >>> print(output)
        [[[[[10. 11.]]]
          [[[22. 23.]]]]]
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="VALID", pad_list=0, ceil_mode=None, data_format="NCDHW"):
        """Initialize MaxPool3D."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])
        validator.check_value_type('kernel_size', kernel_size, [int, tuple], self.name)
        validator.check_value_type('strides', strides, [int, tuple], self.name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME', 'PAD'], 'pad_mode', self.name)
        if pad_mode.upper() == "PAD":
            self.pad_mode = "CALCULATED"
        self.add_prim_attr("pad_mode", self.pad_mode)
        self.data_format = validator.check_string(data_format, ['NCDHW'], 'data_format', self.name)
        self.kernel_size = _check_3d_int_or_tuple("kernel_size", kernel_size, self.name,
                                                  allow_five=False, ret_five=True)
        self.add_prim_attr("kernel_size", self.kernel_size)
        self.strides = _check_3d_int_or_tuple("strides", strides, self.name, allow_five=False, ret_five=True)
        self.add_prim_attr("strides", self.strides)
        if ceil_mode is None:
            self.ceil_mode = not self.pad_mode == "CALCULATED"
        else:
            self.ceil_mode = validator.check_value_type('ceil_mode', ceil_mode, [bool], self.name)
            if self.pad_mode != "CALCULATED":
                raise ValueError("When pad_mode is same or valid, ceil_mode only support 'None'.")
        self.add_prim_attr("ceil_mode", int(self.ceil_mode))

        validator.check_value_type('pad_list', pad_list, (int, tuple), self.name)
        self.pad_list = pad_list
        if isinstance(self.pad_list, int):
            self.pad_list = (self.pad_list,) * 6
        if len(self.pad_list) == 3:
            self.pad_list = (pad_list[0], pad_list[0], pad_list[1], pad_list[1], pad_list[2], pad_list[2])
        if len(self.pad_list) != 3 and len(self.pad_list) != 6:
            raise ValueError(f"For '{self.name}', attr 'pad_list' should be an positive int number or a tuple of "
                             f"three or six positive int numbers, but got {len(self.pad_list)} numbers.")
        if self.pad_mode != 'CALCULATED' and self.pad_list != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad_list' must be zero when 'pad_mode' is not \"CALCULATED\", "
                             f"but got 'pad_list' is {self.pad_list} and 'pad_mode' is {pad_mode}.")
        if self.pad_mode == 'CALCULATED':
            for item in self.pad_list:
                validator.check_non_negative_int(item, 'pad_list item', self.name)
        self.add_prim_attr("pad_list", self.pad_list)

    def infer_shape(self, x_shape):
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        batch, channel, input_d, input_h, input_w = x_shape
        self.add_prim_attr("x_shape", x_shape)
        _, _, kernel_d, kernel_h, kernel_w = self.kernel_size
        _, _, stride_d, stride_h, stride_w = self.strides

        if self.pad_mode == "VALID":
            out_d = math.ceil((input_d - (kernel_d - 1)) / stride_d)
            out_h = math.ceil((input_h - (kernel_h - 1)) / stride_h)
            out_w = math.ceil((input_w - (kernel_w - 1)) / stride_w)
        elif self.pad_mode == "SAME":
            out_d = math.ceil(input_d / stride_d)
            out_h = math.ceil(input_h / stride_h)
            out_w = math.ceil(input_w / stride_w)
        else:
            out_d = ((input_d + self.pad_list[0] + self.pad_list[1] -
                      (kernel_d - 1) - 1) / stride_d) + 1
            out_h = ((input_h + self.pad_list[2] + self.pad_list[3] -
                      (kernel_h - 1) - 1) / stride_h) + 1
            out_w = ((input_w + self.pad_list[4] + self.pad_list[5] -
                      (kernel_w - 1) - 1) / stride_w) + 1
            if self.ceil_mode:
                out_d = math.ceil(out_d)
                out_h = math.ceil(out_h)
                out_w = math.ceil(out_w)
            else:
                out_d = math.floor(out_d)
                out_h = math.floor(out_h)
                out_w = math.floor(out_w)
        out_shape = [batch, channel, out_d, out_h, out_w]

        _check_shape('output', out_shape, self.name)
        return out_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, [mstype.float16, mstype.float32], self.name)
        return x_dtype


class AvgPool(_Pool):
    r"""
    Average pooling operation.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.
    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, AvgPool outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    .. warning::
        - Only single input and single output are supported.
        - Global pooling is supported.
        - The height of "kernel_size" and the weight of "kernel_size" are positive integers within the range [1, 255].
          ksize_H * ksize_W < 256.
        - Due to instruction restrictions, the values of "strides_h" and "strides_w" are
          positive integers within the range [1, 63].

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents height and width of the kernel, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The format of input and output data. It should be 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.avgpool_op = ops.AvgPool(pad_mode="VALID", kernel_size=2, strides=1)
        ...
        ...     def construct(self, x):
        ...         result = self.avgpool_op(x)
        ...         return result
        ...
        >>> x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float32)
        >>> net = Net()
        >>> output = net(x)
        >>> print(output)
        [[[[ 2.5   3.5   4.5]
           [ 6.5   7.5   8.5]]
          [[14.5  15.5  16.5]
           [18.5  19.5  20.5]]
          [[26.5  27.5  28.5]
           [30.5  31.5  32.5]]]]
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", data_format="NCHW"):
        """Initialize AvgPool."""
        super(AvgPool, self).__init__(kernel_size, strides, pad_mode, data_format)


class Conv2DBackpropInput(Primitive):
    r"""
    Computes the gradients of convolution with respect to the input.

    Args:
        out_channel (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both the height and the width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        pad_mode (str): Modes to fill padding. It could be "valid", "same", or "pad". Default: "valid".
        pad (Union[int, tuple[int]]): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four integers, the
                    padding of top, bottom, left and right equal to pad[0], pad[1], pad[2], and pad[3] correspondingly.
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        stride (Union[int. tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        dilation (Union[int. tuple[int]]): Specifies the dilation rate to be used for the dilated convolution.
            Default: 1.
        group (int): Splits input into groups. Default: 1.
        data_format (str): The format of input and output data. It should be 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **dout** (Tensor) - The gradients write respect to the output of the convolution. The shape conforms
          to the default data_format :math:`(N, C_{out}, H_{out}, W_{out})`.
        - **weight** (Tensor) - Set size of kernel is :math:`(\text{ks_w}, \text{ks_h})`, where :math:`\text{ks_w}`
          and :math:`\text{ks_h}` are the height and width of the convolution kernel, then the shape is
          :math:`(C_{out}, C_{in}, \text{ks_w}, \text{ks_h})`.
        - **input_size** (Tensor) - A tuple describes the shape of the input which conforms to the format
          :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the gradients with respect to the input of convolution. It has the same shape as the input.

    Raises:
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int nor a tuple.
        TypeError: If `out_channel` or `group` is not an int.
        ValueError: If `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` it not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
        >>> weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
        >>> input_x = Tensor(np.ones([10, 32, 32, 32]))
        >>> conv2d_backprop_input = ops.Conv2DBackpropInput(out_channel=32, kernel_size=3)
        >>> output = conv2d_backprop_input(dout, weight, ops.shape(input_x))
        >>> print(output.shape)
        (10, 32, 32, 32)
    """
    __mindspore_signature__ = (
        sig.make_sig('out_backprop', dtype=sig.sig_dtype.T),
        sig.make_sig('filter', dtype=sig.sig_dtype.T1),
        sig.make_sig('input_sizes', dtype=sig.sig_dtype.T2)
    )

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 pad_mode="valid",
                 pad=0,
                 pad_list=None,
                 mode=1,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCHW"):
        """Initialize Conv2DBackpropInput"""
        self.init_prim_io_names(inputs=['out_backprop', 'filter', 'input_sizes'], outputs=['output'])
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.kernel_size = _check_positive_int_or_tuple('kernel_size', kernel_size, self.name)
        self.add_prim_attr('kernel_size', self.kernel_size)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the format is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)
        self.stride = _check_positive_int_or_tuple('stride', stride, self.name, allow_four=True, ret_four=True)
        self.stride = _update_attr_by_format(self.stride, self.format)
        self.add_prim_attr('stride', self.stride)
        self.dilation = _check_positive_int_or_tuple('dilation', dilation, self.name, allow_four=True, ret_four=True)
        self.dilation = _update_attr_by_format(self.dilation, self.format)
        self.add_prim_attr('dilation', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 4
        else:
            validator.check_equal_int(len(pad), 4, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.padding = pad
        self.pad_mode = validator.check_string(pad_mode, ['valid', 'same', 'pad'], 'pad_mode', self.name)
        if pad_mode != 'pad' and pad != (0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be zero when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {pad} and 'pad_mode' is {pad_mode}.")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)

        pad_mode = pad_mode.upper()
        self.add_prim_attr('pad_mode', pad_mode)
        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.group = validator.check_positive_int(group, 'group', self.name)
        self.add_prim_attr('groups', self.group)
        if pad_list:
            for x in pad_list:
                validator.check_non_negative_int(x, 'element of pad_list', self.name)
            self.pad_list = pad_list


class Conv2DTranspose(Conv2DBackpropInput):
    """
    Compute a 2D transposed convolution, which is also known as a deconvolution
    (although it is not an actual deconvolution).

    Args:
        out_channel (int): The dimensionality of the output space.
        kernel_size (Union[int, tuple[int]]): The size of the convolution window.
        pad_mode (str): Modes to fill padding. It could be "valid", "same", or "pad". Default: "valid".
        pad (Union[int, tuple[int]]): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four integers, the
                    padding of top, bottom, left and right equal to pad[0], pad[1], pad[2], and pad[3] correspondingly.
        mode (int): Modes for different convolutions. 0 Math convolutiuon, 1 cross-correlation convolution ,
                       2 deconvolution, 3 depthwise convolution. Default: 1.
        stride (Union[int. tuple[int]]): The stride to be applied to the convolution filter. Default: 1.
        dilation (Union[int. tuple[int]]): Specifies the dilation rate to be used for the dilated convolution.
            Default: 1.
        group (int): Splits input into groups. Default: 1.
        data_format (str) - The format of input and output data. It should be 'NHWC' or 'NCHW'，\
            default is 'NCHW'.

    Inputs:
        - **dout** (Tensor) - the gradients w.r.t the output of the convolution. The shape conforms to the default
          data_format :math:`(N, C_{out}, H_{out}, W_{out})`.
        - **weight** (Tensor) - Set size of kernel is :math:`(K_1, K_2)`, then the shape is
          :math:`(C_{out}, C_{in}, K_1, K_2)`.
        - **input_size** (Tensor) - A tuple describes the shape of the input which conforms to the format
          :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the gradients w.r.t the input of convolution. It has the same shape as the input.

    Raises:
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int nor a tuple.
        TypeError: If `out_channel` or `group` is not an int.
        ValueError: If `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `padding` is a tuple whose length is not equal to 4.
        ValueError: If `pad_mode` it not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0).
        ValueError: If `data_format` is neither 'NCHW' not 'NHWC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
        >>> weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
        >>> x = Tensor(np.ones([10, 32, 32, 32]))
        >>> conv2d_transpose_input = ops.Conv2DTranspose(out_channel=32, kernel_size=3)
        >>> output = conv2d_transpose_input(dout, weight, ops.shape(x))
        >>> print(output.shape)
        (10, 32, 32, 32)
    """

    @prim_attr_register
    def __init__(self, out_channel, kernel_size, pad_mode="valid", pad=0,
                 pad_list=None, mode=1, stride=1, dilation=1, group=1, data_format="NCHW"):
        """Initialize Conv2DTranspose."""
        super(Conv2DTranspose, self).__init__(out_channel, kernel_size, pad_mode, pad,
                                              pad_list, mode, stride, dilation, group, data_format)


class BiasAdd(Primitive):
    r"""
    Returns sum of input and bias tensor.

    Adds the 1-D bias tensor to the input tensor, and broadcasts the shape on all axis
    except for the channel axis.

    Args:
        data_format (str): The format of input and output data. It should be 'NHWC', 'NCHW' or 'NCDHW'.
            Default is 'NCHW'.

    Inputs:
        - **input_x** (Tensor) - The input tensor. The shape can be 2-5 dimensions.
          The data type should be float16 or float32.
        - **bias** (Tensor) - The bias tensor, with shape :math:`(C)`. The shape of
          `bias` must be the same as `input_x`'s channel dimension. The data type should be float16 or float32.

    Outputs:
        Tensor, with the same shape and data type as `input_x`.

    Raises:
        TypeError: If `data_format` is not a str.
        TypeError: If `input_x` or `bias` is not a Tensor.
        TypeError: If dtype of `input_x` or `bias` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.arange(6).reshape((2, 3)), mindspore.float32)
        >>> bias = Tensor(np.random.random(3).reshape((3,)), mindspore.float32)
        >>> bias_add = ops.BiasAdd()
        >>> output = bias_add(input_x, bias)
        >>> print(output.shape)
        (2, 3)
    """

    @prim_attr_register
    def __init__(self, data_format="NCHW"):
        """Initialize BiasAdd."""
        self.init_prim_io_names(inputs=['x', 'b'], outputs=['output'])
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC', 'NCDHW'], 'format', self.name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.name}', the \"NHWC\" format only support in GPU target, "
                             f"but got the format is {self.format} and "
                             f"the platform is {context.get_context('device_target')}.")
        self.add_prim_attr('data_format', self.format)


class TopK(PrimitiveWithInfer):
    """
    Finds values and indices of the `k` largest entries along the last dimension.

    .. warning::
        - If sorted set to 'False', it will use aicpu operator, performance may be reduced.

    If the `input_x` is a one-dimensional Tensor, finds the `k` largest entries in the Tensor,
    and outputs its value and index as a Tensor. Therefore, values[`k`] is the `k` largest item in `input_x`,
    and its index is indices [`k`].

    For a multi-dimensional matrix,
    calculates the first `k` entries in each row (corresponding vector along the last dimension), therefore:

    .. math::

        values.shape = indices.shape = input.shape[:-1] + [k].

    If the two compared elements are the same, the one with the smaller index value is returned first.

    Args:
        sorted (bool): If true, the obtained elements will
            be sorted by the values in descending order. Default: True.

    Inputs:
        - **input_x** (Tensor) - Input to be computed, data type must be float16, float32 or int32.
        - **k** (int) - The number of top elements to be computed along the last dimension, constant input is needed.

    Outputs:
        Tuple of 2 tensors, the values and the indices.

        - **values** (Tensor) - The `k` largest elements in each slice of the last dimensional.
        - **indices** (Tensor) - The indices of values within the last dimension of input.

    Raises:
        TypeError: If `sorted` is not a bool.
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `k` is not an int.
        TypeError: If dtype of `input_x` is not one of the following: float16, float32 or int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> topk = ops.TopK(sorted=True)
        >>> input_x = Tensor([1, 2, 3, 4, 5], mindspore.float16)
        >>> k = 3
        >>> values, indices = topk(input_x, k)
        >>> print((values, indices))
        (Tensor(shape=[3], dtype=Float16, value= [ 5.0000e+00,  4.0000e+00,  3.0000e+00]), Tensor(shape=[3],
          dtype=Int32, value= [4, 3, 2]))
    """

    @prim_attr_register
    def __init__(self, sorted=True):
        """Initialize TopK."""
        self.sorted = validator.check_value_type("sorted", sorted, [bool], self.name)
        self.add_prim_attr("sorted", self.sorted)
        self.init_prim_io_names(inputs=['input', 'k'],
                                outputs=['values', 'indices'])

    def __infer__(self, input_x, k):
        x_dtype = input_x['dtype']
        valid_dtypes = (mstype.int32, mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid('x', x_dtype, valid_dtypes, self.name)
        k_v = k['value']
        validator.check_value_type('k', k_v, (int,), self.name)
        x_shape = list(input_x['shape'])
        ndim = len(x_shape) - 1
        x_shape[ndim] = k_v
        return {'shape': (x_shape, x_shape),
                'dtype': (x_dtype, mstype.int32),
                'value': None}


class NLLLoss(PrimitiveWithInfer):
    r"""
    Gets the negative log likelihood loss between logits and labels.

    The nll loss with reduction=none can be described as:

    .. math::

        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot 1

    where :math:`x` is the logits, :math:`t` is the labels, :math:`w` is the weight,
    N is the batch size, :math:`c` belonging [0, C-1] is class index, where :math:`C` is the number of classes.

    If reduction is not 'none' (default 'mean'), then

    .. math::

        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean'; } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    Args:
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum', Default: "mean".

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`. Data type only support float32 or float16.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N,)`. Data type only support int32.
        - **weight** (Tensor) - The rescaling weight to each class, with shape :math:`(C,)` and data type only
          support float32 or float16.

    Outputs:
        Tuple of 2 tensors composed with `loss` and `total_weight`.

        - **loss** (Tensor) - When `reduction` is 'none' and `logits` is 2D tensor, the `loss` shape is :math:`(N,)`.
          Otherwise, the `loss` is a scalar. The data type is same with `input's`.
        - **total_weight** (Tensor) - The `total_weight` is a scalar. The data type is same with `weight's`.

    Raises:
        TypeError: If dtype of `logits` or `weight` is neither float16 nor float32, `labels` is not int32.
        ValueError: If `logits` is not a one or two dimension tensor, `labels` and `weight` not a one dimension tensor.
                    When `logits` is a two dimension tensor, the first dimension of `logits` is not equal to `labels`,
                    and second dimension of `logits` is not equal to `weight`.
                    When `logits` is a one dimension tensor, the dimensions of `logits`, `labels`
                    and `weight` should be equal to each other.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> logits = Tensor(np.array([[0.5488135, 0.71518934],
        ...                           [0.60276335, 0.5448832],
        ...                           [0.4236548, 0.6458941]]).astype(np.float32))
        >>> labels = Tensor(np.array([0, 0, 0]).astype(np.int32))
        >>> weight = Tensor(np.array([0.3834415, 0.79172504]).astype(np.float32))
        >>> nll_loss = ops.NLLLoss(reduction="mean")
        >>> loss, weight = nll_loss(logits, labels, weight)
        >>> print(loss)
        -0.52507716
        >>> print(weight)
        1.1503246
    """

    @prim_attr_register
    def __init__(self, reduction="mean"):
        """Initialize NLLLoss"""
        self.init_prim_io_names(inputs=['x', 'target', "weight"], outputs=['loss'])
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)
        self.add_prim_attr('reduction', self.reduction)

    def infer_shape(self, x_shape, t_shape, w_shape):
        validator.check_int(len(x_shape), [1, 2], Rel.IN, "x rank", self.name)
        validator.check_int(len(t_shape), 1, Rel.EQ, "target rank", self.name)
        validator.check_int(len(w_shape), 1, Rel.EQ, "weight rank", self.name)
        validator.check(f"input_shape[0]", x_shape[0], "target_shape", t_shape[0], Rel.EQ, self.name)
        if len(x_shape) == 1:
            validator.check(f"input_shape[0]", x_shape[0], "weight_shape", w_shape[0], Rel.EQ, self.name)
        else:
            validator.check(f"input_shape[1]", x_shape[1], "weight_shape", w_shape[0], Rel.EQ, self.name)
        if self.reduction == "none":
            return t_shape, ()
        return (), ()

    def infer_dtype(self, x_dtype, t_dtype, w_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("x_dtype", x_dtype, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("t_dtype", t_dtype, mstype.int32, self.name)
        validator.check_tensor_dtype_valid("w_dtype", w_dtype, valid_dtypes, self.name)
        return x_dtype, w_dtype


class SoftmaxCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Gets the softmax cross-entropy value between logits and labels with one-hot encoding.

    The updating formulas of SoftmaxCrossEntropyWithLogits algorithm are as follows,

    .. math::
        \begin{array}{ll} \\
            p_{ij} = softmax(X_{ij}) = \frac{\exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)} \\
            loss_{ij} = -\sum_j{Y_{ij} * ln(p_{ij})}
        \end{array}

    where :math:`X` represents `logits`.
    :math:`Y` represents `label`.
    :math:`loss` represents `output`.

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`. Data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N, C)`, has the same data type with `logits`.

    Outputs:
        Tuple of 2 tensors(loss, dlogits), the `loss` shape is :math:`(N,)`,
        and the `dlogits` with the same shape as `logits`.

    Raises:
        TypeError: If dtype of `logits` or `labels` is neither float16 nor float32.
        TypeError: If `logits` or `labels` is not a Tensor.
        ValueError: If shape of `logits` is not the same as `labels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor([[2, 4, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
        >>> labels = Tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], mindspore.float32)
        >>> softmax_cross = ops.SoftmaxCrossEntropyWithLogits()
        >>> loss, dlogits = softmax_cross(logits, labels)
        >>> print(loss)
        [0.5899297  0.52374405]
        >>> print(dlogits)
        [[ 0.02760027  0.20393994  0.01015357  0.20393994 -0.44563377]
         [ 0.08015892  0.02948882  0.08015892 -0.4077012   0.21789455]]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, logits_shape, labels_shape):
        validator.check("logits_shape", logits_shape, "labels_shape", labels_shape, Rel.EQ, self.name)
        loss_shape = [logits_shape[0]]
        dlogits_shape = logits_shape
        return loss_shape, dlogits_shape

    def infer_dtype(self, logits_type, labels_type):
        args = {"logits": logits_type, "labels": labels_type}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return logits_type, logits_type


class SparseSoftmaxCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Computes the softmax cross-entropy value between logits and sparse encoding labels.

    Sets input logits as `X`, input label as `Y`, output as `loss`. Then,

    .. math::
        \begin{array}{ll} \\
            p_{ij} = softmax(X_{ij}) = \frac{\exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)} \\
            loss_{ij} = \begin{cases} -ln(p_{ij}), &j = y_i \cr -ln(1 - p_{ij}), & j \neq y_i \end{cases} \\
            loss = \sum_{ij} loss_{ij}
        \end{array}

    Args:
        is_grad (bool): If true, this operation returns the computed gradient. Default: False.

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`. Data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N)`.
          Data type must be int32 or int64.

    Outputs:
        Tensor, if `is_grad` is False, the output tensor is the value of loss which is a scalar tensor;
        if `is_grad` is True, the output tensor is the gradient of input with the same shape as `logits`.

    Raises:
        TypeError: If `is_grad` is not a bool.
        TypeError: If dtype of `logits` is neither float16 nor float32.
        TypeError: If dtype of `labels` is neither int32 nor int64.
        ValueError: If logits.shape[0] != labels.shape[0].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
        >>> labels = Tensor([0, 1], mindspore.int32)
        >>> sparse_softmax_cross = ops.SparseSoftmaxCrossEntropyWithLogits()
        >>> loss = sparse_softmax_cross(logits, labels)
        >>> print(loss)
        3.4878292
        >>> sparse_softmax_cross_grad = ops.SparseSoftmaxCrossEntropyWithLogits(is_grad=True)
        >>> loss_grad = sparse_softmax_cross_grad(logits, labels)
        >>> print(loss_grad)
        [[-0.48415753  0.04306427  0.00582811  0.11706084  0.3182043 ]
         [ 0.04007946 -0.4852556   0.04007946  0.2961494   0.10894729]]
    """

    @prim_attr_register
    def __init__(self, is_grad=False):
        """Initialize SparseSoftmaxCrossEntropyWithLogits."""
        validator.check_value_type('is_grad', is_grad, [bool], self.name)
        self.init_prim_io_names(inputs=['features', 'labels'], outputs=['output'])
        self.is_grad = is_grad
        self.add_prim_attr('sens', 1.0)

    def infer_shape(self, logits_shape, labels_shape):
        validator.check("logits_shape[0]", logits_shape[0], "labels_shape[0]", labels_shape[0], Rel.EQ, self.name)
        loss_shape = []
        if self.is_grad:
            return logits_shape
        return loss_shape

    def infer_dtype(self, logits_type, labels_type):
        validator.check_tensor_dtype_valid("logits", logits_type, (mstype.float16, mstype.float32),
                                           self.name)
        validator.check_tensor_dtype_valid("labels", labels_type, (mstype.int32, mstype.int64), self.name)
        return logits_type


class ApplyMomentum(PrimitiveWithInfer):
    """
    Optimizer that implements the Momentum algorithm.

    Refer to the paper `On the importance of initialization and momentum in deep
    learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_  for more details.

    Refer to :class:`mindspore.nn.Momentum` for more details about the formula and usage.

    Inputs of `variable`, `accumulation` and `gradient` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    Data type conversion of Parameter is not supported. RuntimeError exception will be thrown.

    Args:
        use_locking (bool): Whether to enable a lock to protect the variable and accumulation tensors
                            from being updated. Default: False.
        use_nesterov (bool): Enable Nesterov momentum. Default: False.
        gradient_scale (float): The scale of the gradient. Default: 1.0.

    Inputs:
        - **variable** (Parameter) - Weights to be updated. data type must be float.
        - **accumulation** (Parameter) - Accumulated gradient value by moment weight.
          Has the same data type with `variable`.
        - **learning_rate** (Union[Number, Tensor]) - The learning rate value, must be a float number or
          a scalar tensor with float data type.
        - **gradient** (Tensor) - Gradient, has the same data type as `variable`.
        - **momentum** (Union[Number, Tensor]) - Momentum, must be a float number or
          a scalar tensor with float data type.

    Outputs:
        Tensor, parameters to be updated.

    Raises:
        TypeError: If the `use_locking` or `use_nesterov` is not a bool or `gradient_scale` is not a float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        Please refer to the usage in :class:`mindspore.nn.Momentum`.
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accumulation', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('learning_rate', dtype=sig.sig_dtype.T1),
        sig.make_sig('gradient', dtype=sig.sig_dtype.T),
        sig.make_sig('momentum', dtype=sig.sig_dtype.T2)
    )

    @prim_attr_register
    def __init__(self, use_nesterov=False, use_locking=False, gradient_scale=1.0):
        """Initialize ApplyMomentum."""
        self.use_nesterov = validator.check_bool(use_nesterov, "use_nesterov", self.name)
        self.use_locking = validator.check_bool(use_locking, "use_locking", self.name)
        validator.check_value_type('gradient_scale', gradient_scale, [float], self.name)
        self.init_prim_io_names(inputs=['variable', 'accumulation', 'learning_rate', 'gradient', 'momentum'],
                                outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, v_shape, a_shape, l_shape, g_shape, m_shape):
        return v_shape

    def infer_dtype(self, v_dtype, a_dtype, l_dtype, g_dtype, m_dtype):
        valid_dtypes = [mstype.float16, mstype.float32, mstype.float64]
        if v_dtype != mstype.type_refkey and a_dtype != mstype.type_refkey:
            validator.check_tensor_dtype_valid("v", v_dtype, valid_dtypes, self.name)
            validator.check_tensor_dtype_valid("a", a_dtype, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l_dtype": l_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"g_dtype": g_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"m_dtype": m_dtype}, valid_dtypes, self.name)
        return v_dtype


class SmoothL1Loss(PrimitiveWithInfer):
    r"""
    Computes smooth L1 loss, a robust L1 loss.

    SmoothL1Loss is a Loss similar to MSELoss but less sensitive to outliers as described in the
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ by Ross Girshick.

    Given two input :math:`x,\  y` of length :math:`N`, the unreduced SmoothL1Loss can be described
    as follows:

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 \text{beta}, & \text{otherwise. }
        \end{cases}

    Here :math:`\text{beta}` controls the point where the loss function changes from quadratic to linear.
    Its default value is 1.0. :math:`N` is the batch size. This function returns an
    unreduced loss Tensor.

    .. warning::
        This operator does not perform the "reduce" operation on the loss value.
        Call other reduce operators to perform "reduce" operation on the loss if required.

    Args:
        beta (float): A parameter used to control the point where the function will change from
            quadratic to linear. Default: 1.0.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
          additional dimensions. Data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth data, tensor of shape :math:`(N, *)`,
          same shape and dtype as the `logits`.

    Outputs:
        Tensor, loss float tensor, same shape and dtype as the `logits`.

    Raises:
        TypeError: If `beta` is not a float.
        TypeError: If dtype of `logits` or `labels` is neither float16 not float32.
        ValueError: If `beta` is less than or equal to 0.
        ValueError: If shape of `logits` is not the same as `labels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = ops.SmoothL1Loss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [0.  0.  0.5]
    """

    @prim_attr_register
    def __init__(self, beta=1.0):
        """Initialize SmoothL1Loss."""
        validator.check_value_type('beta', beta, [float], self.name)
        validator.check('beta', beta, '', 0, Rel.GT, self.name)
        self.init_prim_io_names(inputs=['prediction', 'target'], outputs=['output'])

    def infer_shape(self, prediction, target):
        validator.check('prediction shape', prediction, 'target shape', target, Rel.EQ, self.name)
        return prediction

    def infer_dtype(self, prediction, target):
        args = {"prediction": prediction, "target": target}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float16, mstype.float32), self.name)
        return prediction


class SoftMarginLoss(Primitive):
    r"""
    SoftMarginLoss operation.

    Creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and target tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    Args:
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **logits** (Tensor) - Predict data. Data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth data, with the same type and shape as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is "none", its shape is the same as `logits`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If `logits` or `labels` is not a Tensor.
        TypeError: If dtype of `logits` or `labels` is neither float16 nor float32.
        ValueError: If shape of `logits` is not the same as `labels`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> loss = ops.SoftMarginLoss()
        >>> logits = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32)
        >>> labels = Tensor(np.array([[-1, 1], [1, -1]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.6764238
    """

    @prim_attr_register
    def __init__(self, reduction="mean"):
        """Initialize SoftMarginLoss"""
        self.init_prim_io_names(inputs=['predict', 'label'], outputs=['loss'])
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)


class L2Loss(PrimitiveWithInfer):
    """
    Calculates half of the L2 norm of a tensor without using the `sqrt`.

    Set `input_x` as x and output as loss.

    .. math::
        loss = sum(x ** 2) / 2

    Inputs:
        - **input_x** (Tensor) - A input Tensor. Data type must be float16 or float32.

    Outputs:
        Tensor, has the same dtype as `input_x`. The output tensor is the value of loss which is a scalar tensor.

    Raises:
        TypeError: If `input_x` not a Tensor.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float16)
        >>> l2_loss = ops.L2Loss()
        >>> output = l2_loss(input_x)
        >>> print(output)
        7.0
    """

    @prim_attr_register
    def __init__(self):
        """Initialize L2Loss"""

    def infer_shape(self, input_x):
        loss_shape = []
        return loss_shape

    def infer_dtype(self, x_type):
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensor_dtype_valid('x_type', x_type, valid_dtypes, self.name)
        return x_type


class DataFormatDimMap(PrimitiveWithInfer):
    """
    Returns the dimension index in the destination data format given in the source data format.

    Args:
        src_format (str): An optional value for source data format. The format can be 'NHWC' and 'NCHW'.
            Default: 'NHWC'.
        dst_format (str): An optional value for destination data format. The format can be 'NHWC' and 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **input_x** (Tensor) - A Tensor with each element as a dimension index in source data format.
          The suggested values is in the range [-4, 4). Only supports int32.

    Outputs:
        Tensor, Return the dimension index in the given target data format,
        has the same data type and shape as the `input_x`.

    Raises:
        TypeError: If `src_format` or `dst_format` is not a str.
        TypeError: If `input_x` is not a Tensor whose dtype is not int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor([0, 1, 2, 3], mindspore.int32)
        >>> dfdm = ops.DataFormatDimMap()
        >>> output = dfdm(input_x)
        >>> print(output)
        [0 3 1 2]
    """

    @prim_attr_register
    def __init__(self, src_format='NHWC', dst_format='NCHW'):
        """Initialize DataFormatDimMap."""
        valid_values = ['NHWC', 'NCHW']
        self.src_format = validator.check_string(src_format, valid_values, "src_format", self.name)
        self.dst_format = validator.check_string(dst_format, valid_values, "dst_format", self.name)
        self.init_prim_io_names(inputs=['input_x'], outputs=['output'])

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        valid_dtypes = [mstype.int32]
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        return x_dtype


class RNNTLoss(PrimitiveWithInfer):
    """
    Computes the RNNTLoss and its gradient with respect to the softmax outputs.

    Args:
        blank_label (int): blank label. Default: 0.

    Inputs:
        - **acts** (Tensor) - Tensor of shape :math:`(B, T, U, V)`. Data type must be float16 or float32.
        - **labels** (Tensor) - Tensor of shape :math:`(B, U-1)`. Data type is int32.
        - **input_lengths** (Tensor) - Tensor of shape :math:`(B,)`. Data type is int32.
        - **label_lengths** (Tensor) - Tensor of shape :math:`(B,)`. Data type is int32.

    Outputs:
        - **costs** (Tensor) - Tensor of shape :math:`(B,)`. Data type is int32.
        - **grads** (Tensor) - Has the same shape and dtype as `acts`.

    Raises:
        TypeError: If `acts`, `labels`, `input_lengths` or `label_lengths` is not a Tensor.
        TypeError: If dtype of `acts` is neither float16 nor float32.
        TypeError: If dtype of `labels`, `input_lengths` or `label_lengths` is not int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> B, T, U, V = 1, 2, 3, 5
        >>> blank = 0
        >>> acts = np.random.random((B, T, U, V)).astype(np.float32)
        >>> labels = np.array([[1, 2]]).astype(np.int32)
        >>> input_length = np.array([T] * B).astype(np.int32)
        >>> label_length = np.array([len(l) for l in labels]).astype(np.int32)
        >>> rnnt_loss = ops.RNNTLoss(blank_label=0)
        >>> costs, grads = rnnt_loss(Tensor(acts), Tensor(labels), Tensor(input_length), Tensor(label_length))
        >>> print(costs.shape)
        (1,)
        >>> print(grads.shape)
        (1, 2, 3, 5)
    """

    @prim_attr_register
    def __init__(self, blank_label=0):
        """Initialize RNNTLoss."""
        validator.check_value_type('blank_label', blank_label, [int], self.name)
        self.init_prim_io_names(inputs=['acts', 'labels', 'input_length', 'label_length'],
                                outputs=['costs', 'grads'])

    def infer_shape(self, acts_shape, labels_shape, input_length_shape, label_length_shape):
        validator.check_equal_int(len(acts_shape), 4, 'acts_rank', self.name)
        validator.check_equal_int(len(labels_shape), 2, 'labels_rank', self.name)
        validator.check_equal_int(len(input_length_shape), 1, 'input_length_rank', self.name)
        validator.check_equal_int(len(label_length_shape), 1, 'label_length_rank', self.name)
        validator.check('labels shape[0]', labels_shape[0], 'acts shape[0]', acts_shape[0], Rel.EQ, self.name)
        validator.check('labels shape[1]', labels_shape[1], 'acts shape[2]-1', acts_shape[2] - 1, Rel.EQ, self.name)
        validator.check('input_length size', input_length_shape[0], 'acts shape[0]', acts_shape[0], Rel.EQ, self.name)
        validator.check('label_length size', label_length_shape[0], 'acts shape[0]', acts_shape[0], Rel.EQ, self.name)
        costs_shape = (acts_shape[0],)
        return costs_shape, acts_shape

    def infer_dtype(self, acts_type, labels_type, input_length_type, label_length_type):
        validator.check_tensor_dtype_valid("acts_type", acts_type, [mstype.float32, mstype.float16], self.name)
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.int32,), prim_name=self.name),
                  ("labels", "input_length", "label_length"),
                  (labels_type, input_length_type, label_length_type)))
        return acts_type, acts_type


class SGD(PrimitiveWithCheck):
    """
    Computes the stochastic gradient descent. Momentum is optional.

    Nesterov momentum is based on the formula from paper `On the importance of
    initialization and momentum in deep learning <http://proceedings.mlr.press/v28/sutskever13.html>`_.

    Note:
        For more details, please refer to :class:`nn.SGD`.

    Args:
        dampening (float): The dampening for momentum. Default: 0.0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        nesterov (bool): Enable Nesterov momentum. Default: False.

    Inputs:
        - **parameters** (Tensor) - Parameters to be updated. With float16 or float32 data type.
        - **gradient** (Tensor) - Gradient, with float16 or float32 data type.
        - **learning_rate** (Tensor) - Learning rate, a scalar tensor with float16 or float32 data type.
          e.g. Tensor(0.1, mindspore.float32)
        - **accum** (Tensor) - Accum(velocity) to be updated. With float16 or float32 data type.
        - **momentum** (Tensor) - Momentum, a scalar tensor with float16 or float32 data type.
          e.g. Tensor(0.1, mindspore.float32).
        - **stat** (Tensor) - States to be updated with the same shape as gradient, with float16 or float32 data type.

    Outputs:
        Tensor, parameters to be updated.

    Raises:
        TypeError: If `dampening` or `weight_decay` is not a float.
        TypeError: If `nesterov` is not a bool.
        TypeError: If `parameters`, `gradient`, `learning_rate`, `accum`, `momentum` or `stat` is not a Tensor.
        TypeError: If dtype of `parameters`, `gradient`, `learning_rate`, `accum`, `momentum` or `stat` is neither
                   float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> sgd = ops.SGD()
        >>> parameters = Tensor(np.array([2, -0.5, 1.7, 4]), mindspore.float32)
        >>> gradient = Tensor(np.array([1, -1, 0.5, 2]), mindspore.float32)
        >>> learning_rate = Tensor(0.01, mindspore.float32)
        >>> accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mindspore.float32)
        >>> momentum = Tensor(0.1, mindspore.float32)
        >>> stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mindspore.float32)
        >>> output = sgd(parameters, gradient, learning_rate, accum, momentum, stat)
        >>> print(output)
        (Tensor(shape=[4], dtype=Float32,
         value= [ 1.98989999e+00, -4.90300000e-01,  1.69520009e+00,  3.98009992e+00]),)
    """

    @prim_attr_register
    def __init__(self, dampening=0.0, weight_decay=0.0, nesterov=False):
        """Initialize SGD."""
        validator.check_value_type("nesterov", nesterov, [bool], self.name)
        if nesterov and dampening != 0:
            raise ValueError(f"For '{self.name}', the 'dampening' must be 0 when 'nesterov' is True, "
                             f"but got 'dampening' is {dampening} and 'nesterov' is {nesterov}.")
        self.init_prim_io_names(inputs=['parameters', 'gradient', 'learning_rate', 'accum', 'momentum', 'stat'],
                                outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def check_shape(self, parameters_shape, gradient_shape, learning_rate_shape,
                    accum_shape, momentum_shape, stat_shape):
        validator.check_positive_int(len(parameters_shape), "parameters rank", self.name)
        validator.check_int(len(gradient_shape), 0, Rel.GE, f'gradient rank', self.name)
        validator.check_int(len(learning_rate_shape), 0, Rel.GE, f'learning rate rank', self.name)
        validator.check_positive_int(len(accum_shape), "accumulation rank", self.name)
        validator.check_int(len(momentum_shape), 0, Rel.GE, f'momentum rank', self.name)
        validator.check_int(len(stat_shape), 0, Rel.GE, f'stat rank', self.name)
        validator.check("gradient shape", gradient_shape, "stat shape", stat_shape, Rel.EQ, self.name)

    def check_dtype(self, parameters_dtype, gradient_dtype, learning_rate_dtype,
                    accum_dtype, momentum_dtype, stat_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("parameters", "gradient", "learning_rate", "accum", "momentum", "stat"),
                  (parameters_dtype, gradient_dtype, learning_rate_dtype, accum_dtype, momentum_dtype, stat_dtype)))


class ApplyRMSProp(PrimitiveWithInfer):
    r"""
    Optimizer that implements the Root Mean Square prop(RMSProp) algorithm.
    Please refer to the usage in source code of :class:`nn.RMSProp`.

    The updating formulas of ApplyRMSProp algorithm are as follows,

    .. math::
        \begin{array}{ll} \\
            s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2 \\
            m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} + \epsilon}} \nabla Q_{i}(w) \\
            w = w - m_{t+1}
        \end{array}

    where :math:`w` represents `var`, which will be updated.
    :math:`s_{t+1}` represents `mean_square`, :math:`s_{t}` is the last momentent of :math:`s_{t+1}`,
    :math:`m_{t+1}` represents `moment`, :math:`m_{t}` is the last momentent of :math:`m_{t+1}`.
    :math:`\rho` represents `decay`. :math:`\beta` is the momentum term, represents `momentum`.
    :math:`\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
    :math:`\eta` represents `learning_rate`. :math:`\nabla Q_{i}(w)` represents `grad`.

    .. warning::
        Note that in dense implementation of this algorithm, "mean_square" and "moment" will update even if "grad" is 0,
        but in this sparse implementation, "mean_square" and "moment" will not update
        in iterations during which "grad" is 0.

    Args:
        use_locking (bool): Whether to enable a lock to protect the variable and accumlation tensors
                            from being updated. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be update.
        - **mean_square** (Tensor) - Mean square gradients, must have the same type as `var`.
        - **moment** (Tensor) - Delta of `var`, must have the same type as `var`.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate. Must be a float number or
          a scalar tensor with float16 or float32 data type.
        - **grad** (Tensor) - Gradient, must have the same type as `var`.
        - **decay** (float) - Decay rate. Only constant value is allowed.
        - **momentum** (float) - Momentum. Only constant value is allowed.
        - **epsilon** (float) - Ridge term. Only constant value is allowed.

    Outputs:
        Tensor, parameters to be update.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `var`, `mean_square`, `moment` or `decay` is not a Tensor.
        TypeError: If `learning_rate` is neither a Number nor a Tensor.
        TypeError: If dtype of `decay`, `momentum` or `epsilon` is not float.
        TypeError: If dtype of `learning_rate` is neither float16 nor float32.
        ValueError: If `decay`, `momentum` or `epsilon` is not a constant value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_rms_prop = ops.ApplyRMSProp()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...
        ...     def construct(self, mean_square, moment, grad, decay, momentum, epsilon, lr):
        ...         out = self.apply_rms_prop(self.var, mean_square, moment, lr, grad, decay, momentum, epsilon)
        ...         return out
        ...
        >>> net = Net()
        >>> mean_square = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> moment = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> grad = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> output = net(mean_square, moment, grad, 0.0, 1e-10, 0.001, 0.01)
        >>> print(net.var.asnumpy())
        [[0.990005  0.990005]
         [0.990005  0.990005]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ApplyRMSProp."""
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['var', 'mean_square', 'moment', 'learning_rate', 'grad',
                                        'rho', 'momentum', 'epsilon'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, mean_square_shape, moment_shape, learning_rate_shape, grad_shape, decay_shape,
                    momentum_shape, epsilon_shape):
        validator.check("var_shape", var_shape, "mean_square_shape", mean_square_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "moment_shape", moment_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return var_shape

    def infer_dtype(self, var_dtype, mean_square_dtype, moment_dtype, learning_rate_dtype, grad_dtype, decay_dtype,
                    momentum_dtype, epsilon_dtype):
        args = {"var": var_dtype, "mean_square": mean_square_dtype, "moment": moment_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        valid_dtypes = [mstype.float16, mstype.float32]
        args_decay = {"decay": decay_dtype, 'momentum': momentum_dtype, "epsilon": epsilon_dtype}
        validator.check_types_same_and_valid(args_decay, valid_dtypes, self.name)
        args_lr = {"learning_rate": learning_rate_dtype, "decay": decay_dtype}
        validator.check_scalar_or_tensor_types_same(args_lr, valid_dtypes, self.name, allow_mix=True)
        return var_dtype

    def infer_value(self, var, mean_square, moment, learning_rate, grad, decay, momentum, epsilon):
        if decay is None or momentum is None or epsilon is None:
            raise ValueError(f"For '{self.name}', 'decay', 'momentum' and 'epsilon' can not be None, "
                             f"but got 'decay': {decay}, 'momentum': {momentum} and 'epsilon':{epsilon}.")


class ApplyCenteredRMSProp(PrimitiveWithInfer):
    r"""
    Optimizer that implements the centered RMSProp algorithm.
    Please refer to the usage in source code of :class:`nn.RMSProp`.

    The updating formulas of ApplyCenteredRMSProp algorithm are as follows,

    .. math::
        \begin{array}{ll} \\
            g_{t+1} = \rho g_{t} + (1 - \rho)\nabla Q_{i}(w) \\
            s_{t+1} = \rho s_{t} + (1 - \rho)(\nabla Q_{i}(w))^2 \\
            m_{t+1} = \beta m_{t} + \frac{\eta} {\sqrt{s_{t+1} - g_{t+1}^2 + \epsilon}} \nabla Q_{i}(w) \\
            w = w - m_{t+1}
        \end{array}

    where :math:`w` represents `var`, which will be updated.
    :math:`g_{t+1}` represents `mean_gradient`, :math:`g_{t}` is the last momentent of :math:`g_{t+1}`.
    :math:`s_{t+1}` represents `mean_square`, :math:`s_{t}` is the last momentent of :math:`s_{t+1}`,
    :math:`m_{t+1}` represents `moment`, :math:`m_{t}` is the last momentent of :math:`m_{t+1}`.
    :math:`\rho` represents `decay`. :math:`\beta` is the momentum term, represents `momentum`.
    :math:`\epsilon` is a smoothing term to avoid division by zero, represents `epsilon`.
    :math:`\eta` represents `learning_rate`. :math:`\nabla Q_{i}(w)` represents `grad`.

    Note:
        The difference between `ApplyCenteredRMSProp` and `ApplyRMSProp` is that the fromer
        uses the centered RMSProp algorithm, and the centered RRMSProp algorithm uses an estimate of the centered second
        moment(i.e., the variance) for normalization, as opposed to regular RMSProp, which uses the (uncentered)
        second moment. This often helps with training, but is slightly more exapnsive interms of computation and memory.

    .. warning::
        In dense implementation of this algorithm, mean_gradient, mean_square, and moment will update
        even if the grad is zero. But in this sparse implementation, mean_gradient, mean_square, and moment
        will not update in iterations during which the grad is zero.

    Args:
        use_locking (bool): Whether to enable a lock to protect the variable and accumlation tensors
                            from being updated. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be update.
        - **mean_gradient** (Tensor) - Mean gradients, must have the same type as `var`.
        - **mean_square** (Tensor) - Mean square gradients, must have the same type as `var`.
        - **moment** (Tensor) - Delta of `var`, must have the same type as `var`.
        - **grad** (Tensor) - Gradient, must have the same type as `var`.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate. Must be a float number or
          a scalar tensor with float16 or float32 data type.
        - **decay** (float) - Decay rate.
        - **momentum** (float) - Momentum.
        - **epsilon** (float) - Ridge term.

    Outputs:
        Tensor, parameters to be update.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If `var`, `mean_gradient`, `mean_square`, `moment` or `grad` is not a Tensor.
        TypeError: If `learing_rate` is neither a Number nor a Tensor.
        TypeError: If dtype of `learing_rate` is neither float16 nor float32.
        TypeError: If `decay`, `momentum` or `epsilon` is not a float.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_centerd_rms_prop = ops.ApplyCenteredRMSProp()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...
        ...     def construct(self, mean_grad, mean_square, moment, grad, decay, momentum, epsilon, lr):
        ...         out = self.apply_centerd_rms_prop(self.var, mean_grad, mean_square, moment, grad,
        ...                                           lr, decay, momentum, epsilon)
        ...         return out
        ...
        >>> net = Net()
        >>> mean_grad = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> mean_square = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> moment = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> grad = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> output = net(mean_grad, mean_square, moment, grad, 0.0, 1e-10, 0.001, 0.01)
        >>> print(net.var.asnumpy())
        [[0.68377227  0.68377227]
         [0.68377227  0.68377227]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ApplyCenteredRMSProp."""
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, mean_gradient_shape, mean_square_shape, moment_shape, grad_shape,
                    learning_rate_shape, decay_shape, momentum_shape, epsilon_shape):
        validator.check("var_shape", var_shape, "mean_gradient_shape", mean_gradient_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "mean_square_shape", mean_square_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "moment_shape", moment_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return var_shape

    def infer_dtype(self, var_dtype, mean_gradient_dtype, mean_square_dtype, moment_dtype, grad_dtype,
                    learning_rate_dtype, rho_dtype, momentum_dtype, epsilon_dtype):
        args = {"var": var_dtype, "mean_gradient": mean_gradient_dtype,
                "mean_square": mean_square_dtype, "moment": moment_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        valid_dtypes = [mstype.float16, mstype.float32]
        args_rho = {"rho": rho_dtype, 'momentum': momentum_dtype, "epsilon": epsilon_dtype}
        validator.check_types_same_and_valid(args_rho, valid_dtypes, self.name)
        args_lr = {"learning_rate": learning_rate_dtype, "rho": rho_dtype}
        validator.check_scalar_or_tensor_types_same(args_lr, valid_dtypes, self.name, allow_mix=True)
        return var_dtype


class LayerNorm(Primitive):
    r"""
    Applies the Layer Normalization to the input tensor.

    This operator will normalize the input tensor on given axis. LayerNorm is described in the paper
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`_.

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon.

    Args:
        begin_norm_axis (int): The begin axis of the `input_x` to apply LayerNorm,
            the value must be in [-1, rank(input)). Default: 1.
        begin_params_axis (int): The begin axis of the parameter input (`gamma`, `beta`) to
            apply LayerNorm, the value must be in [-1, rank(input)). Default: 1.
        epsilon (float): A value added to the denominator for numerical stability. Default: 1e-7.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
          The input of LayerNorm.
        - **gamma** (Tensor) - Tensor of shape :math:`(P_0, \ldots, P_\text{begin_params_axis})`.
          The learnable parameter `gamma` as the scale on norm.
        - **beta** (Tensor) - Tensor of shape :math:`(P_0, \ldots, P_\text{begin_params_axis})`.
          The learnable parameter `beta` as the scale on norm.

    Outputs:
        tuple[Tensor], tuple of 3 tensors, the normalized input and the updated parameters.

        - **output_x** (Tensor) - The normalized input, has the same type and shape as the `input_x`.
          The shape is :math:`(N, C)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.

    Raises:
        TypeError: If `begin_norm_axis` or `begin_params_axis` is not an int.
        TypeError: If `epsilon` is not a float.
        TypeError: If `input_x`, `gamma` or `beta` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
        >>> gamma = Tensor(np.ones([3]), mindspore.float32)
        >>> beta = Tensor(np.ones([3]), mindspore.float32)
        >>> layer_norm = ops.LayerNorm()
        >>> output, mean, variance = layer_norm(input_x, gamma, beta)
        >>> print(output)
        [[-0.2247448  1.         2.2247448]
         [-0.2247448  1.         2.2247448]]
        >>> print(mean)
        [[2.]
         [2.]]
        >>> print(variance)
        [[0.6666667]
         [0.6666667]]
    """

    @prim_attr_register
    def __init__(self, begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7):
        """Initialize LayerNorm."""
        validator.check_value_type('begin_norm_axis', begin_norm_axis, [int], self.name)
        validator.check_value_type('begin_params_axis', begin_params_axis, [int], self.name)
        validator.check_value_type('epsilon', epsilon, [float], self.name)


class L2Normalize(PrimitiveWithInfer):
    r"""
    L2 Normalization Operator.

    This operator will normalize the input using the given axis. The function is shown as follows:

    .. math::
        \text{output} = \frac{x}{\sqrt{\text{max}(\text{sum} (\text{x}^2), \epsilon)}},

    where :math:`\epsilon` is epsilon.

    Args:
        axis (Union[list(int), tuple(int), int]): The starting axis for the input to apply the L2 Normalization.
                                                  Default: 0.
        epsilon (float): A small value added for numerical stability. Default: 1e-4.

    Inputs:
        - **x** (Tensor) - Input to compute the normalization. Tensor of shape :math:`(N, \ldots)`.
          Data type must be float16 or float32.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `axis` is not one of the following: list, tuple or int.
        TypeError: If `epsilon` is not a float.
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> l2_normalize = ops.L2Normalize()
        >>> x = Tensor(np.random.randint(-256, 256, (2, 3, 4)), mindspore.float32)
        >>> output = l2_normalize(x)
        >>> print(output.shape)
        (2, 3, 4)
    """

    @prim_attr_register
    def __init__(self, axis=0, epsilon=1e-4):
        """Initialize L2Normalize."""
        axis = [axis] if isinstance(axis, int) else axis
        validator.check_value_type('axis', axis, [list, tuple], self.name)
        validator.check_value_type('epsilon', epsilon, [int, float], self.name)
        self.add_prim_attr('axis', axis)
        self.init_attrs['axis'] = axis
        if len(axis) != 1:
            raise TypeError(f"For '{self.name}', the length of 'axis' must be 1, but got {len(axis)}, "
                            f"later will support multiple axis!")
        self.axis = axis

    def infer_shape(self, input_x):
        dim = len(input_x)
        validator.check_int_range(self.axis[0], -dim, dim, Rel.INC_LEFT, 'axis value', self.name)
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, [mstype.float16, mstype.float32], self.name)
        return input_x


class DropoutGenMask(Primitive):
    """
    Generates the mask value for the input shape.

    Dropout means that neural network units are temporarily dropped from the network according to a certain probability
    during the deep learning network training. Generally, The effect of Dropout is the same as that of DropoutGenMask
    and DropoutDoMask. The DropoutGenMask generates a mask shape that is specified based on the input. Next,
    The DropoutDoMask is a mask generated using DropoutGenMask.
    The input tensor is randomly set to zero based on the probability p.


    Args:
        Seed0 (int): Seed0 value for random generating. Default: 0.
        Seed1 (int): Seed1 value for random generating. Default: 0.

    Inputs:
        - **shape** (tuple[int]) - The shape of target mask.
        - **keep_prob** (Tensor) - The keep rate, greater than 0 and less equal than 1, e.g. keep_prob = 0.9,
          means dropping out 10% of input units.

    Outputs:
        Tensor, the value of generated mask for Inputs `shape`.

    Raises:
        TypeError: If neither `seed0` nor `seed1` is an int.
        TypeError: If `shape` is not a tuple.
        TypeError: If `keep_prob` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> dropout_gen_mask = ops.DropoutGenMask()
        >>> shape = (2, 4, 5)
        >>> keep_prob = Tensor(0.5, mindspore.float32)
        >>> output = dropout_gen_mask(shape, keep_prob)
        >>> print(output.shape)
        (16,)
    """

    @prim_attr_register
    def __init__(self, Seed0=0, Seed1=0):
        """Initialize DropoutGenMask."""
        self.init_prim_io_names(inputs=['shape', 'keep_prob'], outputs=['output'])
        validator.check_value_type("Seed0", Seed0, [int], self.name)
        validator.check_value_type("Seed1", Seed1, [int], self.name)
        self.add_prim_attr("_random_effect", True)


class DropoutDoMask(Primitive):
    r"""
    Applies dropout mask on the input tensor.

    Take the mask output of DropoutGenMask as input, and apply dropout on the input.

    Dropout means that neural network units are temporarily dropped from the network according to a certain probability
    during the deep learning network training. Generally, The effect of Dropout is the same as that of DropoutGenMask
    and DropoutDoMask. The DropoutGenMask generates a mask shape that is specified based on the input. Next,
    The DropoutDoMask is a mask generated using DropoutGenMask.
    The input tensor is randomly set to zero based on the probability p.

    Inputs:
        - **input_x** (Tensor) - The input tensor. Tensor of shape :math:`(N, \ldots)`.
          The data type should be float32, float16 or int32
        - **mask** (Tensor) - The mask to be applied on `input_x`, which is the output of `DropoutGenMask`. And the
          shape of `input_x` must be the same as the value of `DropoutGenMask`'s input `shape`. If input wrong `mask`,
          the output of `DropoutDoMask` are unpredictable.
        - **keep_prob** (Union[Tensor, float]) - The keep rate, greater than 0 and less equal than 1, e.g. keep_prob =
          0.9, means dropping out 10% of input units. The value of `keep_prob` is the same as the input `keep_prob` of
          the operator `DropoutGenMask`.

    Outputs:
        Tensor, the value that applied dropout on, as the same data type and shape as `input_x`.

    Raises:
        TypeError: If `input_x`, `mask` or `keep_prob` is not a Tensor.
        TypeError: If `keep_prob` is not a float.
        ValueError: If value of `keep_prob` is not same as `DropoutGenMaks`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> shape = (2, 2, 3)
        >>> keep_prob = Tensor(0.5, mindspore.float32)
        >>> dropout_gen_mask = ops.DropoutGenMask()
        >>> dropout_do_mask = ops.DropoutDoMask()
        >>> mask = dropout_gen_mask(shape, keep_prob)
        >>> output = dropout_do_mask(input_x, mask, keep_prob)
        >>> print(output.shape)
        (2, 2, 3)
    """

    @prim_attr_register
    def __init__(self):
        pass


class ResizeBilinear(PrimitiveWithInfer):
    r"""
    Resizes an image to a certain size using the bilinear interpolation.

    The resizing only affects the lower two dimensions which represent the height and width. The input images
    can be represented by different data types, but the data types of output images are always float32.

    Args:
        size (Union[tuple[int], list[int]]): A tuple or list of 2 int elements :math:`(new\_height, new\_width)`,
            the new size of the images.
        align_corners (bool): If true, rescale input by :math:`(new\_height - 1) / (height - 1)`,
                       which exactly aligns the 4 corners of images and resized images. If false,
                       rescale by :math:`new\_height / height`. Default: False.

    Inputs:
        - **x** (Tensor) - Image to be resized. Input images must be a 4-D tensor with shape
          :math:`(batch, channels, height, width)`, with data type of float32 or float16.

    Outputs:
        Tensor, resized image. 4-D with shape :math:`(batch, channels, new\_height, new\_width)`,
        with the same data type as input `x`.

    Raises:
        TypeError: If `size` is neither a tuple nor list.
        TypeError: If `align_corners` is not a bool.
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a Tensor.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mindspore.float32)
        >>> resize_bilinear = ops.ResizeBilinear((5, 5))
        >>> output = resize_bilinear(x)
        >>> print(output)
        [[[[1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]]]]
    """

    @prim_attr_register
    def __init__(self, size, align_corners=False):
        """Initialize ResizeBilinear."""
        validator.check_value_type("size", size, [tuple, list], self.name)
        validator.check_equal_int(len(size), 2, "size len", self.name)
        for item in size:
            validator.check_positive_int(item, 'size item', self.name)
            validator.check_value_type("size item", item, int, self.name)
        validator.check_value_type("align_corners", align_corners, [bool], self.name)
        for i, value in enumerate(size):
            validator.check_positive_int(value, f'{i}th value of size', self.name)

    def infer_shape(self, input_shape):
        validator.check("input shape rank", len(input_shape), "", 4, Rel.EQ, self.name)
        input_shape = list(input_shape)
        batch, channel, _, _ = input_shape
        out_shape = [batch, channel]
        for i in self.size:
            out_shape.append(int(i))
        return out_shape

    def infer_dtype(self, input_dtype):
        validator.check_tensor_dtype_valid('input_dtype', input_dtype, [mstype.float16, mstype.float32],
                                           self.name)
        return input_dtype


class OneHot(Primitive):
    r"""
    Computes a one-hot tensor.

    Makes a new tensor, whose locations represented by indices in `indices` take value `on_value`, while all
    other locations take value `off_value`.

    Note:
        If the input indices is rank `N`, the output will have rank `N+1`. The new axis is created at dimension `axis`.

    Args:
        axis (int): Position to insert the value. e.g. If shape of `indices` is :math:`(N, C)`, and `axis` is -1,
            the output shape will be :math:`(N, C, D)`, If `axis` is 0, the output shape will be :math:`(D, N, C)`.
            Default: -1.

    Inputs:
        - **indices** (Tensor) - A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
          Data type must be int32 or int64.
        - **depth** (int) - A scalar defining the depth of the one hot dimension.
        - **on_value** (Tensor) - A value to fill in output when `indices[j] = i`.
          With data type of float16 or float32.
        - **off_value** (Tensor) - A value to fill in output when `indices[j] != i`.
          Has the same data type as `on_value`.

    Outputs:
        Tensor, one-hot tensor. Tensor of shape :math:`(X_0, \ldots, X_{axis}, \text{depth} ,X_{axis+1}, \ldots, X_n)`.

    Raises:
        TypeError: If `axis` or `depth` is not an int.
        TypeError: If dtype of `indices` is neither int32 nor int64.
        TypeError: If `indices`, `on_value` or `off_value` is not a Tensor.
        ValueError: If `axis` is not in range [-1, len(indices_shape)].
        ValueError: If `depth` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> depth, on_value, off_value = 3, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        >>> onehot = ops.OneHot()
        >>> output = onehot(indices, depth, on_value, off_value)
        >>> print(output)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
    """

    @prim_attr_register
    def __init__(self, axis=-1):
        """Initialize OneHot."""
        self.init_prim_io_names(inputs=['indices', 'depth', 'on_value', 'off_value'], outputs=['output'])
        validator.check_value_type("axis", axis, [int], self.name)


class Gelu(PrimitiveWithInfer):
    """
    Same as operator GeLU. Gelu will be deprecated in the future.
    Please use GeLU instead.
    """

    @deprecated("1.1", "GeLU", True)
    @prim_attr_register
    def __init__(self):
        """Initialize Gelu"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, (mstype.float16, mstype.float32), self.name)
        return input_x


class GeLU(Primitive):
    r"""
    Gaussian Error Linear Units activation function.

    GeLU is described in the paper `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    And also please refer to `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    <https://arxiv.org/abs/1810.04805>`_.

    GeLU is defined as follows:

    .. math::
        \text{output} = 0.5 * x * (1 + erf(x / \sqrt{2})),

    where :math:`erf` is the "Gauss error function" .

    Inputs:
        - **x** (Tensor) - Input to compute the GeLU with data type of float16 or float32.

    Outputs:
        Tensor, with the same type and shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> gelu = ops.GeLU()
        >>> result = gelu(x)
        >>> print(result)
        [0.841192  1.9545976  2.9963627]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize GeLU"""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])


class FastGelu(PrimitiveWithInfer):
    """
    Same as operator FastGeLU. FastGelu will be deprecated in the future.
    Please use FastGeLU instead.
    """

    @deprecated("1.1", "FastGeLU", True)
    @prim_attr_register
    def __init__(self):
        """Initialize FastGelu."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, (mstype.float16, mstype.float32), self.name)
        return input_x


class FastGeLU(PrimitiveWithInfer):
    r"""
    Fast Gaussian Error Linear Units activation function.

    FastGeLU is defined as follows:

    .. math::
        \text{output} = \frac {x} {1 + \exp(-1.702 * \left| x \right|)} * \exp(0.851 * (x - \left| x \right|)),

    where :math:`x` is the element of the input.

    Inputs:
        - **x** (Tensor) - Input to compute the FastGeLU with data type of float16 or float32.

    Outputs:
        Tensor, with the same type and shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> fast_gelu = ops.FastGeLU()
        >>> output = fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize FastGeLU."""
        self.init_prim_io_names(inputs=['x'], outputs=['output'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        validator.check_tensor_dtype_valid("input_x", input_x, (mstype.float16, mstype.float32), self.name)
        return input_x


class GetNext(Primitive):
    """
    Returns the next element in the dataset queue.

    Note:
        The GetNext operation needs to be associated with network and it also depends on the init_dataset interface,
        it can't be used directly as a single operation.
        For details, please refer to `connect_network_with_dataset` source code.

    Args:
        types (list[:class:`mindspore.dtype`]): The type of the outputs.
        shapes (list[tuple[int]]): The dimensionality of the outputs.
        output_num (int): The output number, length of `types` and `shapes`.
        shared_name (str): The queue name of `init_dataset` interface.

    Inputs:
        No inputs.

    Outputs:
        tuple[Tensor], the output of Dataset. The shape is described in `shapes`
        and the type is described in `types`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> train_dataset = create_custom_dataset()
        >>> dataset_helper = mindspore.DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>> dataset = dataset_helper.iter.dataset
        >>> dataset_types, dataset_shapes = dataset_helper.types_shapes()
        >>> queue_name = dataset.__transfer_dataset__.queue_name
        >>> get_next = ops.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)
        >>> data, label = get_next()
        >>> relu = ops.ReLU()
        >>> result = relu(data).asnumpy()
        >>> print(result.shape)
        (32, 1, 32, 32)
    """

    @prim_attr_register
    def __init__(self, types, shapes, output_num, shared_name):
        """Initialize GetNext."""
        validator.check_value_type("types", types, [list, tuple], self.name)
        validator.check_value_type("shapes", shapes, [list, tuple], self.name)
        validator.check("types length", len(types), "shapes length", len(shapes), Rel.EQ, self.name)
        validator.check_value_type("output_num", output_num, [int], self.name)


class PReLU(PrimitiveWithInfer):
    r"""
    Parametric Rectified Linear Unit activation function.

    PReLU is described in the paper `Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification <https://arxiv.org/abs/1502.01852>`_. Defined as follows:

    .. math::
        prelu(x_i)= \max(0, x_i) + \min(0, w * x_i),

    where :math:`x_i` is an element of an channel of the input, `w` is the weight of the channel.

    Note:
        0-D or 1-D input_x is not supported on Ascend.

    Inputs:
        - **x** (Tensor) - The first input tensor, representing the output of the preview layer.
          With data type of float16 or float32.
          The shape is :math:`(N, C, *)` where :math:`*` means, any number of additional dimensions.
        - **weight** (Tensor) -  The second input tensor. The data type is float16 or float32.
          There are only two shapes are legitimate, 1 or the number of channels of the `input_x`.
          Channel dim is the 2nd dim of input. When input is 0-D or 1-D tensor, the number of channels is 1.

    Outputs:
        Tensor, with the same type as `x`.

    For detailed information, please refer to :class:`nn.PReLU`.

    Raises:
        TypeError: If dtype of `x` or `weight` is neither float16 nor float32.
        TypeError: If the `x` or the `weight` is not a Tensor.
        ValueError: If the `x` is a 0-D or 1-D Tensor on Ascned.
        ValueError: If the `weight` is not a 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.prelu = ops.PReLU()
        ...     def construct(self, x, weight):
        ...         result = self.prelu(x, weight)
        ...         return result
        ...
        >>> x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)), mindspore.float32)
        >>> weight = Tensor(np.array([0.1, 0.6, -0.3]), mindspore.float32)
        >>> net = Net()
        >>> output = net(x, weight)
        >>> print(output)
        [[[-0.60 -0.50]
          [-2.40 -1.80]
          [ 0.60  0.30]]
         [[ 0.00  1.00]
          [ 2.00  3.00]
          [ 4.0   5.00]]]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, input_x_shape, weight_shape):
        input_x_dim = len(input_x_shape)
        if input_x_dim in (0, 1):
            if context.get_context("device_target") == "Ascend":
                raise ValueError(f"For '{self.name}', the dimension of 'x' can not be 0-D or 1-D when the platform is "
                                 f"\"Ascend\", but got dimension of 'x' is {input_x_dim}.")
            channel_num = 1
        else:
            channel_num = input_x_shape[1]

        weight_dim = len(weight_shape)
        if weight_dim != 1:
            raise ValueError(f"For '{self.name}', the dimension of 'weight' should be 1, while got {weight_dim}.")
        if weight_shape[0] != 1 and weight_shape[0] != channel_num:
            raise ValueError(f"For '{self.name}', the first dimension of 'weight' should be (1,) or "
                             f"it should be equal to number of channels: {channel_num}, but got {weight_shape}")
        return input_x_shape

    def infer_dtype(self, input_x_dtype, weight_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        args = {"input_x": input_x_dtype, "weight": weight_dtype}
        if context.get_context("device_target") == "GPU":
            validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        else:
            validator.check_tensor_dtype_valid("input_x", input_x_dtype, valid_dtypes, self.name)
            validator.check_tensor_dtype_valid("weight", weight_dtype, valid_dtypes, self.name)
        return input_x_dtype


class LSTM(PrimitiveWithInfer):
    """
    Performs the Long Short-Term Memory (LSTM) on the input.

    For detailed information, please refer to :class:`nn.LSTM`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`.
        bidirectional (bool): Specifies whether it is a bidirectional LSTM.
        dropout (float): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. The range of dropout is [0.0, 1.0].

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`) or
          (batch_size, seq_len, `input_size`).
        - **h** (tuple) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c** (tuple) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`, `c_n`, `reserve`, `state`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** (Tensor) - Tensor of shape (r, 1).
        - **state** (Tensor) - Random number generator state and its shape is (s, 1).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_size = 10
        >>> hidden_size = 2
        >>> num_layers = 1
        >>> seq_len = 5
        >>> batch_size = 2
        >>>
        >>> net = ops.LSTM(input_size, hidden_size, num_layers, True, False, 0.0)
        >>> input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
        >>> h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> w = Tensor(np.ones([112, 1, 1]).astype(np.float32))
        >>> output, hn, cn, _, _ = net(input_tensor, h0, c0, w)
        >>> print(output)
        [[[0.9640267  0.9640267 ]
          [0.9640267  0.9640267 ]]
         [[0.9950539  0.9950539 ]
          [0.9950539  0.9950539 ]]
         [[0.99932843 0.99932843]
          [0.99932843 0.99932843]]
         [[0.9999084  0.9999084 ]
          [0.9999084  0.9999084 ]]
         [[0.9999869  0.9999869 ]
          [0.9999869  0.9999869 ]]]
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        """Initialize LSTM."""
        self.input_size = validator.check_positive_int(input_size, "input_size", self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, "hidden_size", self.name)
        self.num_layers = validator.check_positive_int(num_layers, "num_layers", self.name)
        self.has_bias = validator.check_value_type("has_bias", has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type("bidirectional", bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, Rel.INC_BOTH, 'dropout', self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, h_shape, c_shape, w_shape):
        validator.check_equal_int(len(x_shape), 3, "x rank", self.name)
        validator.check_equal_int(x_shape[2], self.input_size, "x[2]", self.name)

        # h and c should be same shape
        validator.check_equal_int(len(h_shape), 3, "h rank", self.name)
        validator.check("h_shape", h_shape, "c_shape", c_shape, Rel.EQ, self.name)

        validator.check_int(h_shape[0], self.num_layers * self.num_directions, Rel.EQ, "h[0]", self.name)
        validator.check_equal_int(h_shape[1], x_shape[1], "h[1]", self.name)
        validator.check_int(h_shape[2], self.hidden_size, Rel.EQ, "h[2]", self.name)

        y_shape = (x_shape[0], x_shape[1], self.hidden_size * self.num_directions)

        # set arbitrary shape for reserved space
        reserved_shape = (1, 1)
        state_shape = (1, 1)
        return y_shape, h_shape, c_shape, reserved_shape, state_shape

    def infer_dtype(self, x_dtype, h_dtype, c_dtype, w_dtype):
        args = {'x': x_dtype, 'h': h_dtype, 'c': c_dtype, 'w': w_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32, mstype.float16), self.name)
        return x_dtype, x_dtype, x_dtype, x_dtype, x_dtype


class SigmoidCrossEntropyWithLogits(PrimitiveWithInfer):
    r"""
    Uses the given logits to compute sigmoid cross entropy between the logits and the label.

    Measures the distribution error in discrete classification tasks where each class is independent
    and not mutually exclusive using cross entropy loss.

    Sets input logits as :math:`X`, input label as :math:`Y`, output as :math:`loss`. Then,

    .. math::

        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            loss_{ij} = -[Y_{ij} * ln(p_{ij}) + (1 - Y_{ij})ln(1 - p_{ij})]
        \end{array}

    Inputs:
        - **logits** (Tensor) - Input logits. Tensor of shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions.
        - **label** (Tensor) - Ground truth label. With the same shape and type as `logits`.

    Outputs:
        Tensor, with the same shape and type as input `logits`.

    Raises:
        TypeError: If `logits` or `label` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
        >>> labels = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
        >>> sigmoid = ops.SigmoidCrossEntropyWithLogits()
        >>> output = sigmoid(logits, labels)
        >>> print(output)
        [[ 0.6111007   0.5032824   0.26318604]
         [ 0.58439666  0.5530153  -0.4368139 ]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize SigmoidCrossEntropyWithLogits"""
        self.init_prim_io_names(inputs=['predict', 'target'], outputs=['loss'])

    def infer_shape(self, x_shape, y_shape):
        validator.check("x_shape", x_shape, "y_shape", y_shape, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_dtype, y_dtype):
        args = {"x_dtype": x_dtype, "y_dtype": y_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return x_dtype


class BCEWithLogitsLoss(PrimitiveWithInfer):
    r"""
    Adds sigmoid activation function to input `logits`, and uses the given logits to compute binary cross entropy
    between the logits and the label.

    Sets input logits as :math:`X`, input label as :math:`Y`, input weight as :math:`W`, output as :math:`L`. Then,

    .. math::

        \begin{array}{ll} \\
            L_{ij} = -W_{ij}[Y_{ij}log(X_{ij}) + (1 - Y_{ij})log(1 - X_{ij})]
        \end{array}

    :math:`i` indicates the :math:`i^{th}` sample, :math:`j` indicates the category. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`\ell` indicates the method of calculating the loss. There are three methods:
    the first method is to provide the loss value directly,
    the second method is to calculate the average value of all losses,
    and the third method is to calculate the sum of all losses.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are 'mean', 'sum', and 'none',
             not case sensitive. If 'none', do not perform reduction. Default:'mean'.

    Inputs:
        - **logits** (Tensor) - Input logits. Data type must be float16 or float32.
          Tensor of shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **label** (Tensor) - Ground truth label, has the same shape as `logits`.
          Data type must be float16 or float32.
        - **weight** (Tensor) - A rescaling weight applied to the loss of each batch element. It can be
          broadcast to a tensor with shape of `logits`. Data type must be float16 or float32.
        - **pos_weight** (Tensor) - A weight of positive examples. Must be a vector with length equal to the
          number of classes. It can be broadcast to a tensor with shape of `logits`.
          Data type must be float16 or float32.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', it's a tensor with the same shape and type as input `logits`.
        Otherwise, the output is a scalar.

    Raises:
        TypeError: If data type of any input is neither float16 nor float32.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `logits`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
        >>> label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
        >>> weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> loss = ops.BCEWithLogitsLoss()
        >>> output = loss(logits, label, weight, pos_weight)
        >>> print(output)
        0.3463612
    """

    @prim_attr_register
    def __init__(self, reduction='mean'):
        """Initialize BCEWithLogitsLoss"""
        self.reduction = validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.name)

    def infer_shape(self, logits, label, weight, pos_weight):
        validator.check('logits_shape', logits, 'label_shape', label, Rel.EQ, self.name)
        reversed_weight_shape = tuple(reversed(weight))
        reversed_label = tuple(reversed(logits))
        for i, v in enumerate(reversed_weight_shape):
            if v not in (reversed_label[i], 1):
                raise ValueError(f"For {self.name}, the shapes of 'logits' and 'weight' can not broadcast. "
                                 f"'logits': {tuple(logits)}, 'weight' shape {tuple(weight)}.")

        reversed_pos_shape = tuple(reversed(pos_weight))
        reversed_label = tuple(reversed(logits))
        for i, v in enumerate(reversed_pos_shape):
            if v not in (reversed_label[i], 1):
                raise ValueError(f"For {self.name}, the shapes of 'logits' and 'pos_weight' can not broadcast. "
                                 f"'logits': {tuple(logits)}, 'pos_weight' shape {tuple(pos_weight)}.")

        if self.reduction in ('mean', 'sum'):
            shape = []
        else:
            shape = logits
        return shape

    def infer_dtype(self, logits, label, weight, pos_weight):
        validator.check_tensor_dtype_valid('logits dtype', logits, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('label dtype', label, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('weight dtype', weight, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('pos_weight dtype', pos_weight, [mstype.float16, mstype.float32], self.name)
        return logits


class Pad(PrimitiveWithInfer):
    """
    Pads the input tensor according to the paddings.

    Args:
        paddings (tuple): The shape of parameter `paddings` is (N, 2). N is the rank of input data. All elements of
            paddings are int type. For the input in `D` th dimension, paddings[D, 0] indicates how many sizes to be
            extended ahead of the input tensor in the `D` th dimension, and paddings[D, 1] indicates how many sizes to
            be extended behind the input tensor in the `D` th dimension.

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `paddings` is not a tuple.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If shape of `paddings` is not :math:`(N, 2)`.
        ValueError: If paddings.size is not equal to 2 * len(input_x).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> pad_op = ops.Pad(((1, 2), (2, 1)))
        >>> output = pad_op(input_x)
        >>> print(output)
        [[ 0.   0.   0.   0.   0.   0. ]
         [ 0.   0.  -0.1  0.3  3.6  0. ]
         [ 0.   0.   0.4  0.5 -3.2  0. ]
         [ 0.   0.   0.   0.   0.   0. ]
         [ 0.   0.   0.   0.   0.   0. ]]
    """

    @prim_attr_register
    def __init__(self, paddings):
        """Initialize Pad"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        if not isinstance(paddings, tuple):
            raise TypeError(f"For '{self.name}', the type of 'paddings' must be tuple, "
                            f"but got {type(paddings)}.")
        for item in paddings:
            if len(item) != 2:
                raise ValueError(f"For '{self.name}', the shape of 'paddings' must be (n, 2), "
                                 f"but got {paddings}.")
        self.paddings = paddings

    def infer_shape(self, x_shape):
        validator.check_int(len(self.paddings), len(x_shape), Rel.EQ, 'paddings.shape', self.name)
        paddings = np.array(self.paddings)
        if not np.all(paddings >= 0):
            raise ValueError(f"For '{self.name}', all elements of paddings must be >= 0.")
        y_shape = ()
        for i in range(int(paddings.size / 2)):
            y_shape += ((x_shape[i] + paddings[i, 0] + paddings[i, 1]),)
        return y_shape

    def infer_dtype(self, x_dtype):
        validator.check_subclass("input_x", x_dtype, mstype.tensor, self.name)
        return x_dtype


class MirrorPad(PrimitiveWithInfer):
    """
    Pads the input tensor according to the paddings and mode.

    Args:
        mode (str): Specifies the padding mode. The optional values are "REFLECT" and "SYMMETRIC".
            Default: "REFLECT".

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions.
        - **paddings** (Tensor) - The paddings tensor. The value of `paddings` is a matrix(list),
          and its shape is (N, 2). N is the rank of input data. All elements of paddings
          are int type. For the input in the `D` th dimension, paddings[D, 0] indicates how many sizes to be
          extended ahead of the input tensor in the `D` th dimension, and paddings[D, 1] indicates how many sizes to
          be extended behind the input tensor in the `D` th dimension.

    Outputs:
        Tensor, the tensor after padding.

        - If `mode` is "REFLECT", it uses a way of symmetrical copying through the axis of symmetry to fill in.
          If the `input_x` is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the
          Outputs is [[6,5,4,5,6,5,4], [3,2,1,2,3,2,1], [6,5,4,5,6,5,4], [9,8,7,8,9,8,7], [6,5,4,5,6,5,4]].
        - If `mode` is "SYMMETRIC", the filling method is similar to the "REFLECT". It is also copied
          according to the symmetry axis, except that it includes the symmetry axis. If the `input_x`
          is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the Outputs is
          [[2,1,1,2,3,3,2], [2,1,1,2,3,3,2], [5,4,4,5,6,6,5], [8,7,7,8,9,9,8], [8,7,7,8,9,9,8]].

    Raises:
        TypeError: If `input_x` or `paddings` is not a Tensor.
        TypeError: If `mode` is not a str.
        ValueError: If paddings.size is not equal to 2 * len(input_x).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case1: mode="REFLECT"
        >>> class Net(nn.Cell):
        ...    def __init__(self, mode):
        ...        super(Net, self).__init__()
        ...        self.pad = ops.MirrorPad(mode=mode)
        ...        self.paddings = Tensor([[1, 1], [2, 2]])
        ...    def construct(self, input_x):
        ...        return self.pad(input_x, self.paddings)
        ...
        >>> input_x = Tensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> pad = Net("REFLECT")
        >>> output = pad(input_x)
        >>> print(output)
        [[6 5 4 5 6 5 4]
         [3 2 1 2 3 2 1]
         [6 5 4 5 6 5 4]
         [9 8 7 8 9 8 7]
         [6 5 4 5 6 5 4]]
        >>> # case2: mode="SYMMETRIC"
        >>> pad = Net("SYMMETRIC")
        >>> output = pad(input_x)
        >>> print(output)
        [[2 1 1 2 3 3 2]
         [2 1 1 2 3 3 2]
         [5 4 4 5 6 6 5]
         [8 7 7 8 9 9 8]
         [8 7 7 8 9 9 8]]
    """

    @prim_attr_register
    def __init__(self, mode='REFLECT'):
        """Initialize Pad"""
        validator.check_string(mode, ['REFLECT', 'SYMMETRIC'], 'mode', self.name)
        self.mode = mode
        self.set_const_input_indexes([1])

    def __infer__(self, input_x, paddings):
        validator.check_subclass("input_x", input_x['dtype'], mstype.tensor, self.name)
        validator.check_subclass("paddings", paddings['dtype'], mstype.tensor, self.name)
        x_shape = list(input_x['shape'])
        paddings_value = paddings['value'].asnumpy()
        paddings_size = paddings_value.size
        validator.check_int(paddings_size, len(x_shape) * 2, Rel.EQ, 'paddings.shape', self.name)
        if not np.all(paddings_value >= 0):
            raise ValueError(f"For '{self.name}', all elements of 'paddings' must be >= 0.")
        adjust = 0
        if self.mode == 'SYMMETRIC':
            adjust = 1
        for i in range(0, int(paddings_size / 2)):
            if (paddings_value[i, 0] >= x_shape[i] + adjust) or (paddings_value[i, 1] >= x_shape[i] + adjust):
                msg = "x_shape[D] + 1" if adjust == 1 else "x_shape[D]"
                raise ValueError(f"For '{self.name}', both paddings[D, 0] and paddings[D, 1] must be less than {msg}, "
                                 f"but got paddings[{i}, 0]: {paddings[i, 0]}, "
                                 f"paddings[{i}, 1]: {paddings[i, 1]}, x_shape[{i}]: {x_shape[i]}.")
        y_shape = ()
        for i in range(0, int(paddings_size / 2)):
            y_shape += ((x_shape[i] + paddings_value[i, 0] + paddings_value[i, 1]),)
        return {'shape': y_shape,
                'dtype': input_x['dtype'],
                'value': None}


class ComputeAccidentalHits(PrimitiveWithCheck):
    r"""
    Compute accidental hits of sampled classes which match target classes.

    When a target class matches the sample class, we call it "accidental hit".
    The result of calculating accidental hits contain three parts (index, id, weight),
    where index represents the row number in true_classes, and id represents the position in sampled_candidates,
    the weight is -FLOAT_MAX. FLOAT_MAX indicates the max value in the type of Float

    Args:
        num_true (int): The number of target classes per training example. Default: 1.

    Inputs:
        - **true_classes** (Tensor) - The target classes. With data type of int32 or int64
          and shape :math:`(batch\_size, num\_true)`.
        - **sampled_candidates** (Tensor) - The Candidate sampling results of operators, types of training samples,
          with data type of int32 or int64 and shape :math:`(num\_sampled, )`.

    Outputs:
        Tuple of 3 Tensors.

        - **indices** (Tensor) - A Tensor with shape :math:`(num\_accidental\_hits, )`,
          with the same type as `true_classes`.
        - **ids** (Tensor) - A Tensor with shape :math:`(num\_accidental\_hits, )`,
          with the same type as `true_classes`.
        - **weights** (Tensor) - A Tensor with shape :math:`(num\_accidental\_hits, )`, with the type float32.

    Raises:
        TypeError: If dtype of `num_true` is not int.
        TypeError: If `true_classes` or `sampled_candidates` is not a Tensor.
        TypeError: If dtype of `true_classes` or `sampled_candidates` is neither int32 nor int64.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> true_classes = np.array([[1, 2], [0, 4], [3, 3]])
        >>> sampled_candidates = np.array([0, 1, 2, 3, 4])
        >>> sampler = ops.ComputeAccidentalHits(2)
        >>> indices, ids, weights = sampler(Tensor(true_classes), Tensor(sampled_candidates))
        >>> print(indices, ids, weights)
        [0 0 1 1 2 2]
        [1 2 0 4 3 3]
        [-3.4028235e+38 -3.4028235e+38 -3.4028235e+38 -3.4028235e+38 -3.4028235e+38 -3.4028235e+38]

    """

    @prim_attr_register
    def __init__(self, num_true=1):
        """Initialize ComputeAccidentalHits"""
        self.init_prim_io_names(inputs=['true_classes', 'sampled_candidates'],
                                outputs=['indices', 'ids', 'weights'])
        validator.check_value_type("num_true", num_true, [int], self.name)
        validator.check_number("num_true", num_true, 1, Rel.GE, self.name)
        self.num_true = num_true

    def check_shape(self, true_classes_shape, sampled_candidates_shape):
        validator.check_int(len(true_classes_shape), 2, Rel.EQ, 'dim of true_classes', self.name)
        validator.check_int(len(sampled_candidates_shape), 1, Rel.EQ, 'dim of sampled_candidates', self.name)
        validator.check("true_classes shape[1]", true_classes_shape[1], "num_true", self.num_true, Rel.EQ, self.name)

        indices_len = -1
        return (indices_len,), (indices_len,), (indices_len,)

    def check_dtype(self, true_classes_type, sampled_candidates_type):
        validator.check_subclass("true_classes_type", true_classes_type, mstype.tensor, self.name)
        validator.check_subclass("sampled_candidates_type", sampled_candidates_type, mstype.tensor, self.name)
        valid_types = (mstype.int32, mstype.int64)
        validator.check_tensor_dtype_valid("true_classes_type", true_classes_type, valid_types, self.name)
        validator.check_tensor_dtype_valid("sampled_candidates_type", sampled_candidates_type, valid_types, self.name)
        weights_type = mstype.float32
        return true_classes_type, true_classes_type, weights_type


class ROIAlign(PrimitiveWithInfer):
    r"""
    Computes the Region of Interest (RoI) Align operator.

    The operator computes the value of each sampling point by bilinear interpolation from the nearby grid points on the
    feature map. No quantization is performed on any coordinates involved in the RoI, its bins, or the sampling
    points. The details of (RoI) Align operator are described in `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_.

    Args:
        pooled_height (int): The output features height.
        pooled_width (int): The output features width.
        spatial_scale (float): A scaling factor that maps the raw image coordinates to the input
            feature map coordinates. Suppose the height of a RoI is `ori_h` in the raw image and `fea_h` in the
            input feature map, the `spatial_scale` must be `fea_h / ori_h`.
        sample_num (int): Number of sampling points. Default: 2.
        roi_end_mode (int): Number must be 0 or 1. Default: 1.

    Inputs:
        - **features** (Tensor) - The input features, whose shape must be :math:`(N, C, H, W)`.
        - **rois** (Tensor) - The shape is :math:`(rois\_n, 5)`. With data type of float16 or float32.
          `rois_n` represents the number of RoI. The size of the second dimension must be `5` and the `5` colunms
          are :math:`(image\_index, top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)`.
          `image_index` represents the index of image. `top_left_x` and `top_left_y` represent the `x, y`
          coordinates of the top left corner of corresponding RoI, respectively. `bottom_right_x` and `bottom_right_y`
          represent the `x, y` coordinates of the bottom right corner of corresponding RoI, respectively.

    Outputs:
        Tensor, the shape is :math:`(rois\_n, C, pooled\_height, pooled\_width)`.

    Raises:
        TypeError: If `pooled_height`, `pooled_width`, `sample_num` or `roi_end_mode` is not an int.
        TypeError: If `spatial_scale` is not a float.
        TypeError: If `features` or `rois` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> features = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
        >>> rois = Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), mindspore.float32)
        >>> roi_align = ops.ROIAlign(2, 2, 0.5, 2)
        >>> output = roi_align(features, rois)
        >>> print(output)
        [[[[1.775 2.025]
           [2.275 2.525]]]]
    """

    @prim_attr_register
    def __init__(self, pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1):
        """Initialize ROIAlign"""
        validator.check_value_type("pooled_height", pooled_height, [int], self.name)
        validator.check_value_type("pooled_width", pooled_width, [int], self.name)
        validator.check_value_type("spatial_scale", spatial_scale, [float], self.name)
        validator.check_value_type("sample_num", sample_num, [int], self.name)
        validator.check_value_type("roi_end_mode", roi_end_mode, [int], self.name)
        validator.check_int_range(roi_end_mode, 0, 1, Rel.INC_BOTH, "roi_end_mode", self.name)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num
        self.roi_end_mode = roi_end_mode

    def infer_shape(self, inputs_shape, rois_shape):
        validator.check("input shape rank", len(inputs_shape), "", 4, Rel.LE, self.name)
        return [rois_shape[0], inputs_shape[1], self.pooled_height, self.pooled_width]

    def infer_dtype(self, inputs_type, rois_type):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("inputs_type", inputs_type, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid("rois_type", rois_type, valid_dtypes, self.name)
        return inputs_type


class Adam(PrimitiveWithInfer):
    r"""
    Updates gradients by the Adaptive Moment Estimation (Adam) algorithm.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    For more details, please refer to :class:`nn.Adam`.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t(\beta_1^{t})` and :math:`beta_2^t(\beta_2^{t})`
    represent `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `var`,
    :math:`\epsilon` represents
    `epsilon`.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be updated. The shape is :math:`(N, *)` where :math:`*` means,
          any number of additional dimensions. The data type can be float16 or float32.
        - **m** (Tensor) - The 1st moment vector in the updating formula,
          the shape and data type value should be the same as `var`.
        - **v** (Tensor) - the 2nd moment vector in the updating formula,
          the shape and data type value should be the same as `var`. Mean square gradients with the same type as `var`.
        - **beta1_power** (float) - :math:`beta_1^t(\beta_1^{t})` in the updating formula,
          the data type value should be the same as `var`.
        - **beta2_power** (float) - :math:`beta_2^t(\beta_2^{t})` in the updating formula,
          the data type value should be the same as `var`.
        - **lr** (float) - :math:`l` in the updating formula. The paper suggested value is :math:`10^{-8}`,
          the data type value should be the same as `var`.
        - **beta1** (float) - The exponential decay rate for the 1st moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.9`
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.999`
        - **epsilon** (float) - Term added to the denominator to improve numerical stability.
        - **gradient** (Tensor) - Gradient, has the same shape and data type as `var`.

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as Inputs `var`.
        - **m** (Tensor) - The same shape and data type as Inputs `m`.
        - **v** (Tensor) - The same shape and data type as Inputs `v`.

    Raises:
        TypeError: If neither `use_locking` nor `use_nesterov` is a bool.
        TypeError: If `var`, `m` or `v` is not a Tensor.
        TypeError: If `beta1_power`, `beta2_power1`, `lr`, `beta1`, `beta2`, `epsilon` or `gradient` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_adam = ops.Adam()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="m")
        ...         self.v = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="v")
        ...     def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        ...         out = self.apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2,
        ...                               epsilon, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> gradient = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> output = net(0.9, 0.999, 0.001, 0.9, 0.999, 1e-8, gradient)
        >>> print(net.var.asnumpy())
        [[0.9996838 0.9996838]
         [0.9996838 0.9996838]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False, use_nesterov=False):
        """Initialize Adam."""
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return var_shape, m_shape, v_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, beta1_power_dtype, beta2_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        args = {"beta1_power": beta1_power_dtype, "beta2_power": beta2_power_dtype, 'lr': lr_dtype,
                "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float16, mstype.float32], self.name, True)
        return var_dtype, m_dtype, v_dtype


class AdamWeightDecay(PrimitiveWithInfer):
    r"""
    Updates gradients by the Adaptive Moment Estimation (AdamWeightDecay) algorithm with weight decay.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.
    The AdamWeightDecay variant was proposed in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            update = \frac{m}{\sqrt{v} + eps} \\
            update =
            \begin{cases}
                update + weight\_decay * w
                    & \text{ if } weight\_decay > 0 \\
                update
                    & \text{ otherwise }
            \end{cases} \\
            w  = w - lr * update
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`lr` represents `learning_rate`, :math:`w` represents `var`, :math:`decay` represents `weight_decay`,
    :math:`\epsilon` represents `epsilon`.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.

    Inputs:
        - **var** (Tensor) - Weights to be updated. The shape is :math:`(N, *)` where :math:`*` means,
          any number of additional dimensions. The data type can be float16 or float32.
        - **m** (Tensor) - The 1st moment vector in the updating formula,
          the shape and data type value should be the same as `var`.
        - **v** (Tensor) - the 2nd moment vector in the updating formula,
          the shape and data type value should be the same as `var`. Mean square gradients with the same type as `var`.
        - **lr** (float) - :math:`l` in the updating formula. The paper suggested value is :math:`10^{-8}`,
          the data type value should be the same as `var`.
        - **beta1** (float) - The exponential decay rate for the 1st moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.9`
        - **beta2** (float) - The exponential decay rate for the 2nd moment estimations,
          the data type value should be the same as `var`. The paper suggested value is :math:`0.999`
        - **epsilon** (float) - Term added to the denominator to improve numerical stability.
        - **decay** (float) - The weight decay value, must be a scalar tensor with float data type.
          Default: 0.0.
        - **gradient** (Tensor) - Gradient, has the same shape and data type as `var`.
    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.
        - **v** (Tensor) - The same shape and data type as `v`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, Parameter, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.adam_weight_decay = ops.AdamWeightDecay()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="m")
        ...         self.v = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="v")
        ...     def construct(self, lr, beta1, beta2, epsilon, decay, grad):
        ...         out = self.adam_weight_decay(self.var, self.m, self.v, lr, beta1, beta2,
        ...                               epsilon, decay, grad)
        ...         return out
        >>> net = Net()
        >>> gradient = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> output = net(0.001, 0.9, 0.999, 1e-8, 0.0, gradient)
        >>> print(net.var.asnumpy())
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize AdamWeightDecay."""
        self.add_prim_attr('side_effect_mem', True)
        validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, m_shape, v_shape, lr_shape, beta1_shape, beta2_shape,
                    epsilon_shape, decay_shape, grad_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "grad_shape", grad_shape, Rel.EQ, self.name)
        return var_shape, m_shape, v_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, lr_dtype, beta1_dtype, beta2_dtype,
                    epsilon_dtype, decay_dtype, grad_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        args = {"lr": lr_dtype, "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype,
                "decay": decay_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float32], self.name, True)
        return var_dtype, m_dtype, v_dtype


class AdamNoUpdateParam(PrimitiveWithInfer):
    r"""
    Updates gradients by Adaptive Moment Estimation (Adam) algorithm. This operator do not update the parameter, but
    calculate the value that should be added to the parameter instead.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            \Delta{w} = - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t(\beta_1^{t})` and :math:`beta_2^t(\beta_2^{t})`
    represent `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`,
    :math:`w` represents the parameter to be updated, :math:`\epsilon` represents `epsilon`.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.

    Inputs:
        - **m** (Tensor) - The 1st moment vector in the updating formula. The shape is :math:`(N, *)`
          where :math:`*` means, any number of additional dimensions. The data type must be float32.
        - **v** (Tensor) - the 2nd moment vector in the updating formula. The shape must be the same as `m`.
          The data type must be float32.
        - **beta1_power** (Tensor) - :math:`beta_1^t(\beta_1^{t})` in the updating formula.
          The shape is :math:`(1, )` and the data type must be float32.
        - **beta2_power** (Tensor) - :math:`beta_2^t(\beta_1^{t})` in the updating formula.
          The shape is :math:`(1, )` and the data type must be float32.
        - **lr** (Tensor) - :math:`l` in the updating formula.
          The shape is :math:`(1, )` and the data type must be float32.
        - **beta1** (Tensor) - The exponential decay rate for the 1st moment estimations.
          The shape is :math:`(1, )` and the data type must be float32.
        - **beta2** (Tensor) - The exponential decay rate for the 2nd moment estimations.
          The shape is :math:`(1, )` and the data type must be float32.
        - **epsilon** (Tensor) - Term added to the denominator to improve numerical stability.
          The shape is :math:`(1, )` and the data type must be float32.
        - **gradient** (Tensor) - Gradient, the shape must be the same as `m`, the data type must be float32.

    Outputs:
        Tensor, whose shape and data type are the same with Inputs `gradient`, is a value that should be added to the
        parameter to be updated.

    Raises:
        TypeError: If neither `use_locking` nor `use_nesterov` is a bool.
        TypeError: If `m`,  `v`, `beta1_power`, `beta2_power1`, `lr`, `beta1`, `beta2`, `epsilon` or `gradient`
                   is not a Tensor.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.adam = ops.AdamNoUpdateParam()
        ...         self.m = Parameter(Tensor(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]).astype(np.float32)),
        ...                            name="m")
        ...         self.v = Parameter(Tensor(np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]).astype(np.float32)),
        ...                            name="v")
        ...     def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad):
        ...         out = self.adam(self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad)
        ...         return out
        >>> net = Net()
        >>> beta1_power = Tensor(0.9, ms.float32)
        >>> beta2_power = Tensor(0.999, ms.float32)
        >>> lr = Tensor(0.001, ms.float32)
        >>> beta1 = Tensor(0.9, ms.float32)
        >>> beta2 = Tensor(0.999, ms.float32)
        >>> epsilon = Tensor(1e-8, ms.float32)
        >>> gradient = Tensor(np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]).astype(np.float32))
        >>> result = net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient)
        >>> print(result)
        [[-0.00010004 -0.00010004 -0.00010004]
        [-0.00013441 -0.00013441 -0.00013441]]

    """

    @prim_attr_register
    def __init__(self, use_locking=False, use_nesterov=False):
        """Initialize AdamNoUpdateParam."""
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.name)

    def infer_shape(self, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape):
        validator.check("grad_shape", grad_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("grad_shape", grad_shape, "v_shape", v_shape, Rel.EQ, self.name)
        return grad_shape

    def infer_dtype(self, m_dtype, v_dtype, beta1_power_dtype, beta2_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype):
        args = {"m": m_dtype, "v": v_dtype, "grad": grad_dtype,
                "beta1_power": beta1_power_dtype, "beta2_power": beta2_power_dtype, 'lr': lr_dtype,
                "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float32], self.name)
        return grad_dtype


class FusedSparseAdam(PrimitiveWithInfer):
    r"""
    Merges the duplicate value of the gradient and then updates parameters by the Adaptive Moment Estimation (Adam)
    algorithm. This operator is used when the gradient is sparse.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent `beta1_power` and
    `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `var`, :math:`\epsilon` represents
    `epsilon`.

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.

    Inputs:
        - **var** (Parameter) - Parameters to be updated with float32 data type. The shape is :math:`(N, *)`
          where :math:`*` means, any number of additional dimensions.
        - **m** (Parameter) - The 1st moment vector in the updating formula, has the same shape and data type as `var`.
        - **v** (Parameter) - The 2nd moment vector in the updating formula, has the same shape and data type as `var`.
          Mean square gradients, has the same type as `var` with float32 data type.
        - **beta1_power** (Tensor) - :math:`beta_1^t` in the updating formula with float32 data type.
          The shape is :math:`(1, )`.
        - **beta2_power** (Tensor) - :math:`beta_2^t` in the updating formula with float32 data type.
          The shape is :math:`(1, )`.
        - **lr** (Tensor) - :math:`l` in the updating formula. With float32 data type.
          The shape is :math:`(1, )`.
        - **beta1** (Tensor) - The exponential decay rate for the 1st moment estimations with float32 data type.
          The shape is :math:`(1, )`.
        - **beta2** (Tensor) - The exponential decay rate for the 2nd moment estimations with float32 data type.
          The shape is :math:`(1, )`.
        - **epsilon** (Tensor) - Term added to the denominator to improve numerical stability with float32 data type.
          The shape is :math:`(1, )`.
        - **gradient** (Tensor) - Gradient, has the same data type as `var` and
          gradient.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - Gradient indices with int32 data type and indices.shape[0] = gradient.shape[0].

    Outputs:
        Tuple of 3 Tensors, this operator will update the input parameters directly, the outputs are useless.

        - **var** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **m** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **v** (Tensor) - A Tensor with shape :math:`(1, )`.

    Raises:
        TypeError: If neither `use_locking` nor `use_neserov` is a bool.
        TypeError: If dtype of `var`, `m`, `v`, `beta1_power`, `beta2_power`, `lr`, `beta1`, `beta2`, `epsilon`,
                   `gradient` or `indices` is not float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_adam = ops.FusedSparseAdam()
        ...         self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="m")
        ...         self.v = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="v")
        ...     def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, indices):
        ...         out = self.sparse_apply_adam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1, beta2,
        ...                                      epsilon, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> beta1_power = Tensor(0.9, mindspore.float32)
        >>> beta2_power = Tensor(0.999, mindspore.float32)
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> beta1 = Tensor(0.9, mindspore.float32)
        >>> beta2 = Tensor(0.999, mindspore.float32)
        >>> epsilon = Tensor(1e-8, mindspore.float32)
        >>> gradient = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]]]), mindspore.float32)
        >>> indices = Tensor([0, 1], mindspore.int32)
        >>> output = net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient, indices)
        >>> print(net.var.asnumpy())
        [[[0.9997121  0.9997121 ]]
         [[0.9997121  0.9997121 ]]
         [[0.99971527 0.99971527]]]
    """
    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('m', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('v', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('beta1_power', dtype=sig.sig_dtype.T),
        sig.make_sig('beta2_power', dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T),
        sig.make_sig('beta1', dtype=sig.sig_dtype.T),
        sig.make_sig('beta2', dtype=sig.sig_dtype.T),
        sig.make_sig('epsilon', dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, use_locking=False, use_nesterov=False):
        """Initialize FusedSparseAdam."""
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.name)
        self.init_prim_io_names(inputs=['var', 'm', 'v', 'beta1_power', 'beta2_power', 'lr', 'beta1', 'beta2',
                                        'epsilon', 'grad', 'indices'],
                                outputs=['var', 'm', 'v'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape, indices_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        if len(var_shape) > 1 and grad_shape != indices_shape + var_shape[1:]:
            raise ValueError(f"For '{self.name}', the shape of updates should be [] or "
                             f"grad_shape = indices_shape + var_shape[1:], but got var_shape: {var_shape}, "
                             f"indices_shape: {indices_shape}, grad_shape: {grad_shape}.")
        return [1], [1], [1]

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, beta1_power_dtype, beta2_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype, indices_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        args = {"beta1_power": beta1_power_dtype, "beta2_power": beta2_power_dtype, 'lr': lr_dtype,
                "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float16, mstype.float32], self.name, True)
        validator.check_tensor_dtype_valid("indices_dtype", indices_dtype, [mstype.int32], self.name)
        return var_dtype, m_dtype, v_dtype


class FusedSparseLazyAdam(PrimitiveWithInfer):
    r"""
    Merges the duplicate value of the gradient and then updates parameters by the Adaptive Moment Estimation (LazyAdam)
    algorithm. This operator is used when the gradient is sparse. The behavior is not equivalent to the
    original Adam algorithm, as only the current indices parameters will be updated.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector, :math:`v` represents the 2nd moment vector, :math:`g` represents
    `gradient`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent `beta1_power` and
    `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `var`, :math:`\epsilon` represents
    `epsilon`.

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): Whether to enable a lock to protect variable tensors from being updated.
            If true, updates of the var, m, and v tensors will be protected by a lock.
            If false, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If true, update the gradients using NAG.
            If false, update the gradients without using NAG. Default: False.

    Inputs:
        - **var** (Parameter) - Parameters to be updated with float32 data type. The shape is :math:`(N, *)`
          where :math:`*` means, any number of additional dimensions.
        - **m** (Parameter) - The 1st moment vector in the updating formula, has the same shape and data type as `var`.
        - **v** (Parameter) - The 2nd moment vector in the updating formula, has the same shape and data type as `var`.
          Mean square gradients, has the same type as `var` with float32 data type.
        - **beta1_power** (Tensor) - :math:`beta_1^t` in the updating formula with float32 data type.
          The shape is :math:`(1, )`.
        - **beta2_power** (Tensor) - :math:`beta_2^t` in the updating formula with float32 data type.
          The shape is :math:`(1, )`.
        - **lr** (Tensor) - :math:`l` in the updating formula with float32 data type.
          The shape is :math:`(1, )`.
        - **beta1** (Tensor) - The exponential decay rate for the 1st moment estimations with float32 data type.
          The shape is :math:`(1, )`.
        - **beta2** (Tensor) - The exponential decay rate for the 2nd moment estimations with float32 data type.
          The shape is :math:`(1, )`.
        - **epsilon** (Tensor) - Term added to the denominator to improve numerical stability with float32 data type.
          The shape is :math:`(1, )`.
        - **gradient** (Tensor) - Gradient value with float32 data type and
          gradient.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - Gradient indices with int32 data type and indices.shape[0] = gradient.shape[0].

    Outputs:
        Tuple of 3 Tensors, this operator will update the input parameters directly, the outputs are useless.

        - **var** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **m** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **v** (Tensor) - A Tensor with shape :math:`(1, )`.

    Raises:
        TypeError: If neither `use_locking` nor `use_nestrov` is a bool.
        TypeError: If dtype of `var`, `m`, `v`, `beta1_power`, `beta2_power`, `lr`, `beta1`, `beta2`, `epsilon` or
                   gradient is not float32.
        TypeError: If dtype of `indices` is not int32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_lazyadam = ops.FusedSparseLazyAdam()
        ...         self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="m")
        ...         self.v = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="v")
        ...     def construct(self, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, indices):
        ...         out = self.sparse_apply_lazyadam(self.var, self.m, self.v, beta1_power, beta2_power, lr, beta1,
        ...                                          beta2, epsilon, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> beta1_power = Tensor(0.9, mindspore.float32)
        >>> beta2_power = Tensor(0.999, mindspore.float32)
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> beta1 = Tensor(0.9, mindspore.float32)
        >>> beta2 = Tensor(0.999, mindspore.float32)
        >>> epsilon = Tensor(1e-8, mindspore.float32)
        >>> gradient = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]]]), mindspore.float32)
        >>> indices = Tensor([0, 1], mindspore.int32)
        >>> output = net(beta1_power, beta2_power, lr, beta1, beta2, epsilon, gradient, indices)
        >>> print(net.var.asnumpy())
        [[[0.9997121  0.9997121 ]]
         [[0.9997121  0.9997121 ]]
         [[1.         1.        ]]]
    """
    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('m', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('v', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('beta1_power', dtype=sig.sig_dtype.T),
        sig.make_sig('beta2_power', dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T),
        sig.make_sig('beta1', dtype=sig.sig_dtype.T),
        sig.make_sig('beta2', dtype=sig.sig_dtype.T),
        sig.make_sig('epsilon', dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, use_locking=False, use_nesterov=False):
        """Initialize FusedSparseLazyAdam."""
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.name)
        self.init_prim_io_names(inputs=['var', 'm', 'v', 'beta1_power', 'beta2_power', 'lr', 'beta1', 'beta2',
                                        'epsilon', 'grad', 'indices'],
                                outputs=['var', 'm', 'v'])
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, v_shape, beta1_power_shape, beta2_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape, indices_shape):
        validator.check("var_shape", var_shape, "m_shape", m_shape, Rel.EQ, self.name)
        validator.check("var_shape", var_shape, "v_shape", v_shape, Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        if len(var_shape) > 1 and grad_shape != indices_shape + var_shape[1:]:
            raise ValueError(f"For '{self.name}', the shape of updates should be [] or "
                             f"grad_shape = indices_shape + var_shape[1:], but got var_shape: {var_shape}, "
                             f"indices_shape: {indices_shape}, grad_shape: {grad_shape}.")
        return [1], [1], [1]

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, beta1_power_dtype, beta2_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype, indices_dtype):
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)

        args = {"beta1_power": beta1_power_dtype, "beta2_power": beta2_power_dtype, 'lr': lr_dtype,
                "beta1": beta1_dtype, "beta2": beta2_dtype, "epsilon": epsilon_dtype}
        validator.check_scalar_or_tensor_types_same(args, [mstype.float16, mstype.float32], self.name, True)

        validator.check_tensor_dtype_valid("indices_dtype", indices_dtype, [mstype.int32], self.name)
        return var_dtype, m_dtype, v_dtype


class FusedSparseFtrl(PrimitiveWithInfer):
    """
    Merges the duplicate value of the gradient and then updates relevant entries according to the FTRL-proximal scheme.

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        lr (float): The learning rate value, must be positive.
        l1 (float): l1 regularization strength, must be greater than or equal to zero.
        l2 (float): l2 regularization strength, must be greater than or equal to zero.
        lr_power (float): Learning rate power controls how the learning rate decreases during training,
            must be less than or equal to zero. Use fixed learning rate if `lr_power` is zero.
        use_locking (bool): Use locks for updating operation if true . Default: False.

    Inputs:
        - **var** (Parameter) - The variable to be updated. The data type must be float32. The shape is :math:`(N, *)`
          where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - The accumulation to be updated, must be same type and shape as `var`.
        - **linear** (Parameter) - the linear coefficient to be updated, must be same type and shape as `var`.
        - **grad** (Tensor) - A tensor of the same type as `var` and
          grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A vector of indices into the first dimension of `var` and `accum`.
          The type must be int32 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 3 Tensor, this operator will update the input parameters directly, the outputs are useless.

        - **var** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **accum** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **linear** (Tensor) - A Tensor with shape :math:`(1, )`.

    Raises:
        TypeError: If `lr`, `l1`, `l2` or `lr_power` is not a float.
        ValueError: If shape of `lr_power` less than or equal to zero.
        TypeError: If dtype of `var` is not float32.
        TypeError: If dtype of `indices` is not int32.
        TypeError: If shape of `accum`, `linear` or `grad` is not same as `var`.
        TypeError: If shape of `indices` is not same as shape of first dimension of `grad`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> class SparseApplyFtrlNet(nn.Cell):
        ...     def __init__(self):
        ...         super(SparseApplyFtrlNet, self).__init__()
        ...         self.sparse_apply_ftrl = ops.FusedSparseFtrl(lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
        ...         self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="accum")
        ...         self.linear = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="linear")
        ...
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        ...         return out
        ...
        >>> net = SparseApplyFtrlNet()
        >>> grad = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]]]).astype(np.float32))
        >>> indices = Tensor(np.array([0, 1]).astype(np.int32))
        >>> output = net(grad, indices)
        >>> print(net.var.asnumpy())
        [[[-0.00598256 -0.00598256]]
         [[-0.00598256 -0.00598256]]
         [[ 1.          1.        ]]]
    """
    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('linear', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, lr, l1, l2, lr_power, use_locking=False):
        """Initialize FusedSparseFtrl."""
        self.init_prim_io_names(inputs=['var', 'accum', 'linear', 'grad', 'indices'],
                                outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

        validator.check_value_type("lr", lr, [float], self.name)
        validator.check_value_type("l1", l1, [float], self.name)
        validator.check_value_type("l2", l2, [float], self.name)
        validator.check_value_type("lr_power", lr_power, [float], self.name)
        self.lr = validator.check_positive_float(lr, "lr", self.name)
        self.l1 = validator.check_non_negative_float(l1, "l1", self.name)
        self.l2 = validator.check_non_negative_float(l2, "l2", self.name)
        self.lr_power = validator.check_number("lr_power", lr_power, 0, Rel.LE, self.name)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, accum_shape, linear_shape, grad_shape, indices_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('var shape', var_shape, 'linear shape', linear_shape, Rel.EQ, self.name)
        if len(var_shape) > 1:
            validator.check('var_shape[1:]', var_shape[1:], 'grad_shape[1:]', grad_shape[1:], Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        return [1], [1], [1]

    def infer_dtype(self, var_dtype, accum_dtype, linear_dtype, grad_dtype, indices_dtype):
        args = {"var_dtype": var_dtype, "accum_dtype": accum_dtype,
                "linear_dtype": linear_dtype, "grad_dtype": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float32], self.name)
        validator.check_tensor_dtype_valid("indices_dtype", indices_dtype, [mstype.int32], self.name)
        return var_dtype, accum_dtype, linear_dtype


class FusedSparseProximalAdagrad(PrimitiveWithInfer):
    r"""
    Merges the duplicate value of the gradient and then updates relevant entries according to the proximal adagrad
    algorithm.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            \text{prox_v} = var - lr * grad * \frac{1}{\sqrt{accum}} \\
            var = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)
        \end{array}

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): If true, the variable and accumulation tensors will be protected from being updated.
            Default: False.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. The data type must be float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Variable tensor to be updated, has the same shape and data type as `var`.
        - **lr** (Tensor) - The learning rate value. The data type must be float32. The shape is :math:`(1, )`.
        - **l1** (Tensor) - l1 regularization strength. The data type must be float32. The shape is :math:`(1, )`.
        - **l2** (Tensor) - l2 regularization strength. The data type must be float32. The shape is :math:`(1, )`.
        - **grad** (Tensor) - A tensor of the same data type as `var` and
          grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A vector of indices into the first dimension of `var` and `accum`.
          The type must be int32 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 2 Tensors, this operator will update the input parameters directly, the outputs are useless.

        - **var** (Tensor) - A Tensor with shape :math:`(1, )`.
        - **accum** (Tensor) - A Tensor with shape :math:`(1, )`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `var`, `accum`, `lr`, `l1`, `l2` or `grad` is not float32.
        TypeError: If dtype of `indices` is not int32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_proximal_adagrad = ops.FusedSparseProximalAdagrad()
        ...         self.var = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="accum")
        ...         self.lr = Tensor(0.01, mindspore.float32)
        ...         self.l1 = Tensor(0.0, mindspore.float32)
        ...         self.l2 = Tensor(0.0, mindspore.float32)
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1,
        ...                                                  self.l2, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[[0.1, 0.1]], [[0.1, 0.1]]]).astype(np.float32))
        >>> indices = Tensor(np.array([0, 1]).astype(np.int32))
        >>> output = net(grad, indices)
        >>> print(net.var.asnumpy())
        [[[0.99900496 0.99900496]]
         [[0.99900496 0.99900496]]
         [[1.         1.        ]]]
    """
    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T),
        sig.make_sig('l1', dtype=sig.sig_dtype.T),
        sig.make_sig('l2', dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize FusedSparseProximalAdagrad"""
        self.init_prim_io_names(inputs=['var', 'accum', 'lr', 'l1', 'l2', 'grad', 'indices'],
                                outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, accum_shape, lr_shape, l1_shape, l2_shape,
                    grad_shape, indices_shape):
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        return [1], [1]

    def infer_dtype(self, var_dtype, accum_dtype, lr_dtype, l1_dtype, l2_dtype,
                    grad_dtype, indices_dtype):
        args = {'var': var_dtype, 'accum': accum_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, [mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"l1": l1_dtype}, [mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"l2": l2_dtype}, [mstype.float32], self.name)
        valid_dtypes = [mstype.int16, mstype.int32, mstype.int64,
                        mstype.uint16, mstype.uint32, mstype.uint64]
        validator.check_tensor_dtype_valid('indices', indices_dtype, valid_dtypes, self.name)
        return var_dtype, accum_dtype


class KLDivLoss(PrimitiveWithInfer):
    r"""
    Computes the Kullback-Leibler divergence between the logits and the labels.

    The updating formulas of KLDivLoss algorithm are as follows,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = y_n \cdot (\log y_n - x_n)

    Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`x` represents `logits`.
    :math:`y` represents `labels`.
    :math:`\ell(x, y)` represents `output`.

    Args:
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'mean'.

    Inputs:
        - **logits** (Tensor) - The input Tensor. The data type must be float32.
        - **labels** (Tensor) - The label Tensor which has the same shape and data type as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `logits`.
        Otherwise it is a scalar.

    Raises:
        TypeError: If `reduction` is not a str.
        TypeError: If neither `logits` nor `labels` is a Tensor.
        TypeError: If dtype of `logits` or `labels` is not float32.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.kldiv_loss = ops.KLDivLoss()
        ...     def construct(self, logits, labels):
        ...         result = self.kldiv_loss(logits, labels)
        ...         return result
        ...
        >>> net = Net()
        >>> logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> output = net(logits, labels)
        >>> print(output)
        -0.23333333
    """

    @prim_attr_register
    def __init__(self, reduction='mean'):
        """Initialize KLDivLoss."""
        self.reduction = validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', self.name)

    def infer_shape(self, x_shape, y_shape):
        validator.check('x_shape', x_shape, 'y_shape', y_shape, Rel.EQ, self.name)
        if self.reduction in ('mean', 'sum'):
            shape = []
        else:
            shape = x_shape
        return shape

    def infer_dtype(self, x_type, y_type):
        args = {'x': x_type, 'y': y_type}
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        return x_type


class BinaryCrossEntropy(PrimitiveWithInfer):
    r"""
    Computes the binary cross entropy between the logits and the labels.

    Sets logits as :math:`x`, labels as :math:`y`, output as :math:`\ell(x, y)`.
    Let,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    In which, :math:`L` indicates the loss of all batch_sizes, :math:`l` indicates the loss of one batch_size,
    and n indicates one batch_size in the 1-N range. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. warning::
        - The value of "x" must range from 0 to 1.
        - The value of "y" must be "0" or "1".

    Args:
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'mean'.

    Inputs:
        - **logits** (Tensor) - The input Tensor. The data type must be float16 or float32,
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **labels** (Tensor) - The label Tensor which has same shape and data type as `logits`.
        - **weight** (Tensor, optional) - A rescaling weight applied to the loss of each batch element.
          And it must have same shape and data type as `logits`. Default: None.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `logits`.
        Otherwise, the output is a scalar.

    Raises:
        TypeError: If dtype of `logits`, `labels` or `weight` (if given) is neither float16 not float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If shape of `labels` is not the same as `logits` or `weight` (if given).
        TypeError: If `logits`, `labels` or `weight` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.binary_cross_entropy = ops.BinaryCrossEntropy()
        ...     def construct(self, logits, labels, weight):
        ...         result = self.binary_cross_entropy(logits, labels, weight)
        ...         return result
        ...
        >>> net = Net()
        >>> logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> weight = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = net(logits, labels, weight)
        >>> print(output)
        0.38240486
    """

    @prim_attr_register
    def __init__(self, reduction='mean'):
        """Initialize BinaryCrossEntropy."""
        self.reduction = validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', self.name)

    def infer_shape(self, x_shape, y_shape, weight_shape):
        validator.check('x_shape', x_shape, 'y_shape', y_shape, Rel.EQ, self.name)
        if weight_shape:
            validator.check('y_shape', y_shape, 'weight_shape', weight_shape, Rel.EQ, self.name)
        if self.reduction in ('mean', 'sum'):
            shape = []
        else:
            shape = x_shape
        return shape

    def infer_dtype(self, x_type, y_type, weight_type):
        args = {'x': x_type, 'y': y_type}
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        if weight_type:
            validator.check_tensors_dtypes_same_and_valid({'x': x_type, 'weight': weight_type}, valid_dtypes,
                                                          self.name)
        return x_type


class ApplyAdaMax(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adamax scheme.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta_1 * m_{t} + (1 - \beta_1) * g \\
            v_{t+1} = \max(\beta_2 * v_{t}, \left| g \right|) \\
            var = var - \frac{l}{1 - \beta_1^{t+1}} * \frac{m_{t+1}}{v_{t+1} + \epsilon}
        \end{array}

    :math:`t` represents updating step while :math:`m` represents the 1st moment vector, :math:`m_{t}`
    is the last momentent of :math:`m_{t+1}`, :math:`v` represents the 2nd moment vector, :math:`v_{t}`
    is the last momentent of :math:`v_{t+1}`, :math:`l` represents scaling factor `lr`,
    :math:`g` represents `grad`, :math:`\beta_1, \beta_2` represent `beta1` and `beta2`,
    :math:`beta_1^{t+1}` represents `beta1_power`, :math:`var` represents the variable to be updated,
    :math:`\epsilon` represents `epsilon`.

    Inputs of `var`, `m`, `v` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Variable to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **m** (Parameter) - The 1st moment vector in the updating formula, has the same shape and type as `var`.
          With float32 or float16 data type.
        - **v** (Parameter) - The 2nd moment vector in the updating formula. Mean square gradients
          with the same shape and type as `var`. With float32 or float16 data type.
        - **beta1_power** (Union[Number, Tensor]) - :math:`beta_1^t` in the updating formula, must be scalar.
          With float32 or float16 data type.
        - **lr** (Union[Number, Tensor]) - Learning rate, :math:`l` in the updating formula, must be scalar.
          With float32 or float16 data type.
        - **beta1** (Union[Number, Tensor]) - The exponential decay rate for the 1st moment estimations,
          must be scalar. With float32 or float16 data type.
        - **beta2** (Union[Number, Tensor]) - The exponential decay rate for the 2nd moment estimations,
          must be scalar. With float32 or float16 data type.
        - **epsilon** (Union[Number, Tensor]) - A small value added for numerical stability, must be scalar.
          With float32 or float16 data type.
        - **grad** (Tensor) - A tensor for gradient, has the same shape and type as `var`.
          With float32 or float16 data type.

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.
        - **v** (Tensor) - The same shape and data type as `v`.

    Raises:
        TypeError: If dtype of `var`, `m`, `v`, `beta_power`, `lr`, `beta1`, `beta2`, `epsilon` or `grad` is neither
                   float16 nor float32.
        TypeError: If `beta_power`, `lr`, `beta1`, `beta2` or `epsilon` is neither a Number nor a Tensor.
        TypeError: If `grad` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_ada_max = ops.ApplyAdaMax()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                             [0.2, 0.6]]).astype(np.float32)), name="m")
        ...         self.v = Parameter(Tensor(np.array([[0.9, 0.1],
        ...                                             [0.7, 0.8]]).astype(np.float32)), name="v")
        ...     def construct(self, beta1_power, lr, beta1, beta2, epsilon, grad):
        ...         out = self.apply_ada_max(self.var, self.m, self.v, beta1_power, lr, beta1, beta2, epsilon, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> beta1_power =Tensor(0.9, mindspore.float32)
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> beta1 = Tensor(0.9, mindspore.float32)
        >>> beta2 = Tensor(0.99, mindspore.float32)
        >>> epsilon = Tensor(1e-10, mindspore.float32)
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(beta1_power, lr, beta1, beta2, epsilon, grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.93602717e-01,  3.92571449e-01],
         [ 9.72582996e-02,  4.92249995e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.69999993e-01,  5.19999981e-01],
         [ 1.89999998e-01,  6.20000005e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 8.90999973e-01,  6.99999988e-01],
         [ 6.93000019e-01,  8.00000012e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('m', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('v', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('beta1_power', dtype=sig.sig_dtype.T1),
        sig.make_sig('lr', dtype=sig.sig_dtype.T2),
        sig.make_sig('beta1', dtype=sig.sig_dtype.T3),
        sig.make_sig('beta2', dtype=sig.sig_dtype.T4),
        sig.make_sig('epsilon', dtype=sig.sig_dtype.T5),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyAdaMax"""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, v_shape, beta1_power_shape, lr_shape,
                    beta1_shape, beta2_shape, epsilon_shape, grad_shape):
        validator.check("m_shape", m_shape, "var_shape", var_shape, Rel.EQ, self.name)
        validator.check("v_shape", v_shape, "var_shape", var_shape, Rel.EQ, self.name)
        validator.check("grad_shape", grad_shape, "var_shape", var_shape, Rel.EQ, self.name)
        beta1_power_shp_len = len(beta1_power_shape)
        validator.check_int(beta1_power_shp_len, 1, Rel.LE, "beta1 power's rank", self.name)
        if beta1_power_shp_len == 1:
            validator.check_int(beta1_power_shape[0], 1, Rel.EQ, "beta1_power_shape[0]", self.name)
        lr_shp_len = len(lr_shape)
        validator.check_int(lr_shp_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shp_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        beta1_shp_len = len(beta1_shape)
        validator.check_int(beta1_shp_len, 1, Rel.LE, "beta1's rank", self.name)
        if beta1_shp_len == 1:
            validator.check_int(beta1_shape[0], 1, Rel.EQ, "beta1_shape[0]", self.name)
        beta2_shp_len = len(beta2_shape)
        validator.check_int(beta2_shp_len, 1, Rel.LE, "beta2's rank", self.name)
        if beta2_shp_len == 1:
            validator.check_int(beta2_shape[0], 1, Rel.EQ, "beta2_shape[0]", self.name)
        epsilon_shp_len = len(epsilon_shape)
        validator.check_int(epsilon_shp_len, 1, Rel.LE, "epsilon's rank", self.name)
        if epsilon_shp_len == 1:
            validator.check_int(epsilon_shape[0], 1, Rel.EQ, "epsilon_shape[0]", self.name)
        return var_shape, m_shape, v_shape

    def infer_dtype(self, var_dtype, m_dtype, v_dtype, beta1_power_dtype, lr_dtype,
                    beta1_dtype, beta2_dtype, epsilon_dtype, grad_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {"var": var_dtype, "m": m_dtype, "v": v_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"beta1_power": beta1_power_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"beta1": beta1_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"beta2": beta2_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"epsilon": epsilon_dtype}, valid_dtypes, self.name)
        return var_dtype, m_dtype, v_dtype


class ApplyAdadelta(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adadelta scheme.

    .. math::
        \begin{array}{ll} \\
            accum = \rho * accum + (1 - \rho) * grad^2 \\
            \text{update} = \sqrt{\text{accum_update} + \epsilon} * \frac{grad}{\sqrt{accum + \epsilon}} \\
            \text{accum_update} = \rho * \text{accum_update} + (1 - \rho) * update^2 \\
            var -= lr * update
        \end{array}

    where :math:`\rho` represents `rho`, :math:`\epsilon` represents `epsilon`.

    Inputs of `var`, `accum`, `accum_update` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Weights to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated, has the same shape and data type as `var`.
        - **accum_update** (Parameter) - Accum_update to be updated, has the same shape and data type as `var`.
        - **lr** (Union[Number, Tensor]) - Learning rate, must be scalar. With float32 or float16 data type.
        - **rho** (Union[Number, Tensor]) - Decay rate, must be scalar. With float32 or float16 data type.
        - **epsilon** (Union[Number, Tensor]) - A small value added for numerical stability, must be scalar.
          With float32 or float16 data type.
        - **grad** (Tensor) - Gradients, has the same shape and data type as `var`.

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.
        - **accum_update** (Tensor) - The same shape and data type as `accum_update`.

    Raises:
        TypeError: If dtype of `var`, `accum`, `accum_update`, `lr`, `rho`, `epsilon` or `grad` is neither float16 nor
                   float32.
        TypeError: If `accum_update`, `lr`, `rho` or `epsilon` is neither a Number nor a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_adadelta = ops.ApplyAdadelta()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                                 [0.2, 0.6]]).astype(np.float32)), name="accum")
        ...         self.accum_update = Parameter(Tensor(np.array([[0.9, 0.1],
        ...                                                        [0.7, 0.8]]).astype(np.float32)),
        ...                                                             name="accum_update")
        ...     def construct(self, lr, rho, epsilon, grad):
        ...         out = self.apply_adadelta(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> rho = Tensor(0.0, mindspore.float32)
        >>> epsilon = Tensor(1e-6, mindspore.float32)
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(lr, rho, epsilon, grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.99051356e-01,  3.99683774e-01],
         [ 9.91633832e-02,  4.99105573e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 9.00000036e-02,  4.89999980e-01],
         [ 1.00000007e-02,  6.40000045e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 8.99990976e-01,  1.00000791e-01],
         [ 6.99930906e-01,  7.99999654e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum_update', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('rho', dtype=sig.sig_dtype.T2),
        sig.make_sig('epsilon', dtype=sig.sig_dtype.T3),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyAdadelta"""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, accum_update_shape, lr_shape, rho_shape,
                    epsilon_shape, grad_shape):
        validator.check("accum_shape", accum_shape, "var_shape", var_shape, Rel.EQ, self.name)
        validator.check("accum_update_shape", accum_update_shape, "var_shape", var_shape, Rel.EQ, self.name)
        validator.check("grad_shape", grad_shape, "var_shape", var_shape, Rel.EQ, self.name)
        lr_shp_len = len(lr_shape)
        validator.check_int(lr_shp_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shp_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        rho_shp_len = len(rho_shape)
        validator.check_int(rho_shp_len, 1, Rel.LE, "rho's rank", self.name)
        if rho_shp_len == 1:
            validator.check_int(rho_shape[0], 1, Rel.EQ, "rho_shape[0]", self.name)
        epsilon_shp_len = len(epsilon_shape)
        validator.check_int(epsilon_shp_len, 1, Rel.LE, "lepsilon's rank", self.name)
        if epsilon_shp_len == 1:
            validator.check_int(epsilon_shape[0], 1, Rel.EQ, "epsilon_shape[0]", self.name)
        return var_shape, accum_shape, accum_update_shape

    def infer_dtype(self, var_dtype, accum_dtype, accum_update_dtype, lr_dtype, rho_dtype,
                    epsilon_dtype, grad_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {"var": var_dtype, "accum": accum_dtype, "accum_update": accum_update_dtype, "grad": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"rho": rho_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"epsilon": epsilon_dtype}, valid_dtypes, self.name)
        return var_dtype, accum_dtype, accum_update_dtype


class ApplyAdagrad(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adagrad scheme.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum}}
        \end{array}

    Inputs of `var`, `accum` and `grad`  comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        update_slots (bool): If `True`, `accum` will be updated. Default: True.

    Inputs:
        - **var** (Parameter) - Variable to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated. The shape and data type must be the same as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be scalar. With float32 or float16 data type.
        - **grad** (Tensor) - A tensor for gradient. The shape and data type must be the same as `var`.

    Outputs:
        Tuple of 2 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.

    Raises:
        TypeError: If dtype of `var`, `accum`, `lr` or `grad` is neither float16 nor float32.
        TypeError: If `lr` is neither a Number nor a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_adagrad = ops.ApplyAdagrad()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                                 [0.2, 0.6]]).astype(np.float32)), name="accum")
        ...     def construct(self, lr, grad):
        ...         out = self.apply_adagrad(self.var, self.accum, lr, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(lr, grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.99638879e-01,  3.99296492e-01],
         [ 9.97817814e-02,  4.99281585e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 6.90000057e-01,  9.90000010e-01],
         [ 2.10000008e-01,  1.24000001e+00]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, update_slots=True):
        """Initialize ApplyAdagrad."""
        validator.check_value_type("update_slots", update_slots, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, lr_shape, grad_shape):
        validator.check('accum shape', accum_shape, 'var shape', var_shape, Rel.EQ, self.name)
        validator.check('grad shape', grad_shape, 'var shape', var_shape, Rel.EQ, self.name)
        lr_shp_len = len(lr_shape)
        validator.check_int(lr_shp_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shp_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        return var_shape, accum_shape

    def infer_dtype(self, var_dtype, accum_dtype, lr_dtype, grad_dtype):
        args = {'var': var_dtype, 'accum': accum_dtype, 'grad': grad_dtype}
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({'lr': lr_dtype}, valid_dtypes, self.name)
        return var_dtype, accum_dtype


class ApplyAdagradV2(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adagradv2 scheme.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum} + \epsilon}
        \end{array}

    where :math:`\epsilon` represents `epsilon`.

    Inputs of `var`, `accum` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Note:
        The difference is that `ApplyAdagradV2` has one more small constant value than `ApplyAdagrad`.

    Args:
        epsilon (float): A small value added for numerical stability.
        update_slots (bool): If `True`, `accum` will be updated. Default: True.

    Inputs:
        - **var** (Parameter) - Variable to be updated. With float16 or float32 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated. The shape and data type must be the same as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be a float number or
          a scalar tensor with float16 or float32 data type.
        - **grad** (Tensor) - A tensor for gradient. The shape and data type must be the same as `var`.

    Outputs:
        Tuple of 2 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `m`.

    Raises:
        TypeError: If dtype of `var`, `accum`, `lr` or `grad` is neither float16 nor float32.
        TypeError: If `lr` is neither a Number nor a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_adagrad_v2 = ops.ApplyAdagradV2(epsilon=1e-6)
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                                 [0.2, 0.6]]).astype(np.float32)), name="accum")
        ...     def construct(self, lr, grad):
        ...         out = self.apply_adagrad_v2(self.var, self.accum, lr, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> lr = Tensor(0.001, mindspore.float32)
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(lr, grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.99638879e-01,  3.99296492e-01],
         [ 9.97817814e-02,  4.99281585e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 6.90000057e-01,  9.90000010e-01],
         [ 2.10000008e-01,  1.24000001e+00]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, epsilon, update_slots=True):
        """Initialize ApplyAdagradV2."""
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_value_type("update_slots", update_slots, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, lr_shape, grad_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('var shape', var_shape, 'grad shape', grad_shape, Rel.EQ, self.name)
        lr_shp_len = len(lr_shape)
        validator.check_int(lr_shp_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shp_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        return var_shape, accum_shape

    def infer_dtype(self, var_dtype, accum_dtype, lr_dtype, grad_dtype):
        args = {'var': var_dtype, 'accum': accum_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({'lr': lr_dtype}, [mstype.float16, mstype.float32], self.name)
        return var_dtype, accum_dtype


class SparseApplyAdagrad(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adagrad scheme.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * (1 / sqrt(accum))
        \end{array}

    Inputs of `var`, `accum` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        lr (float): Learning rate.
        update_slots (bool): If `True`, `accum` will be updated. Default: True.
        use_locking (bool): If true, the `var` and `accum` tensors will be protected from being updated.
            Default: False.

    Inputs:
        - **var** (Parameter) - Variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated. The shape and data type must be the same as `var`.
        - **grad** (Tensor) - Gradients has the same data type as `var` and
          grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A vector of indices into the first dimension of `var` and `accum`.
          The type must be int32 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 2 tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.

    Raises:
        TypeError: If `lr` is not a float.
        TypeError: If neither `update_slots` nor `use_locking` is a bool.
        TypeError: If dtype of `var`, `accum` or `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is not int32.


    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_adagrad = ops.SparseApplyAdagrad(lr=1e-8)
        ...         self.var = Parameter(Tensor(np.array([[[0.2]]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[[0.1]]]).astype(np.float32)), name="accum")
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_adagrad(self.var, self.accum, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[[0.7]]]).astype(np.float32))
        >>> indices = Tensor([0], mindspore.int32)
        >>> output = net(grad, indices)
        >>> print(output)
        (Tensor(shape=[1, 1, 1], dtype=Float32, value=
        [[[1.99999988e-01]]]), Tensor(shape=[1, 1, 1], dtype=Float32, value=
        [[[1.00000001e-01]]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, lr, update_slots=True, use_locking=False):
        """Initialize SparseApplyAdagrad."""
        validator.check_is_float(lr, "lr", self.name)
        validator.check_value_type("update_slots", update_slots, [bool], self.name)
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, grad_shape, indices_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('len of var shape', len(var_shape), 'len of grad shape', len(grad_shape), Rel.EQ, self.name)
        if len(var_shape) > 1:
            validator.check('var_shape[1:]', var_shape[1:], 'grad_shape[1:]', grad_shape[1:], Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        return var_shape, accum_shape

    def infer_dtype(self, var_type, accum_type, grad_type, indices_type):
        args = {'var': var_type, 'accum': accum_type, 'grad': grad_type}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('indices', indices_type, [mstype.int32], self.name)
        return var_type, accum_type


class SparseApplyAdagradV2(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the adagrad scheme, one more epsilon attribute than SparseApplyAdagrad.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum} + \epsilon}
        \end{array}

    where :math:`\epsilon` represents `epsilon`.

    Inputs of `var`, `accum` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        lr (float): Learning rate.
        epsilon (float): A small value added for numerical stability.
        use_locking (bool): If `True`, the `var` and `accum` tensors will be protected from being updated.
            Default: False.
        update_slots (bool): If `True`, the computation logic will be different to `False`. Default: True.

    Inputs:
        - **var** (Parameter) - Variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated. The shape and data type must be the same as `var`.
        - **grad** (Tensor) - Gradients has the same data type as `var` and
          grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A vector of indices into the first dimension of `var` and `accum`.
          The type must be int32 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 2 tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.

    Raises:
        TypeError: If neither `lr` nor `epsilon` is a float.
        TypeError: If neither `update_slots` nor `use_locking` is a bool.
        TypeError: If dtype of `var`, `accum` or `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is not int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_adagrad_v2 = ops.SparseApplyAdagradV2(lr=1e-8, epsilon=1e-6)
        ...         self.var = Parameter(Tensor(np.array([[0.2]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.1]]).astype(np.float32)), name="accum")
        ...
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_adagrad_v2(self.var, self.accum, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[0.7]]).astype(np.float32))
        >>> indices = Tensor(np.ones([1]), mindspore.int32)
        >>> output = net(grad, indices)
        >>> print(output)
        (Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 2.00000003e-01]]), Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 1.00000001e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, lr, epsilon, use_locking=False, update_slots=True):
        """Initialize SparseApplyAdagradV2."""
        self.lr = validator.check_value_type("lr", lr, [float], self.name)
        self.epsilon = validator.check_value_type("epsilon", epsilon, [float], self.name)
        self.use_locking = validator.check_value_type("update_slots", update_slots, [bool], self.name)
        self.update_slots = validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, grad_shape, indices_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('len of var shape', len(var_shape), 'len of grad shape', len(grad_shape), Rel.EQ, self.name)
        if len(var_shape) > 1:
            validator.check('var_shape[1:]', var_shape[1:], 'grad_shape[1:]', grad_shape[1:], Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        return var_shape, accum_shape

    def infer_dtype(self, var_type, accum_type, grad_type, indices_type):
        args = {'var': var_type, 'accum': accum_type, 'grad': grad_type}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid('indices', indices_type, [mstype.int32], self.name)
        return var_type, accum_type


class ApplyProximalAdagrad(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the proximal adagrad algorithm.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            \text{prox_v} = var - lr * grad * \frac{1}{\sqrt{accum}} \\
            var = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)
        \end{array}

    Inputs of `var`, `accum` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): If true, the var and accumulation tensors will be protected from being updated.
            Default: False.

    Inputs:
        - **var** (Parameter) - Variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Accumulation to be updated. Must has the same shape and dtype as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be scalar. The data type must be
          float16 or float32.
        - **l1** (Union[Number, Tensor]) - l1 regularization strength, must be scalar. The data type must be
          float16 or float32.
        - **l2** (Union[Number, Tensor]) - l2 regularization strength, must be scalar. The data type must be
          float16 or float32.
        - **grad** (Tensor) - Gradient with the same shape and dtype as `var`.

    Outputs:
        Tuple of 2 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.

    Raises:
        TypeError: If `use_blocking` is not a bool.
        TypeError: If dtype of `var`, `lr`, `l1` or `l2` is neither float16 nor float32.
        TypeError: If `lr`, `l1` or `l2` is neither a Number nor a Tensor.
        TypeError: If `grad` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_proximal_adagrad = ops.ApplyProximalAdagrad()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                                 [0.2, 0.6]]).astype(np.float32)), name="accum")
        ...         self.lr = 0.01
        ...         self.l1 = 0.0
        ...         self.l2 = 0.0
        ...     def construct(self, grad):
        ...         out = self.apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.96388459e-01,  3.92964751e-01],
         [ 9.78178233e-02,  4.92815793e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 6.90000057e-01,  9.90000010e-01],
         [ 2.10000008e-01,  1.24000001e+00]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('l1', dtype=sig.sig_dtype.T2),
        sig.make_sig('l2', dtype=sig.sig_dtype.T3),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ApplyProximalAdagrad."""
        self.init_prim_io_names(inputs=['var', 'accum', 'lr', 'l1', 'l2', 'grad'],
                                outputs=['var', 'accum'])
        self.add_prim_attr('side_effect_mem', True)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, accum_shape, lr_shape, l1_shape, l2_shape, grad_shape):
        validator.check('accum shape', accum_shape, 'var shape', var_shape, Rel.EQ, self.name)
        validator.check('grad shape', grad_shape, 'var shape', var_shape, Rel.EQ, self.name)
        lr_shp_len = len(lr_shape)
        validator.check_int(lr_shp_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shp_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        l1_shp_len = len(l1_shape)
        validator.check_int(l1_shp_len, 1, Rel.LE, "l1's rank", self.name)
        if l1_shp_len == 1:
            validator.check_int(l1_shape[0], 1, Rel.EQ, "l1_shape[0]", self.name)
        l2_shp_len = len(l2_shape)
        validator.check_int(l2_shp_len, 1, Rel.LE, "l2's rank", self.name)
        if l2_shp_len == 1:
            validator.check_int(l2_shape[0], 1, Rel.EQ, "l2_shape[0]", self.name)
        return var_shape, accum_shape

    def infer_dtype(self, var_dtype, accum_dtype, lr_dtype, l1_dtype, l2_dtype, grad_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_dtype, 'accum': accum_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l1": l1_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l2": l2_dtype}, valid_dtypes, self.name)
        return var_dtype, accum_dtype


class SparseApplyProximalAdagrad(PrimitiveWithCheck):
    r"""
    Updates relevant entries according to the proximal adagrad algorithm. Compared with ApplyProximalAdagrad,
    an additional index tensor is input.

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            \text{prox_v} = var - lr * grad * \frac{1}{\sqrt{accum}} \\
            var = \frac{sign(\text{prox_v})}{1 + lr * l2} * \max(\left| \text{prox_v} \right| - lr * l1, 0)
        \end{array}

    Inputs of `var`, `accum` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): If true, the `var` and `accum` tensors will be protected from being updated.
            Default: False.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - Variable tensor to be updated, has the same shape and dtype as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be a float number or
          a scalar tensor with float16 or float32 data type.
        - **l1** (Union[Number, Tensor]) - l1 regularization strength, must be a float number or
          a scalar tensor with float16 or float32 data type.
        - **l2** (Union[Number, Tensor]) - l2 regularization strength, must be a float number or
          a scalar tensor with float16 or float32 data type..
        - **grad** (Tensor) - A tensor of the same type as `var` and
          grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A tensor of indices in the first dimension of `var` and `accum`.
          If there are duplicates in `indices`, the behavior is undefined. Must be one of the
          following types: int32, int64 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 2 tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **accum** (Tensor) - The same shape and data type as `accum`.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `var`, `accum`, `lr`, `l1`, `l2`, `scalar` or `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is neither int32 nor int64.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.sparse_apply_proximal_adagrad = ops.SparseApplyProximalAdagrad()
        ...         self.var = Parameter(Tensor(np.array([[4.1, 7.2], [1.1, 3.0]], np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0, 0], [0, 0]], np.float32)), name="accum")
        ...         self.lr = 1.0
        ...         self.l1 = 1.0
        ...         self.l2 = 0.0
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1,
        ...                                                  self.l2, grad, indices)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[1, 1], [1, 1]], np.float32))
        >>> indices = Tensor(np.array([0, 1], np.int32))
        >>> output = net(grad, indices)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.09999990e+00,  5.19999981e+00],
         [ 0.00000000e+00,  1.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.00000000e+00,  1.00000000e+00],
         [ 1.00000000e+00,  1.00000000e+00]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('l1', dtype=sig.sig_dtype.T2),
        sig.make_sig('l2', dtype=sig.sig_dtype.T3),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T4)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize SparseApplyProximalAdagrad."""
        self.init_prim_io_names(inputs=['var', 'accum', 'lr', 'l1', 'l2', 'grad', 'indices'],
                                outputs=['var', 'accum'])
        self.add_prim_attr('side_effect_mem', True)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def check_shape(self, var_shape, accum_shape, lr_shape, l1_shape, l2_shape,
                    grad_shape, indices_shape):
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)

    def check_dtype(self, var_dtype, accum_dtype, lr_dtype, l1_dtype, l2_dtype,
                    grad_dtype, indices_dtype):
        args = {'var': var_dtype, 'accum': accum_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, [mstype.float16, mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"l1": l1_dtype}, [mstype.float16, mstype.float32], self.name)
        validator.check_scalar_or_tensor_types_same({"l2": l2_dtype}, [mstype.float16, mstype.float32], self.name)
        valid_dtypes = [mstype.int32, mstype.int64]
        validator.check_tensor_dtype_valid('indices', indices_dtype, valid_dtypes, self.name)


class ApplyAddSign(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the AddSign algorithm.

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta * m_{t} + (1 - \beta) * g \\
            \text{update} = (\alpha + \text{sign_decay} * sign(g) * sign(m)) * g \\
            var = var - lr_{t+1} * \text{update}
        \end{array}

    :math:`t` represents updating step while :math:`m` represents the 1st moment vector, :math:`m_{t}`
    is the last momentent of :math:`m_{t+1}`, :math:`lr` represents scaling factor `lr`, :math:`g` represents `grad`,
    :math:`\alpha` represents `alpha`, :math:`\beta` represents `beta`.

    Inputs of `var`, `accum` and `grad`  comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **m** (Parameter) - Variable tensor to be updated, has the same shape and data type as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be a scalar.
          With float32 or float16 data type.
        - **alpha** (Union[Number, Tensor]) - Must be a scalar. With float32 or float16 data type.
        - **sign_decay** (Union[Number, Tensor]) - Must be a scalar. With float32 or float16 data type.
        - **beta** (Union[Number, Tensor]) - The exponential decay rate, must be a scalar.
          With float32 or float16 data type.
        - **grad** (Tensor) - A tensor of the same shape and data type as `var`, for the gradient.

    Outputs:
        Tuple of 2 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.

    Raises:
        TypeError: If dtype of `var`, `lr`, `alpha`, `sign_decay` or `beta` is neither float16 nor float32.
        TypeError: If `lr`, `alpha` or `sign_decay` is neither a Number nor a Tensor.
        TypeError: If `grad` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_add_sign = ops.ApplyAddSign()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                             [0.2, 0.6]]).astype(np.float32)), name="m")
        ...         self.lr = 0.001
        ...         self.alpha = 1.0
        ...         self.sign_decay = 0.99
        ...         self.beta = 0.9
        ...     def construct(self, grad):
        ...         out = self.apply_add_sign(self.var, self.m, self.lr, self.alpha, self.sign_decay, self.beta, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.99403024e-01,  3.98607016e-01],
         [ 9.98010039e-02,  4.98407990e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.70000052e-01,  5.19999981e-01],
         [ 1.89999998e-01,  6.20000064e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('m', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('alpha', dtype=sig.sig_dtype.T2),
        sig.make_sig('sign_decay', dtype=sig.sig_dtype.T3),
        sig.make_sig('beta', dtype=sig.sig_dtype.T3),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyAddSign."""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, lr_shape, alpha_shape, sign_decay_shape,
                    beta_shape, grad_shape):
        validator.check('m_shape', m_shape, 'var_shape', var_shape, Rel.EQ, self.name)
        validator.check('grad_shape', grad_shape, 'var_shape', var_shape, Rel.EQ, self.name)
        lr_shape_len = len(lr_shape)
        validator.check_int(lr_shape_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shape_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        alpha_shape_len = len(alpha_shape)
        validator.check_int(alpha_shape_len, 1, Rel.LE, "alpha's rank", self.name)
        if alpha_shape_len == 1:
            validator.check_int(alpha_shape[0], 1, Rel.EQ, "alpha_shape[0]", self.name)
        sign_decay_shape_len = len(sign_decay_shape)
        validator.check_int(sign_decay_shape_len, 1, Rel.LE, "sign_decay's rank", self.name)
        if sign_decay_shape_len == 1:
            validator.check_int(sign_decay_shape[0], 1, Rel.EQ, "sign_decay_shape[0]", self.name)
        beta_shape_len = len(beta_shape)
        validator.check_int(beta_shape_len, 1, Rel.LE, "beta's rank", self.name)
        if beta_shape_len == 1:
            validator.check_int(beta_shape[0], 1, Rel.EQ, "beta_shape[0]", self.name)
        return var_shape, m_shape

    def infer_dtype(self, var_dtype, m_dtype, lr_dtype, alpha_dtype, sign_decay_dtype,
                    beta_dtype, grad_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_dtype, 'm': m_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"alpha": alpha_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"sign_decay": sign_decay_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"beta": beta_dtype}, valid_dtypes, self.name)
        return var_dtype, m_dtype


class ApplyPowerSign(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the AddSign algorithm.

    .. math::
        \begin{array}{ll} \\
            m_{t+1} = \beta * m_{t} + (1 - \beta) * g \\
            \text{update} = \exp(\text{logbase} * \text{sign_decay} * sign(g) * sign(m)) * g \\
            var = var - lr_{t+1} * \text{update}
        \end{array}

    :math:`t` represents updating step while :math:`m` represents the 1st moment vector, :math:`m_{t}`
    is the last momentent of :math:`m_{t+1}`, :math:`lr` represents scaling factor `lr`, :math:`g` represents `grad`,
    :math:`\beta` represents `beta`.

    All of inputs comply with the implicit type conversion rules to make the data types consistent.
    If `lr`, `logbase`, `sign_decay` or `beta` is a number, the number is automatically converted to Tensor,
    and the data type is consistent with the Tensor data type involved in the operation.
    If inputs are tensors and have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. With float32 or float16 data type.
          If data type of `var` is float16, all inputs must have the same data type as `var`.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **m** (Parameter) - Variable tensor to be updated, has the same shape and data type as `var`.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be a scalar.
          With float32 or float16 data type.
        - **logbase** (Union[Number, Tensor]) - Must be a scalar. With float32 or float16 data type.
        - **sign_decay** (Union[Number, Tensor]) - Must be a scalar. With float32 or float16 data type.
        - **beta** (Union[Number, Tensor]) - The exponential decay rate, must be a scalar.
          With float32 or float16 data type.
        - **grad** (Tensor) - A tensor of the same shape and data type as `var`, for the gradient.

    Outputs:
        Tuple of 2 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **m** (Tensor) - The same shape and data type as `m`.

    Raises:
        TypeError: If dtype of `var`, `lr`, `logbase`, `sign_decay`, `beta` or `grad` is neither float16 nor float32.
        TypeError: If `lr`, `logbase`, `sign_decay` or `beta` is neither a Number nor a Tensor.
        TypeError: If `grad` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_power_sign = ops.ApplyPowerSign()
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.m = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                             [0.2, 0.6]]).astype(np.float32)), name="m")
        ...         self.lr = 0.001
        ...         self.logbase = np.e
        ...         self.sign_decay = 0.99
        ...         self.beta = 0.9
        ...     def construct(self, grad):
        ...         out = self.apply_power_sign(self.var, self.m, self.lr, self.logbase,
        ...                                        self.sign_decay, self.beta, grad)
        ...         return out
        ...
        >>> net = Net()
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(grad)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.95575690e-01,  3.89676481e-01],
         [ 9.85252112e-02,  4.88201708e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.70000052e-01,  5.19999981e-01],
         [ 1.89999998e-01,  6.20000064e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('m', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T),
        sig.make_sig('logbase', dtype=sig.sig_dtype.T),
        sig.make_sig('sign_decay', dtype=sig.sig_dtype.T),
        sig.make_sig('beta', dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyPowerSign."""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, m_shape, lr_shape, logbase_shape, sign_decay_shape,
                    beta_shape, grad_shape):
        validator.check('m_shape', m_shape, 'var_shape', var_shape, Rel.EQ, self.name)
        validator.check('grad_shape', grad_shape, 'var_shape', var_shape, Rel.EQ, self.name)
        lr_shape_len = len(lr_shape)
        validator.check_int(lr_shape_len, 1, Rel.LE, "lr's rank", self.name)
        if lr_shape_len == 1:
            validator.check_int(lr_shape[0], 1, Rel.EQ, "lr_shape[0]", self.name)
        logbase_shape_len = len(logbase_shape)
        validator.check_int(logbase_shape_len, 1, Rel.LE, "logbase's rank", self.name)
        if logbase_shape_len == 1:
            validator.check_int(logbase_shape[0], 1, Rel.EQ, "logbase_shape[0]", self.name)
        sign_decay_shape_len = len(sign_decay_shape)
        validator.check_int(sign_decay_shape_len, 1, Rel.LE, "sign_decay's rank", self.name)
        if sign_decay_shape_len == 1:
            validator.check_int(sign_decay_shape[0], 1, Rel.EQ, "sign_decay_shape[0]", self.name)
        beta_shape_len = len(beta_shape)
        validator.check_int(beta_shape_len, 1, Rel.LE, "beta's rank", self.name)
        if beta_shape_len == 1:
            validator.check_int(beta_shape[0], 1, Rel.EQ, "beta_shape[0]", self.name)
        return var_shape, m_shape

    def infer_dtype(self, var_dtype, m_dtype, lr_dtype, logbase_dtype, sign_decay_dtype,
                    beta_dtype, grad_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_dtype, 'm': m_dtype, 'grad': grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr": lr_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"logbase": logbase_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"sign_decay": sign_decay_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"beta": beta_dtype}, valid_dtypes, self.name)
        return var_dtype, m_dtype


class ApplyGradientDescent(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the following.

    .. math::
        var = var - \alpha * \delta

    where :math:`\alpha` represents `alpha`, :math:`\delta` represents `delta`.

    Inputs of `var` and `delta` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **alpha** (Union[Number, Tensor]) - Scaling factor, must be a scalar. With float32 or float16 data type.
        - **delta** (Tensor) - A tensor for the change, has the same shape and data type as `var`.

    Outputs:
        Tensor, represents the updated `var`.

    Raises:
        TypeError: If dtype of `var` or `alpha` is neither float16 nor float32.
        TypeError: If `delta` is not a Tensor.
        TypeError: If `alpha` is neither a Number nor a Tensor.

    Supported Platforms:
        ``Ascend``  ``GPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_gradient_descent = ops.ApplyGradientDescent()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...         self.alpha = 0.001
        ...     def construct(self, delta):
        ...         out = self.apply_gradient_descent(self.var, self.alpha, delta)
        ...         return out
        ...
        >>> net = Net()
        >>> delta = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]).astype(np.float32))
        >>> output = net(delta)
        >>> print(output)
        [[0.9999 0.9999]
         [0.9999 0.9999]]
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('alpha', dtype=sig.sig_dtype.T1),
        sig.make_sig('delta', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyGradientDescent."""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, alpha_shape, delta_shape):
        validator.check('delta shape', delta_shape, 'var shape', var_shape, Rel.EQ, self.name)
        alpha_shape_len = len(alpha_shape)
        validator.check_int(alpha_shape_len, 1, Rel.LE, "alpha's rank", self.name)
        if alpha_shape_len == 1:
            validator.check_int(alpha_shape[0], 1, Rel.EQ, "alpha_shape[0]", self.name)
        return var_shape

    def infer_dtype(self, var_dtype, alpha_dtype, delta_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_dtype, 'delta': delta_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"alpha": alpha_dtype}, valid_dtypes, self.name)
        return var_dtype


class ApplyProximalGradientDescent(PrimitiveWithInfer):
    r"""
    Updates relevant entries according to the FOBOS(Forward Backward Splitting) algorithm.

    .. math::
        \begin{array}{ll} \\
            \text{prox_v} = var - \alpha * \delta \\
            var = \frac{sign(\text{prox_v})}{1 + \alpha * l2} * \max(\left| \text{prox_v} \right| - \alpha * l1, 0)
        \end{array}

    where :math:`\alpha` represents `alpha`, :math:`\delta` represents `delta`.

    Inputs of `var` and `delta` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Inputs:
        - **var** (Parameter) - Variable tensor to be updated. With float32 or float16 data type.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **alpha** (Union[Number, Tensor]) - Scaling factor, must be a scalar. With float32 or float16 data type.
        - **l1** (Union[Number, Tensor]) - l1 regularization strength, must be scalar.
          With float32 or float16 data type.
        - **l2** (Union[Number, Tensor]) - l2 regularization strength, must be scalar.
          With float32 or float16 data type.
        - **delta** (Tensor) - A tensor for the change, has the same shape and data type as `var`.

    Outputs:
        Tensor, represents the updated `var`.

    Raises:
        TypeError: If dtype of `var`, `alpha`, `l1` or `l2` is neither float16 nor float32.
        TypeError: If `alpha`, `l1` or `l2` is neither a Number nor a Tensor.
        TypeError: If `delta` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.apply_proximal_gradient_descent = ops.ApplyProximalGradientDescent()
        ...         self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
        ...         self.alpha = 0.001
        ...         self.l1 = 0.1
        ...         self.l2 = 0.1
        ...     def construct(self, delta):
        ...         out = self.apply_proximal_gradient_descent(self.var, self.alpha, self.l1, self.l2, delta)
        ...         return out
        ...
        >>> net = Net()
        >>> delta = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]).astype(np.float32))
        >>> output = net(delta)
        >>> print(output)
        [[0.99969995 0.99969995]
         [0.99969995 0.99969995]]
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('alpha', dtype=sig.sig_dtype.T1),
        sig.make_sig('l1', dtype=sig.sig_dtype.T2),
        sig.make_sig('l2', dtype=sig.sig_dtype.T3),
        sig.make_sig('delta', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize ApplyGradientDescent."""
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, alpha_shape, l1_shape, l2_shape, delta_shape):
        validator.check('delta shape', delta_shape, 'var shape', var_shape, Rel.EQ, self.name)
        alpha_shape_len = len(alpha_shape)
        validator.check_int(alpha_shape_len, 1, Rel.LE, "alpha's rank", self.name)
        if alpha_shape_len == 1:
            validator.check_int(alpha_shape[0], 1, Rel.EQ, "alpha_shape[0]", self.name)
        l1_shape_len = len(l1_shape)
        validator.check_int(l1_shape_len, 1, Rel.LE, "l1's rank", self.name)
        if l1_shape_len == 1:
            validator.check_int(l1_shape[0], 1, Rel.EQ, "l1_shape[0]", self.name)
        l2_shape_len = len(l2_shape)
        validator.check_int(l2_shape_len, 1, Rel.LE, "l2's rank", self.name)
        if l2_shape_len == 1:
            validator.check_int(l2_shape[0], 1, Rel.EQ, "l2_shape[0]", self.name)
        return var_shape

    def infer_dtype(self, var_dtype, alpha_dtype, l1_dtype, l2_dtype, delta_dtype):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_dtype, 'delta': delta_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"alpha": alpha_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l1": l1_dtype}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l2": l2_dtype}, valid_dtypes, self.name)
        return var_dtype


class LARSUpdate(PrimitiveWithInfer):
    """
    Conducts LARS (layer-wise adaptive rate scaling) update on the sum of squares of gradient.

    For more details, please refer to :class:`nn.LARS`.

    Args:
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        hyperpara (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.

    Inputs:
        - **weight** (Tensor) - A tensor, representing the weight.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **gradient** (Tensor) - The gradient of weight, which has the same shape and dtype with weight.
        - **norm_weight** (Tensor) - A scalar tensor, representing the sum of squares of weight.
        - **norm_gradient** (Tensor) - A scalar tensor, representing the sum of squares of gradient.
        - **weight_decay** (Union[Number, Tensor]) - Weight decay. It must be a scalar tensor or number.
        - **learning_rate** (Union[Number, Tensor]) - Learning rate. It must be a scalar tensor or number.

    Outputs:
        Tensor, represents the new gradient.

    Raises:
        TypeError: If neither `epsilon` nor `hyperpara` is a float.
        TypeError: If `use_clip` is a bool.
        TypeError: If `weight`, `gradient`, `norm_weight` or `norm_gradient` is not a Tensor.
        TypeError: If `weight_decay` or `learning_rate` is neither a Number nor a Tensor.
        TypeError: If shape of `gradient` is not same as `weight`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.lars = ops.LARSUpdate()
        ...         self.reduce = ops.ReduceSum()
        ...         self.square = ops.Square()
        ...     def construct(self, weight, gradient):
        ...         w_square_sum = self.reduce(self.square(weight))
        ...         grad_square_sum = self.reduce(self.square(gradient))
        ...         grad_t = self.lars(weight, gradient, w_square_sum, grad_square_sum, 0.0, 1.0)
        ...         return grad_t
        ...
        >>> weight = Tensor(np.array([[0.5, 0.8, 0.2], [0.6, 0.4, 0.2]]).astype(np.float32))
        >>> gradient = Tensor(np.array([[0.4, 0.4, 0.5], [0.2, 0.4, 0.3]]).astype(np.float32))
        >>> net = Net()
        >>> output = net(Tensor(weight), Tensor(gradient))
        >>> print(output)
        [[0.0005265  0.0005265 0.00065813]
         [0.00026325 0.0005265 0.00039488]]
    """

    @prim_attr_register
    def __init__(self, epsilon=1e-05, hyperpara=0.001, use_clip=False):
        """Initialize LARSUpdate."""
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_value_type("hyperpara", hyperpara, [float], self.name)
        validator.check_value_type("use_clip", use_clip, [bool], self.name)

    def infer_shape(self, weight_shape, gradient_shape, norm_weight_shape, norm_gradient_shape, weight_decay_shape,
                    learning_rate_shape):
        validator.check("weight shape", weight_shape, "gradient shape", gradient_shape, Rel.EQ, self.name)
        validator.check("norm weight shape", norm_weight_shape, "norm gradient shape", norm_gradient_shape, Rel.EQ,
                        self.name)
        shp_len = len(weight_decay_shape)
        validator.check_int(shp_len, 1, Rel.LE, "weight decay's rank", self.name)
        if shp_len == 1:
            validator.check_int(weight_decay_shape[0], 1, Rel.EQ, "weight_decay_shape[0]", self.name)
        shp_len = len(learning_rate_shape)
        validator.check_int(shp_len, 1, Rel.LE, "learning rate's rank", self.name)
        if shp_len == 1:
            validator.check_int(learning_rate_shape[0], 1, Rel.EQ, "learning_rate_shape[0]", self.name)
        return weight_shape

    def infer_dtype(self, weight_dtype, gradient_dtype, norm_weight_dtype, norm_gradient_dtype,
                    weight_decay_dtype, learning_rate_dtype):
        args = {"Weight dtype": weight_dtype, "gradient dtype": gradient_dtype, "norm weight dtype": norm_weight_dtype,
                "norm gradient dtype": norm_gradient_dtype}
        validator.check_tensors_dtypes_same_and_valid(args,
                                                      [mstype.float16, mstype.float32, mstype.int16, mstype.int32],
                                                      self.name)
        validator.check_scalar_or_tensor_types_same({"weight_decay": weight_decay_dtype},
                                                    [mstype.float16, mstype.float32, mstype.float64], self.name)
        validator.check_scalar_or_tensor_types_same({"learning_rate": learning_rate_dtype},
                                                    [mstype.float16, mstype.float32, mstype.float64], self.name)
        return weight_dtype


class ApplyFtrl(PrimitiveWithInfer):
    """
    Updates relevant entries according to the FTRL scheme.

    For more details, please refer to :class:`nn.FTRL`.

    Args:
        use_locking (bool): Use locks for updating operation if true . Default: False.

    Inputs:
        - **var** (Parameter) - The variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - The accumulation to be updated, must be same shape and data type as `var`.
        - **linear** (Parameter) - The linear coefficient to be updated, must be same shape and data type as `var`.
        - **grad** (Tensor) - Gradient. The data type must be float16 or float32.
        - **lr** (Union[Number, Tensor]) - The learning rate value, must be positive. Default: 0.001.
          It must be a float number or a scalar tensor with float16 or float32 data type.
        - **l1** (Union[Number, Tensor]) - l1 regularization strength, must be greater than or equal to zero.
          Default: 0.0. It must be a float number or a scalar tensor with float16 or float32 data type.
        - **l2** (Union[Number, Tensor]) - l2 regularization strength, must be greater than or equal to zero.
          Default: 0.0. It must be a float number or a scalar tensor with float16 or float32 data type.
        - **lr_power** (Union[Number, Tensor]) - Learning rate power controls how the learning rate decreases
          during training, must be less than or equal to zero. Use fixed learning rate if lr_power is zero.
          Default: -0.5. It must be a float number or a scalar tensor with float16 or float32 data type.

    Outputs:
        - **var** (Tensor) - Represents the updated `var`. As the input parameters has been updated in-place, this
          value is always zero when the platforms is GPU.

    Raises:
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `var`, `grad`, `lr`, `l1`, `l2` or `lr_power` is neither float16 nor float32.
        TypeError: If `lr`, `l1`, `l2` or `lr_power` is neither a Number nor a Tensor.
        TypeError: If `grad` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class ApplyFtrlNet(nn.Cell):
        ...     def __init__(self):
        ...         super(ApplyFtrlNet, self).__init__()
        ...         self.apply_ftrl = ops.ApplyFtrl()
        ...         self.lr = 0.001
        ...         self.l1 = 0.0
        ...         self.l2 = 0.0
        ...         self.lr_power = -0.5
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4],
        ...                                               [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.6, 0.5],
        ...                                                 [0.2, 0.6]]).astype(np.float32)), name="accum")
        ...         self.linear = Parameter(Tensor(np.array([[0.9, 0.1],
        ...                                                  [0.7, 0.8]]).astype(np.float32)), name="linear")
        ...
        ...     def construct(self, grad):
        ...         out = self.apply_ftrl(self.var, self.accum, self.linear, grad, self.lr, self.l1, self.l2,
        ...                               self.lr_power)
        ...         return out
        ...
        >>> net = ApplyFtrlNet()
        >>> input_x = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> output = net(input_x)
        >>> print(net.var.asnumpy())
        [[ 0.0390525  0.11492836]
         [ 0.00066425 0.15075898]]
    """

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ApplyFtrl."""
        self.init_prim_io_names(inputs=['var', 'accum', 'linear', 'grad', 'lr', 'l1', 'l2', 'lr_power'],
                                outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)

    def infer_shape(self, var_shape, accum_shape, linear_shape, grad_shape, lr_shape, l1_shape, l2_shape,
                    lr_power_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('var shape', var_shape, 'linear shape', linear_shape, Rel.EQ, self.name)
        return var_shape

    def infer_dtype(self, var_type, accum_type, linear_type, grad_type, lr_type, l1_type, l2_type,
                    lr_power_type):
        valid_dtypes = [mstype.float16, mstype.float32]
        args = {'var': var_type, 'accum': accum_type, 'linear': linear_type, 'grad': grad_type}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)

        validator.check_scalar_or_tensor_types_same({"lr": lr_type}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l1": l1_type}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"l2": l2_type}, valid_dtypes, self.name)
        validator.check_scalar_or_tensor_types_same({"lr_power": lr_power_type}, valid_dtypes, self.name)
        return var_type


class SparseApplyFtrl(PrimitiveWithCheck):
    """
    Updates relevant entries according to the FTRL-proximal scheme.

    For more details, please refer to :class:`nn.FTRL`.

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        lr (float): The learning rate value, must be positive.
        l1 (float): l1 regularization strength, must be greater than or equal to zero.
        l2 (float): l2 regularization strength, must be greater than or equal to zero.
        lr_power (float): Learning rate power controls how the learning rate decreases during training,
            must be less than or equal to zero. Use fixed learning rate if `lr_power` is zero.
        use_locking (bool): Use locks for updating operation if true . Default: False.

    Inputs:
        - **var** (Parameter) - The variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - The accumulation to be updated, must be same data type and shape as `var`.
        - **linear** (Parameter) - The linear coefficient to be updated, must be the same data type and shape as `var`.
        - **grad** (Tensor) - A tensor of the same type as `var` and grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A tensor of indices in the first dimension of `var` and `accum`.
          If there are duplicates in `indices`, the behavior is undefined.
          The type must be int32 or int64 and indices.shape[0] = grad.shape[0].

    Outputs:
        - **var** (Tensor) - Tensor, has the same shape and data type as `var`.
        - **accum** (Tensor) - Tensor, has the same shape and data type as `accum`.
        - **linear** (Tensor) - Tensor, has the same shape and data type as `linear`.

    Raises:
        TypeError: If `lr`, `l1`, `l2` or `lr_power` is not a float.
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `var`, `accum`, `linear` or `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is neither int32 nor int64.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> class SparseApplyFtrlNet(nn.Cell):
        ...     def __init__(self):
        ...         super(SparseApplyFtrlNet, self).__init__()
        ...         self.sparse_apply_ftrl = ops.SparseApplyFtrl(lr=0.01, l1=0.0, l2=0.0, lr_power=-0.5)
        ...         self.var = Parameter(Tensor(np.array([[0.2]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.1]]).astype(np.float32)), name="accum")
        ...         self.linear = Parameter(Tensor(np.array([[0.6]]).astype(np.float32)), name="linear")
        ...
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        ...         return out
        ...
        >>> net = SparseApplyFtrlNet()
        >>> grad = Tensor(np.array([[0.7]]).astype(np.float32))
        >>> indices = Tensor(np.ones([1]), mindspore.int32)
        >>> output = net(grad, indices)
        >>> print(output)
        (Tensor(shape=[1, 1], dtype=Float32, value=
        [[2.00000003e-01]]), Tensor(shape=[1, 1], dtype=Float32, value=
        [[1.00000001e-01]]), Tensor(shape=[1, 1], dtype=Float32, value=
        [[6.00000024e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('linear', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, lr, l1, l2, lr_power, use_locking=False):
        """Initialize SparseApplyFtrl."""
        validator.check_value_type("lr", lr, [float], self.name)
        validator.check_value_type("l1", l1, [float], self.name)
        validator.check_value_type("l2", l2, [float], self.name)
        validator.check_value_type("lr_power", lr_power, [float], self.name)
        self.lr = validator.check_positive_float(lr, "lr", self.name)
        self.l1 = validator.check_non_negative_float(l1, "l1", self.name)
        self.l2 = validator.check_non_negative_float(l2, "l2", self.name)
        self.lr_power = validator.check_number("lr_power", lr_power, 0, Rel.LE, self.name)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.init_prim_io_names(inputs=['var', 'accum', 'linear', 'grad', 'indices'],
                                outputs=['var', 'accum', 'linear'])
        self.add_prim_attr('side_effect_mem', True)

    def check_shape(self, var_shape, accum_shape, linear_shape, grad_shape, indices_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('var shape', var_shape, 'linear shape', linear_shape, Rel.EQ, self.name)
        if len(var_shape) > 1:
            validator.check('var_shape[1:]', var_shape[1:], 'grad_shape[1:]', grad_shape[1:], Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)

    def check_dtype(self, var_dtype, accum_dtype, linear_dtype, grad_dtype, indices_dtype):
        args = {"var_dtype": var_dtype, "accum_dtype": accum_dtype,
                "linear_dtype": linear_dtype, "grad_dtype": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid("indices_dtype", indices_dtype, [mstype.int32, mstype.int64], self.name)


class SparseApplyFtrlV2(PrimitiveWithInfer):
    """
    Updates relevant entries according to the FTRL-proximal scheme. This class has one more attribute, named
    l2_shrinkage, than class SparseApplyFtrl.

    All of inputs except `indices` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        lr (float): The learning rate value, must be positive.
        l1 (float): l1 regularization strength, must be greater than or equal to zero.
        l2 (float): l2 regularization strength, must be greater than or equal to zero.
        l2_shrinkage (float): L2 shrinkage regularization.
        lr_power (float): Learning rate power controls how the learning rate decreases during training,
            must be less than or equal to zero. Use fixed learning rate if `lr_power` is zero.
        use_locking (bool): If `True`, the var and accumulation tensors will be protected from being updated.
            Default: False.

    Inputs:
        - **var** (Parameter) - The variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **accum** (Parameter) - The accumulation to be updated, must be same data type and shape as `var`.
        - **linear** (Parameter) - the linear coefficient to be updated, must be same data type and shape as `var`.
        - **grad** (Tensor) - A tensor of the same type as `var` and grad.shape[1:] = var.shape[1:] if var.shape > 1.
        - **indices** (Tensor) - A vector of indices in the first dimension of `var` and `accum`.
          The type must be int32 and indices.shape[0] = grad.shape[0].

    Outputs:
        Tuple of 3 Tensor, the updated parameters.

        - **var** (Tensor) - Tensor, has the same shape and data type as `var`.
        - **accum** (Tensor) - Tensor, has the same shape and data type as `accum`.
        - **linear** (Tensor) - Tensor, has the same shape and data type as `linear`.

    Raises:
        TypeError: If `lr`, `l1`, `l2`, `lr_power` or `use_locking` is not a float.
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `var`, `accum`, `linear` or `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is not int32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class SparseApplyFtrlV2Net(nn.Cell):
        ...     def __init__(self):
        ...         super(SparseApplyFtrlV2Net, self).__init__()
        ...         self.sparse_apply_ftrl_v2 = ops.SparseApplyFtrlV2(lr=0.01, l1=0.0, l2=0.0,
        ...                                                         l2_shrinkage=0.0, lr_power=-0.5)
        ...         self.var = Parameter(Tensor(np.array([[0.2, 0.3]]).astype(np.float32)), name="var")
        ...         self.accum = Parameter(Tensor(np.array([[0.5, 0.9]]).astype(np.float32)), name="accum")
        ...         self.linear = Parameter(Tensor(np.array([[0.7, 0.5]]).astype(np.float32)), name="linear")
        ...
        ...     def construct(self, grad, indices):
        ...         out = self.sparse_apply_ftrl_v2(self.var, self.accum, self.linear, grad, indices)
        ...         return out
        ...
        >>> net = SparseApplyFtrlV2Net()
        >>> grad = Tensor(np.array([[0.8, 0.5]]).astype(np.float32))
        >>> indices = Tensor(np.ones([1]), mindspore.int32)
        >>> output = net(grad, indices)
        >>> print(output)
        (Tensor(shape=[1, 2], dtype=Float32, value=
        [[ 2.00000003e-01,  3.00000012e-01]]), Tensor(shape=[1, 2], dtype=Float32, value=
        [[ 5.00000000e-01,  8.99999976e-01]]), Tensor(shape=[1, 2], dtype=Float32, value=
        [[ 6.99999988e-01,  5.00000000e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('accum', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('linear', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self, lr, l1, l2, l2_shrinkage, lr_power, use_locking=False):
        """Initialize SparseApplyFtrlV2."""
        validator.check_value_type("lr", lr, [float], self.name)
        validator.check_value_type("l1", l1, [float], self.name)
        validator.check_value_type("l2", l2, [float], self.name)
        validator.check_value_type("lr_power", lr_power, [float], self.name)
        self.lr = validator.check_positive_float(lr, "lr", self.name)
        self.l1 = validator.check_non_negative_float(l1, "l1", self.name)
        self.l2 = validator.check_non_negative_float(l2, "l2", self.name)
        self.lr_power = validator.check_number("lr_power", lr_power, 0, Rel.LE, self.name)
        self.l2_shrinkage = validator.check_value_type("l2_shrinkage", l2_shrinkage, [float], self.name)
        self.use_locking = validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.add_prim_attr('side_effect_mem', True)

    def infer_shape(self, var_shape, accum_shape, linear_shape, grad_shape, indices_shape):
        validator.check('var shape', var_shape, 'accum shape', accum_shape, Rel.EQ, self.name)
        validator.check('var shape', var_shape, 'linear shape', linear_shape, Rel.EQ, self.name)
        if len(var_shape) > 1:
            validator.check('var_shape[1:]', var_shape[1:], 'grad_shape[1:]', grad_shape[1:], Rel.EQ, self.name)
        validator.check_int(len(indices_shape), 1, Rel.EQ, "indices rank", self.name)
        validator.check('grad_shape[0]', grad_shape[0], 'indices_shape[0]', indices_shape[0], Rel.EQ, self.name)
        return var_shape, accum_shape, linear_shape

    def infer_dtype(self, var_dtype, accum_dtype, linear_dtype, grad_dtype, indices_dtype):
        args = {"var_dtype": var_dtype, "accum_dtype": accum_dtype,
                "linear_dtype": linear_dtype, "grad_dtype": grad_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        validator.check_tensor_dtype_valid("indicese", indices_dtype, [mstype.int32], self.name)
        return var_dtype, accum_dtype, linear_dtype


class Dropout(PrimitiveWithCheck):
    """
    During training, randomly zeroes some of the elements of the input tensor
    with probability 1-`keep_prob` from a Bernoulli distribution.

    Args:
        keep_prob (float): The keep rate, between 0 and 1, e.g. keep_prob = 0.9,
            means dropping out 10% of input units. Default: 0.5.
        Seed0 (int): Seed0 value for random generating. Default: 0.
        Seed1 (int): Seed1 value for random generating. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.

    Outputs:
        - **output** (Tensor) - With the same shape and data type as `x`.
        - **mask** (Tensor) - With the same shape as `x`.

    Raises:
        TypeError: If `keep_prob` is not a float.
        TypeError: If `Seed0` or `Seed1` is not an int.
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> dropout = ops.Dropout(keep_prob=0.5)
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> output, mask = dropout(x)
        >>> print(output.shape)
        (2, 2)
    """

    @prim_attr_register
    def __init__(self, keep_prob=0.5, Seed0=0, Seed1=0):
        """Initialize Dropout."""
        self.seed0 = validator.check_value_type("Seed0", Seed0, [int], self.name)
        self.seed1 = validator.check_value_type("Seed1", Seed1, [int], self.name)
        self.keep_prob = validator.check_float_range(keep_prob, 0, 1, Rel.INC_RIGHT, "keep_prob", self.name)

    def check_shape(self, x_shape):
        validator.check_int(len(x_shape), 1, Rel.GE, "x_shape", self.name)

    def check_dtype(self, x_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)


class Dropout2D(PrimitiveWithInfer):
    """
    During training, randomly zeroes some of the channels of the input tensor
    with probability 1-`keep_prob` from a Bernoulli distribution.

    Args:
        keep_prob (float): The keep probability of a channel, between 0 and 1, e.g. `keep_prob` = 0.8,
            means dropping out 20% of channels. Default: 0.5.

    Inputs:
        - **x** (Tensor) - A 4-D tensor with shape :math:`(N, C, H, W)`. The data type should be int8, int16,
          int32, int64, float16 or float32.

    Outputs:
        - **output** (Tensor) - With the same shape and data type as `x`.
        - **mask** (Tensor) - With the same shape as `x` and the data type is bool.

    Raises:
        TypeError: If the data type of `keep_prob` is not float.
        ValueError: If `keep_prob` is out of the range [0.0, 1.0];
                    or if the dim of input is not 4-D.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> dropout = ops.Dropout2D(keep_prob=0.5)
        >>> x = Tensor(np.ones([2, 1, 2, 3]), mindspore.float32)
        >>> output, mask = dropout(x)
        >>> print(output.shape)
        (2, 1, 2, 3)
    """

    @prim_attr_register
    def __init__(self, keep_prob=0.5):
        """Initialize Dropout2D."""
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.keep_prob = validator.check_float_range(keep_prob, 0.0, 1.0, Rel.INC_BOTH, "keep_prob", self.name)

    def infer_shape(self, x_shape):
        validator.check_int(len(x_shape), 4, Rel.EQ, "dim of input", self.name)
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        valid_dtypes = mstype.int_type + (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        mask_dtype = mstype.tensor_type(mstype.bool_)
        return x_dtype, mask_dtype


class Dropout3D(PrimitiveWithInfer):
    """
    During training, randomly zeroes some of the channels of the input tensor
    with probability 1-`keep_prob` from a Bernoulli distribution.

    Args:
        keep_prob (float): The keep probability of a channel, between 0 and 1, e.g. `keep_prob` = 0.8,
            means dropping out 20% of channels. Default: 0.5.

    Inputs:
        - **x** (Tensor) - A 5-D tensor with shape :math:`(N, C, D, H, W)`. The data type should be int8, int16,
          int32, int64, float16 or float32.

    Outputs:
        - **output** (Tensor) - With the same shape and data type as `x`.
        - **mask** (Tensor) - With the same shape as `x` and the data type is bool.

    Raises:
        TypeError: If the data type of `keep_prob` is not float.
        ValueError: If `keep_prob` is out of the range [0.0, 1.0];
                    or if the dim of input is not 5-D.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> dropout = ops.Dropout3D(keep_prob=0.5)
        >>> x = Tensor(np.ones([2, 1, 2, 1, 2]), mindspore.float32)
        >>> output, mask = dropout(x)
        >>> print(output.shape)
        (2, 1, 2, 1, 2)
    """

    @prim_attr_register
    def __init__(self, keep_prob=0.5):
        """Initialize Dropout3D."""
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.keep_prob = validator.check_float_range(keep_prob, 0.0, 1.0, Rel.INC_BOTH, "keep_prob", self.name)

    def infer_shape(self, x_shape):
        validator.check_int(len(x_shape), 5, Rel.EQ, "dim of input", self.name)
        return x_shape, x_shape

    def infer_dtype(self, x_dtype):
        valid_dtypes = mstype.int_type + (mstype.float16, mstype.float32)
        validator.check_tensor_dtype_valid("x", x_dtype, valid_dtypes, self.name)
        mask_dtype = mstype.tensor_type(mstype.bool_)
        return x_dtype, mask_dtype


class CTCLoss(Primitive):
    r"""
    Calculates the CTC (Connectionist Temporal Classification) loss and the gradient.

    The CTC algorithm is proposed in `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with
    Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_.

    Args:
        preprocess_collapse_repeated (bool): If true, repeated labels will be collapsed prior to the CTC calculation.
                                             Default: False.
        ctc_merge_repeated (bool): If false, during CTC calculation, repeated non-blank labels will not be merged
                                   and these labels will be interpreted as individual ones. This is a simplfied
                                   version of CTC. Default: True.
        ignore_longer_outputs_than_inputs (bool): If true, sequences with longer outputs than inputs will be ignored.
                                                  Default: False.

    Inputs:
        - **x** (Tensor) - The input Tensor must be a `3-D` tensor whose shape is
          :math:`(max\_time, batch\_size, num\_classes)`. `num_classes` must be `num_labels + 1` classes, `num_labels`
          indicates the number of actual labels. Blank labels are reserved. Default blank label is `num_classes - 1`.
          Data type must be float16, float32 or float64.
        - **labels_indices** (Tensor) - The indices of labels. `labels_indices[i, :] = [b, t]` means
          `labels_values[i]` stores the id for `(batch b, time t)`. The type must be int64 and rank must be 2.
        - **labels_values** (Tensor) - A `1-D` input tensor. The values are associated with the given batch and time.
          The type must be int32. `labels_values[i]` must in the range of `[0, num_classes)`.
        - **sequence_length** (Tensor) - A tensor containing sequence lengths with the shape of :math:`(batch\_size, )`.
          The type must be int32. Each value in the tensor must not be greater than `max_time`.

    Outputs:
        - **loss** (Tensor) - A tensor containing log-probabilities, the shape is :math:`(batch\_size, )`.
          The tensor has the same data type as `x`.
        - **gradient** (Tensor) - The gradient of `loss`, has the same shape and data type as `x`.

    Raises:
        TypeError: If `preprocess_collapse_repeated`, `ctc_merge_repeated` or `ignore_longer_outputs_than_inputs`
                   is not a bool.
        TypeError: If `x`, `labels_indices`, `labels_values` or `sequence_length` is not a Tensor.
        ValueError: If rank of `labels_indices` is not equal 2.
        TypeError: If dtype of `x` is not one of the following: float16, float32 or float64.
        TypeError: If dtype of `labels_indices` is not int64.
        TypeError: If dtype of `labels_values` or `sequence_length` is not int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[0.3, 0.6, 0.6],
        ...                       [0.4, 0.3, 0.9]],
        ...
        ...                      [[0.9, 0.4, 0.2],
        ...                       [0.9, 0.9, 0.1]]]).astype(np.float32))
        >>> labels_indices = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int64)
        >>> labels_values = Tensor(np.array([2, 2]), mindspore.int32)
        >>> sequence_length = Tensor(np.array([2, 2]), mindspore.int32)
        >>> ctc_loss = ops.CTCLoss()
        >>> loss, gradient = ctc_loss(x, labels_indices, labels_values, sequence_length)
        >>> print(loss)
        [ 0.79628  0.5995158 ]
        >>> print(gradient)
        [[[ 0.27029088  0.36485454  -0.6351454  ]
          [ 0.28140804  0.25462854  -0.5360366 ]]
         [[ 0.47548494  0.2883962    0.04510255 ]
          [ 0.4082751   0.4082751    0.02843709 ]]]
    """

    @prim_attr_register
    def __init__(self, preprocess_collapse_repeated=False, ctc_merge_repeated=True,
                 ignore_longer_outputs_than_inputs=False):
        """Initialize CTCLoss."""
        self.init_prim_io_names(inputs=["inputs", "labels_indices", "labels_values", "sequence_length"],
                                outputs=["loss", "gradient"])
        validator.check_value_type("preprocess_collapse_repeated", preprocess_collapse_repeated, [bool], self.name)
        self.preprocess_collapse_repeated_ = preprocess_collapse_repeated
        self.ctc_merge_repeated_ = validator.check_value_type("ctc_merge_repeated", ctc_merge_repeated,
                                                              [bool], self.name)
        validator.check_value_type("ignore_longer_outputs_than_inputs",
                                   ignore_longer_outputs_than_inputs, [bool], self.name)
        self.ignore_longer_outputs_than_inputs_ = ignore_longer_outputs_than_inputs


class CTCGreedyDecoder(PrimitiveWithCheck):
    r"""
    Performs greedy decoding on the logits given in inputs.

    Args:
        merge_repeated (bool): If true, merge repeated classes in output. Default: True.

    Inputs:
        - **inputs** (Tensor) - The input Tensor must be a 3-D tensor whose shape is
          :math:`(max\_time, batch\_size, num\_classes)`. `num_classes` must be `num_labels + 1` classes,
          `num_labels` indicates the number of actual labels. Blank labels are reserved.
          Default blank label is `num_classes - 1`. Data type must be float32 or float64.
        - **sequence_length** (Tensor) - A tensor containing sequence lengths with the shape of :math:`(batch\_size, )`.
          The type must be int32. Each value in the tensor must be equal to or less than `max_time`.

    Outputs:
        - **decoded_indices** (Tensor) - A tensor with shape of :math:`(total\_decoded\_outputs, 2)`.
          Data type is int64.
        - **decoded_values** (Tensor) - A tensor with shape of :math:`(total\_decoded\_outputs, )`,
          it stores the decoded classes. Data type is int64.
        - **decoded_shape** (Tensor) - A tensor with shape of :math:`(batch\_size, max\_decoded\_legth)`.
          Data type is int64.
        - **log_probability** (Tensor) - A tensor with shape of :math:`(batch\_size, 1)`,
          containing sequence log-probability, has the same type as `inputs`.

    Raises:
        TypeError: If `merge_repeated` is not a bool.
        ValueError: If length of shape of `inputs` is not equal to 3.
        ValueError: If length of shape of `sequence_length` is not equal to 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> inputs = Tensor(np.array([[[0.6, 0.4, 0.2], [0.8, 0.6, 0.3]],
        ...                           [[0.0, 0.6, 0.0], [0.5, 0.4, 0.5]]]), mindspore.float32)
        >>> sequence_length = Tensor(np.array([2, 2]), mindspore.int32)
        >>> ctc_greedyDecoder = ops.CTCGreedyDecoder()
        >>> decoded_indices, decoded_values, decoded_shape, log_probability = ctc_greedyDecoder(inputs, sequence_length)
        >>> print(decoded_indices)
        [[0 0]
         [0 1]
         [1 0]]
        >>> print(decoded_values)
        [0 1 0]
        >>> print(decoded_shape)
        [2 2]
        >>> print(log_probability)
        [[-1.2]
         [-1.3]]
    """

    @prim_attr_register
    def __init__(self, merge_repeated=True):
        """Initialize CTCGreedyDecoder."""
        self.merge_repeated = validator.check_value_type("merge_repeated", merge_repeated, [bool], self.name)

    def check_shape(self, inputs_shape, sequence_length_shape):
        validator.check_int(len(inputs_shape), 3, Rel.EQ, "inputs rank", self.name)
        validator.check_int(len(sequence_length_shape), 1, Rel.EQ, "sequence_length rank", self.name)
        validator.check('inputs batch_size', inputs_shape[1], 'sequence_length batch_size',
                        sequence_length_shape[0], Rel.EQ, self.name)
        total_decoded_outputs = -1
        decoded_indices_shape = [total_decoded_outputs, 2]
        decoded_values = [total_decoded_outputs]
        decoded_shape = [2]
        log_probability_shape = [inputs_shape[1], 1]
        return decoded_indices_shape, decoded_values, decoded_shape, log_probability_shape

    def check_dtype(self, inputs_dtype, sequence_length_dtype):
        validator.check_tensor_dtype_valid("inputs_dtype", inputs_dtype, [mstype.float32, mstype.double], self.name)
        validator.check_tensor_dtype_valid("sequence_length_dtype", sequence_length_dtype, [mstype.int32], self.name)
        decoded_type = mstype.tensor_type(mstype.int64)
        return decoded_type, decoded_type, decoded_type, inputs_dtype


class BasicLSTMCell(PrimitiveWithInfer):
    """
    It's similar to operator :class:`DynamicRNN`. BasicLSTMCell will be deprecated in the future.
    Please use DynamicRNN instead.

    Supported Platforms:
        Deprecated
    """

    @prim_attr_register
    def __init__(self, keep_prob=1.0, forget_bias=1.0, state_is_tuple=True, activation='tanh'):
        """Initialize BasicLSTMCell."""
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.keep_prob = validator.check_float_range(keep_prob, 0.0, 1.0, Rel.INC_BOTH, "keep_prob", self.name)
        self.forget_bias = validator.check_value_type("forget_bias", forget_bias, [float], self.name)
        self.state_is_tuple = validator.check_value_type("state_is_tuple", state_is_tuple, [bool], self.name)
        self.activation = validator.check_string(activation, ['tanh'], "activation", self.name)

    def infer_shape(self, x_shape, h_shape, c_shape, w_shape, b_shape):
        validator.check_int(len(x_shape), 2, Rel.EQ, "x rank", self.name)
        validator.check_int(len(h_shape), 2, Rel.EQ, "h rank", self.name)
        validator.check_int(len(c_shape), 2, Rel.EQ, "c rank", self.name)
        validator.check_int(len(w_shape), 2, Rel.EQ, "w rank", self.name)
        validator.check_int(len(b_shape), 1, Rel.EQ, "b rank", self.name)
        validator.check("x_shape[0]", x_shape[0], "h_shape[0]", h_shape[0], Rel.EQ, self.name)
        validator.check("c_shape[0]", c_shape[0], "h_shape[0]", h_shape[0], Rel.EQ, self.name)
        validator.check("c_shape[1]", c_shape[1], "h_shape[1]", h_shape[1], Rel.EQ, self.name)
        validator.check("w_shape[1]", w_shape[1], "4*h_shape[1]", 4 * h_shape[1], Rel.EQ, self.name)
        validator.check("w_shape[0]", w_shape[0], "x_shape[1]+h_shape[1]", x_shape[1] + h_shape[1], Rel.EQ, self.name)
        validator.check("b_shape[0]", b_shape[0], "4*h_shape[1]", 4 * h_shape[1], Rel.EQ, self.name)
        ct_shape = c_shape
        ht_shape = c_shape
        it_shape = c_shape
        jt_shape = c_shape
        ft_shape = c_shape
        ot_shape = c_shape
        tanhct_shape = c_shape

        return ct_shape, ht_shape, it_shape, jt_shape, ft_shape, ot_shape, tanhct_shape

    def infer_dtype(self, x_dtype, h_dtype, c_dtype, w_dtype, b_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid,
                          valid_dtypes=(mstype.float16, mstype.float32), prim_name=self.name),
                  ("x_dtype", "h_dtype", "w_dtype"),
                  (x_dtype, h_dtype, w_dtype)))
        args = {"c_dtype": c_dtype, "b_dtype": b_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, [mstype.float16, mstype.float32], self.name)
        return c_dtype, mstype.float16, c_dtype, c_dtype, c_dtype, c_dtype, c_dtype


class DynamicRNN(PrimitiveWithInfer):
    r"""
    Applies a recurrent neural network to the input.
    Only long short-term memory (LSTM) currently supported.

    .. math::
        \begin{array}{ll} \\
            i_{t+1} = \sigma(W_{ix} x_{t+1} + b_{ix} + W_{ih} h_{(t)} + b_{ih}) \\
            f_{t+1} = \sigma(W_{fx} x_{t+1} + b_{fx} + W_{fh} h_{(t)} + b_{fh}) \\
            \tilde{c}_{t+1} = \tanh(W_{cx} x_{t+1} + b_{cx} + W_{ch} h_{(t)} + b_{ch}) \\
            o_{t+1} = \sigma(W_{ox} x_{t+1} + b_{ox} + W_{oh} h_{(t)} + b_{oh}) \\
            c_{t+1} = f_{t+1} * c_{(t)} + i_t * \tilde{c}_{t+1} \\
            h_{t+1} = o_{t+1} * \tanh(c_{t+1}) \\
        \end{array}

    where :math:`h_{t+1}` is the hidden state at time `t+1`, :math:`x_{t+1}` is the input
    at time `t+1`, :math:`h_{t}` is the hidden state of the layer
    at time `t` or the initial hidden state at time `0`,
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.

    Args:
        cell_type (str): A string identifying the cell type in the op. Default: 'LSTM'.
            Only 'LSTM' is currently supported.
        direction (str): A string identifying the direction in the op. Default: 'UNIDIRECTIONAL'.
            Only 'UNIDIRECTIONAL' is currently supported.
        cell_depth (int): An integer identifying the cell depth in the op. Default: 1.
        use_peephole (bool): A bool identifying if use peephole in the op. Default: False.
        keep_prob (float): A float identifying the keep prob in the op. Default: 1.0.
        cell_clip (float): A float identifying the cell clip in the op. Default: -1.0.
        num_proj (int): An integer identifying the num proj in the op. Default: 0.
        time_major (bool): A bool identifying the time major in the op. Default: True.
            Only `True` is currently supported.
        activation (str): A string identifying the type of activation function in the op. Default: 'tanh'.
            Only 'tanh' is currently supported.
        forget_bias (float): A float identifying the forget bias in the op. Default: 0.0.
        is_training (bool): A bool identifying is training in the op. Default: True.

    Inputs:
        - **x** (Tensor) - Current words. Tensor of shape :math:`(num\_step, batch\_size, input\_size)`.
          The data type must be float16.
        - **w** (Tensor) - Weight. Tensor of shape :math:`(input\_size + hidden\_size, 4 x hidden\_size)`.
          The data type must be float16.
        - **b** (Tensor) - Bias. Tensor of shape :math`(4 x hidden\_size)`.
          The data type must be float16 or float32.
        - **seq_length** (Tensor) - The length of each batch. Tensor of shape :math:`(batch\_size, )`.
          Only `None` is currently supported.
        - **init_h** (Tensor) - Hidden state of initial time. Tensor of shape :math:`(1, batch\_size, hidden\_size)`.
          The data type must be float16.
        - **init_c** (Tensor) - Cell state of initial time. Tensor of shape :math:`(1, batch\_size, hidden\_size)`.
          The data type must be float16.

    Outputs:
        - **y** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **output_h** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          With data type of float16.
        - **output_c** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **i** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **j** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **f** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **o** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.
        - **tanhct** (Tensor) - A Tensor of shape :math:`(num\_step, batch\_size, hidden\_size)`.
          Has the same type with input `b`.

    Raises:
        TypeError: If `cell_type`, `direction` or `activation` is not a str.
        TypeError: If `cell_depth` or `num_proj` is not an int.
        TypeError: If `keep_prob`, `cell_clip` or `forget_bias` is not a float.
        TypeError: If `use_peehpole`, `time_major` or `is_training` is not a bool.
        TypeError: If `x`, `w`, `b`, `seq_length`, `init_h` or `init_c` is not a Tensor.
        TypeError: If dtype of `x`, `w`, `init_h` or `nit_c` is not float16.
        TypeError: If dtype of `b` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.random.rand(2, 16, 64).astype(np.float16))
        >>> w = Tensor(np.random.rand(96, 128).astype(np.float16))
        >>> b = Tensor(np.random.rand(128).astype(np.float16))
        >>> init_h = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
        >>> init_c = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
        >>> dynamic_rnn = ops.DynamicRNN()
        >>> output = dynamic_rnn(x, w, b, None, init_h, init_c)
        >>> print(output[0].shape)
        (2, 16, 32)
    """

    @prim_attr_register
    def __init__(self,
                 cell_type='LSTM',
                 direction='UNIDIRECTIONAL',
                 cell_depth=1,
                 use_peephole=False,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation='tanh',
                 forget_bias=0.0,
                 is_training=True):
        """Initialize DynamicRNN."""
        self.forget_bias = validator.check_value_type("forget_bias", forget_bias, [float], self.name)
        self.cell_depth = validator.check_value_type("cell_depth", cell_depth, [int], self.name)
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.cell_clip = validator.check_value_type("cell_clip", cell_clip, [float], self.name)
        self.num_proj = validator.check_non_negative_int(num_proj, "num_proj", self.name)
        self.forget_bias = validator.check_value_type("forget_bias", forget_bias, [float], self.name)
        self.use_peephole = validator.check_value_type("use_peephole", use_peephole, [bool], self.name)
        self.time_major = validator.check_value_type("time_major", time_major, [bool], self.name)
        self.is_training = validator.check_value_type("is_training", is_training, [bool], self.name)
        validator.check_value_type("cell_type", cell_type, [str], self.name)
        self.cell_type = validator.check_string(cell_type, ['LSTM'], "cell_type", self.name)
        validator.check_value_type("direction", direction, [str], self.name)
        self.direction = validator.check_string(direction, ['UNIDIRECTIONAL'], "direction", self.name)
        validator.check_value_type("activation", activation, [str], self.name)
        self.activation = validator.check_string(activation, ['tanh'], "activation", self.name)

    def infer_shape(self, x_shape, w_shape, b_shape, seq_shape, h_shape, c_shape):
        validator.check_int(len(x_shape), 3, Rel.EQ, "x_shape", self.name)
        validator.check_int(len(w_shape), 2, Rel.EQ, "w rank", self.name)
        validator.check_int(len(b_shape), 1, Rel.EQ, "b rank", self.name)
        validator.check_int(len(h_shape), 3, Rel.EQ, "h_shape", self.name)
        validator.check_int(len(c_shape), 3, Rel.EQ, "c_shape", self.name)
        if seq_shape is not None:
            raise ValueError(f"For '{self.name}', the dimension of 'seq_length' should be None, but got {seq_shape}.")

        num_step, batch_size, input_size = x_shape
        hidden_size = w_shape[-1] // 4

        validator.check("b_shape[-1]", b_shape[-1], "w_shape[-1]", w_shape[-1], Rel.EQ, self.name)
        if w_shape[-1] % 4 != 0:
            raise ValueError(f"For '{self.name}', the last dimension of 'w' should be a multiple of 4, "
                             f"but got {w_shape[-1]}.")
        validator.check("w_shape[0]", w_shape[0], "input_size + hidden_size",
                        input_size + hidden_size, Rel.EQ, self.name)
        validator.check("b_shape[0]", b_shape[0], "w_shape[1]", w_shape[1], Rel.EQ, self.name)
        validator.check_int(h_shape[0], 1, Rel.EQ, "h_shape[0]", self.name)
        validator.check("h_shape[1]", h_shape[1], "batch_size", batch_size, Rel.EQ, self.name)
        validator.check("h_shape[2]", h_shape[2], "hidden_size", hidden_size, Rel.EQ, self.name)
        validator.check("c_shape", c_shape, "h_shape", h_shape, Rel.EQ, self.name)
        self.placeholder_index = [3]
        self.add_prim_attr("placeholder_index", self.placeholder_index)
        self.add_prim_attr("input_size", input_size)
        self.add_prim_attr("hidden_size", hidden_size)
        y_shape = (num_step, batch_size, hidden_size)
        return y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape, y_shape

    def infer_dtype(self, x_dtype, w_dtype, b_dtype, seq_dtype, h_dtype, c_dtype):
        tuple(map(partial(validator.check_tensor_dtype_valid, valid_dtypes=[mstype.float16], prim_name=self.name),
                  ("x", "w", "h", "c"),
                  (x_dtype, w_dtype, h_dtype, c_dtype)))
        validator.check_tensor_dtype_valid("b", b_dtype, (mstype.float16, mstype.float32), self.name)
        return b_dtype, x_dtype, b_dtype, b_dtype, b_dtype, b_dtype, b_dtype, b_dtype


class DynamicGRUV2(PrimitiveWithInfer):
    r"""
    Applies a single-layer gated recurrent unit (GRU) to an input sequence.

    .. math::

        \begin{array}{ll}
            r_{t+1} = \sigma(W_{ir} x_{t+1} + b_{ir} + W_{hr} h_{(t)} + b_{hr}) \\
            z_{t+1} = \sigma(W_{iz} x_{t+1} + b_{iz} + W_{hz} h_{(t)} + b_{hz}) \\
            n_{t+1} = \tanh(W_{in} x_{t+1} + b_{in} + r_{t+1} * (W_{hn} h_{(t)}+ b_{hn})) \\
            h_{t+1} = (1 - z_{t+1}) * n_{t+1} + z_{t+1} * h_{(t)}
        \end{array}

    where :math:`h_{t+1}` is the hidden state at time `t+1`, :math:`x_{t+1}` is the input
    at time `t+1`, :math:`h_{t}` is the hidden state of the layer
    at time `t` or the initial hidden state at time `0`, and :math:`r_{t+1}`,
    :math:`z_{t+1}`, :math:`n_{t+1}` are the reset, update, and new gates, respectively.
    :math:`W`, :math:`b` are the weight parameter and the deviation parameter respectively.
    :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.

    Args:
        direction (str): A string identifying the direction in the op. Default: 'UNIDIRECTIONAL'.
            Only 'UNIDIRECTIONAL' is currently supported.
        cell_depth (int): An integer identifying the cell depth in the op. Default: 1.
        keep_prob (float): A float identifying the keep prob in the op. Default: 1.0.
        cell_clip (float): A float identifying the cell clip in the op. Default: -1.0.
        num_proj (int): An integer identifying the num proj in the op. Default: 0.
        time_major (bool): A bool identifying the time major in the op. Default: True.
        activation (str) : A string identifying the type of activation function in the op. Default: 'tanh'.
            Only 'tanh' is currently supported.
        gate_order (str): A string identifying the gate order in weight and bias. Default: 'rzh.
            'zrh' is another option.
        reset_after (bool): A bool identifying whether to apply reset gate after matrix multiplication. Default: True.
        is_training (bool): A bool identifying is training in the op. Default: True.

    Inputs:
        - **x** (Tensor) - Current words.
          Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{input_size})`.
          The data type must be float16.
        - **weight_input** (Tensor) - Input-hidden weight.
          Tensor of shape :math:`(\text{input_size}, 3 \times \text{hidden_size})`.
          The data type must be float16.
        - **weight_hidden** (Tensor) - Hidden-hidden weight.
          Tensor of shape :math:`(\text{hidden_size}, 3 \times \text{hidden_size})`.
          The data type must be float16.
        - **init_h** (Tensor) - Hidden state of initial time.
          Tensor of shape :math:`(\text{batch_size}, \text{hidden_size})`.
          The data type must be float16 or float32.
        - **bias_input** (Tensor) - Input-hidden bias. Tensor of shape :math:`(3 \times \text{hidden_size})`, or None.
          Has the same data type with input `init_h`.
        - **bias_hidden** (Tensor) - Hidden-hidden bias. Tensor of shape :math:`(3 \times \text{hidden_size})`,
          or None. Has the same data type with input `init_h`.
        - **seq_length** (Tensor) - The length of each batch. Tensor of shape :math:`(\text{batch_size})`.
          Only `None` is currently supported.

    Outputs:
        - **y** (Tensor) - A Tensor of shape:

          - y_shape = :math:`(num\_step, batch\_size, min(hidden\_size, num\_proj))`: `If num_proj > 0`,
          - y_shape = :math:`(num\_step, batch\_size, hidden\_size)`: `If num_proj = 0`.

          Has the same data type with input `bias_type`.
        - **output_h** (Tensor) - A Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})`.
          Has the same data type with input `bias_type`.
        - **update** (Tensor) - A Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})`.
          Has the same data type with input `bias_type`.
        - **reset** (Tensor) - A Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})`.
          Has the same data type with input `bias_type`.
        - **new** (Tensor) - A Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})`.
          Has the same data type with input `bias_type`.
        - **hidden_new** (Tensor) - A Tensor of shape :math:`(\text{num_step}, \text{batch_size}, \text{hidden_size})`.
          Has the same data type with input `bias_type`.

        A note about the bias_type:

        - If `bias_input` and `bias_hidden` both are `None`, `bias_type` is date type of `init_h`.
        - If `bias_input` is not `None`, `bias_type` is the date type of `bias_input`.
        - If `bias_input` is `None` and `bias_hidden` is not `None`, `bias_type` is the date type of `bias_hidden`.

    Raises:
        TypeError: If `direction`, `activation` or `gate_order` is not a str.
        TypeError: If `cell_depth` or `num_proj` is not an int.
        TypeError: If `keep_prob` or `cell_clip` is not a float.
        TypeError: If `time_major`, `reset_after` or `is_training` is not a bool.
        TypeError: If `x`, `weight_input`, `weight_hidden`, `bias_input`, `bias_hidden`, `seq_length` or `ini_h` is not
                   a Tensor.
        TypeError: If dtype of `x`, `weight_input` or `weight_hidden` is not float16.
        TypeError: If dtype of `init_h` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.random.rand(2, 8, 64).astype(np.float16))
        >>> weight_i = Tensor(np.random.rand(64, 48).astype(np.float16))
        >>> weight_h = Tensor(np.random.rand(16, 48).astype(np.float16))
        >>> bias_i = Tensor(np.random.rand(48).astype(np.float16))
        >>> bias_h = Tensor(np.random.rand(48).astype(np.float16))
        >>> init_h = Tensor(np.random.rand(8, 16).astype(np.float16))
        >>> dynamic_gru_v2 = ops.DynamicGRUV2()
        >>> output = dynamic_gru_v2(x, weight_i, weight_h, bias_i, bias_h, None, init_h)
        >>> print(output[0].shape)
        (2, 8, 16)
    """

    @prim_attr_register
    def __init__(self,
                 direction='UNIDIRECTIONAL',
                 cell_depth=1,
                 keep_prob=1.0,
                 cell_clip=-1.0,
                 num_proj=0,
                 time_major=True,
                 activation="tanh",
                 gate_order="rzh",
                 reset_after=True,
                 is_training=True):
        """Initialize DynamicGRUV2."""
        self.cell_depth = validator.check_value_type("cell_depth", cell_depth, [int], self.name)
        self.keep_prob = validator.check_value_type("keep_prob", keep_prob, [float], self.name)
        self.cell_clip = validator.check_value_type("cell_clip", cell_clip, [float], self.name)
        self.num_proj = validator.check_non_negative_int(num_proj, "num_proj", self.name)
        self.time_major = validator.check_value_type("time_major", time_major, [bool], self.name)
        self.is_training = validator.check_value_type("is_training", is_training, [bool], self.name)
        self.direction = validator.check_string(direction, ['UNIDIRECTIONAL'], "direction", self.name)
        self.activation = validator.check_string(activation, ['tanh'], "activation", self.name)
        self.gate_order = validator.check_string(gate_order, ['zrh', 'rzh'], "gate_order", self.name)
        self.reset_after = validator.check_value_type("reset_after", reset_after, [bool], self.name)

    def infer_shape(self, x_shape, winput_shape, whidden_shape, binput_shape, bhidden_shape, seq_shape, h_shape):
        validator.check_int(len(x_shape), 3, Rel.EQ, "x shape", self.name)
        validator.check_int(len(winput_shape), 2, Rel.EQ, "weight input shape rank", self.name)
        validator.check_int(len(whidden_shape), 2, Rel.EQ, "weight hidden shape rank", self.name)

        num_step, batch_size, input_size = x_shape
        hidden_size = winput_shape[-1] // 3
        if winput_shape[-1] % 3 != 0:
            raise ValueError(f"For '{self.name}', the last dimension of 'w' should be a multiple of 3, "
                             f"but got {winput_shape[-1]}.")

        self.placeholder_index = [3, 4, 5]
        if binput_shape is not None:
            validator.check_int(len(binput_shape), 1, Rel.EQ, "bias input shape rank", self.name)
            validator.check("bias_input_shape", binput_shape, "3 * hidden_shape", [3 * hidden_size], Rel.EQ, self.name)
            self.placeholder_index.remove(3)
        if bhidden_shape is not None:
            validator.check_int(len(bhidden_shape), 1, Rel.EQ, "bias hidden shape rank", self.name)
            validator.check("bias_hidden_shape", bhidden_shape,
                            "3 * hidden_shape", [3 * hidden_size], Rel.EQ, self.name)
            self.placeholder_index.remove(4)
        if seq_shape is not None:
            raise ValueError(f"For '{self.name}', the dimension of 'seq_length' should be None, "
                             f"but got {seq_shape}.")

        validator.check_int(len(h_shape), 2, Rel.EQ, "init_h shape rank", self.name)
        validator.check("init_h_shape[0]", h_shape[0], "batch_size", batch_size, Rel.EQ, self.name)
        validator.check("init_h_shape[1]", h_shape[1], "hidden_size", hidden_size, Rel.EQ, self.name)
        validator.check("weight_input_shape[-1]", winput_shape[-1], "weight_hidden_shape[-1]",
                        whidden_shape[-1], Rel.EQ, self.name)
        validator.check("weight_input_shape[0]", winput_shape[0], "input_size", input_size, Rel.EQ, self.name)
        validator.check("weight_hidden_shape[0]", whidden_shape[0], "hidden_size", hidden_size, Rel.EQ, self.name)
        if self.num_proj > 0:
            y_shape = (num_step, batch_size, min(hidden_size, self.num_proj))
        else:
            y_shape = (num_step, batch_size, hidden_size)
        out_shape = (num_step, batch_size, hidden_size)
        self.add_prim_attr("placeholder_index", self.placeholder_index)
        return y_shape, out_shape, out_shape, out_shape, out_shape, out_shape

    def infer_dtype(self, x_dtype, winput_dtype, whidden_dtype, binput_dtype, bhidden_dtype, seq_dtype, h_dtype):
        validator.check_tensor_dtype_valid("x dtype", x_dtype, [mstype.float16], self.name)
        validator.check_tensor_dtype_valid("weight input dtype", winput_dtype, [mstype.float16], self.name)
        validator.check_tensor_dtype_valid("weight hidden dtype", whidden_dtype, [mstype.float16], self.name)
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensor_dtype_valid("init_h dtype", h_dtype, valid_dtypes, self.name)
        b_dtype = h_dtype
        if binput_dtype is not None:
            args = {'init_h': h_dtype, 'bias_input': binput_dtype}
            validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
            b_dtype = binput_dtype
        if bhidden_dtype is not None:
            args = {'init_h': h_dtype, 'bias_hidden': bhidden_dtype}
            validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
            b_dtype = bhidden_dtype

        return b_dtype, b_dtype, b_dtype, b_dtype, b_dtype, b_dtype


class InTopK(PrimitiveWithInfer):
    r"""
    Determines whether the targets are in the top `k` predictions.

    Args:
        k (int): Specifies the number of top elements to be used for computing precision.

    Inputs:
        - **x1** (Tensor) - A 2D Tensor defines the predictions of a batch of samples with float16 or float32
          data type.
        - **x2** (Tensor) - A 1D Tensor defines the labels of a batch of samples with int32 data type. The size of x2
          must be equal to x1's first dimension. The values of `x2` can not be negative and
          must be equal to or less than index of x1's second dimension.

    Outputs:
        Tensor has 1 dimension of type bool and the same shape with `x2`. For labeling sample `i` in `x2`,
        if the label in the first `k` predictions for sample `i` is in `x1`, then the value is True, otherwise False.

    Raises:
        TypeError: If `k` is not an int.
        TypeError: If `x1` or `x2` is not a Tensor.
        TypeError: If dtype of `x1` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x1 = Tensor(np.array([[1, 8, 5, 2, 7], [4, 9, 1, 3, 5]]), mindspore.float32)
        >>> x2 = Tensor(np.array([1, 3]), mindspore.int32)
        >>> in_top_k = ops.InTopK(3)
        >>> output = in_top_k(x1, x2)
        >>> print(output)
        [ True  False]
    """

    @prim_attr_register
    def __init__(self, k):
        """Initialize InTopK"""
        self.init_prim_io_names(inputs=['x1', 'x2', 'k'], outputs=['y'])
        validator.check_value_type("k", k, [int], self.name)

    def infer_dtype(self, x1_dtype, x2_dtype):
        validator.check_tensor_dtype_valid("x1", x1_dtype, (mstype.float16, mstype.float32,), self.name)
        validator.check_tensor_dtype_valid("x2", x2_dtype, (mstype.int32,), self.name)

        return mstype.tensor_type(mstype.bool_)

    def infer_shape(self, x1_shape, x2_shape):
        validator.check("x1 shape", len(x1_shape), "", 2, Rel.EQ, self.name)
        validator.check("x2 shape", len(x2_shape), "", 1, Rel.EQ, self.name)
        validator.check("size of x2", x2_shape[0], "x1's first dimension", x1_shape[0], Rel.EQ, self.name)
        return x2_shape


class LRN(PrimitiveWithInfer):
    r"""
    Local Response Normalization.

    .. math::

        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    where the :math:`a_{c}` indicates the represents the specific value of the pixel corresponding to c in feature map;
    where the :math:`n/2` indicate the `depth_radius`; where the :math:`k` indicate the `bias`;
    where the :math:`\alpha` indicate the`alpha`; where the :math:`\beta` indicate the `beta`.

    Args:
        depth_radius (int): Half-width of the 1-D normalization window with the shape of 0-D. Default: 5.
        bias (float): An offset (usually positive to avoid dividing by 0). Default: 1.0.
        alpha (float): A scale factor, usually positive. Default: 1.0.
        beta (float): An exponent. Default: 0.5.
        norm_region (str): Specifies normalization region. Options: "ACROSS_CHANNELS". Default: "ACROSS_CHANNELS".

    Inputs:
        - **x** (Tensor) - A 4D Tensor with float16 or float32 data type.

    Outputs:
        Tensor, with the same shape and data type as `x`.

    Raises:
        TypeError: If `depth_radius` is not an int.
        TypeError: If `bias`, `alpha` or `beta` is not a float.
        TypeError: If `norm_region` is not a str.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([[[[0.1], [0.2]],
        ...                       [[0.3], [0.4]]]]), mindspore.float32)
        >>> lrn = ops.LRN()
        >>> output = lrn(x)
        >>> print(output)
        [[[[0.09534626]
           [0.1825742 ]]
          [[0.2860388 ]
           [0.3651484 ]]]]
    """

    @prim_attr_register
    def __init__(self, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS"):
        """Initialize LRN"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])
        validator.check_value_type("depth_radius", depth_radius, [int], self.name)
        validator.check_value_type("bias", bias, [float], self.name)
        validator.check_value_type("alpha", alpha, [float], self.name)
        validator.check_value_type("beta", beta, [float], self.name)
        validator.check_value_type("norm_region", norm_region, [str], self.name)
        validator.check_string(norm_region, ['ACROSS_CHANNELS'], 'norm_region', self.name)
        validator.check_non_negative_int(depth_radius, "depth_radius", self.name)

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, (mstype.float16, mstype.float32,), self.name)
        return x_dtype

    def infer_shape(self, x_shape):
        validator.check_int(len(x_shape), 4, Rel.EQ, "x_shape", self.name)
        return x_shape


class AvgPool3D(Primitive):
    r"""
    3D Average pooling operation.

    Applies a 3D average pooling over an input Tensor which can be regarded as a composition of 3D input planes.
    Typically the input is of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`, AvgPool3D outputs
    regional average in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. warning::
        "kernel_size" is in the range [1, 255]. "strides" is in the range [1, 63].

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \frac{1}{d_{ker} * h_{ker} * w_{ker}} \sum_{l=0}^{d_{ker}-1} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents depth, height and width are both kernel_size, or a tuple
            of three int numbers that represent depth, height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "SAME", "VALID", "PAD", not case sensitive.
            Default: "VALID".

            - same: Adopts the way of completion. The depth, height and width of the output will be the same as
              the input. The total number of padding will be calculated in depth, horizontal and vertical
              directions and evenly distributed to head and tail, top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the tail, bottom and the right side.
              If this mode is set, `pad` must be 0.

            - valid: Adopts the way of discarding. The possible largest depth, height and width of output
              will be returned without padding. Extra pixels will be discarded. If this mode is set, `pad`
              must be 0.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of `pad` will
              be padded to the input Tensor borders. `pad` must be greater than or equal to 0.
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six
                    integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
                    pad[3], pad[4] and pad[5] correspondingly.
        ceil_mode (bool): If True, ceil instead of floor to compute the output shape. Default: False.
        count_include_pad (bool): If True, averaging calculation will include the zero-padding. Default: True.
        divisor_override (int): If specified, it will be used as divisor in the averaging calculation,
            otherwise kernel_size will be used. Default: 0.
        data_format (str) : The optional value for data format. Currently only support 'NCDHW'. Default: 'NCDHW'.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C, D_{in}, H_{in}, W_{in})`.
          Currently support float16 and float32 data type.

    Outputs:
        Tensor, with shape :math:`(N, C, D_{out}, H_{out}, W_{out})`. Has the same data type with `x`.

    Raises:
        TypeError: If `kernel_size`, `strides` or `pad` is neither an int not a tuple.
        TypeError: If `ceil_mode` or `count_include_pad` is not a bool.
        TypeError: If `pad_mode` or `data_format` is not a string.
        TypeError: If `divisor_override` is not an int.
        ValueError: If numbers in `kernel_size` or `strides` are not positive.
        ValueError: If `kernel_size` or `strides` is a tuple whose length is not equal to 3.
        ValueError: If `pad_mode` is not one of 'same', 'valid' or 'pad'.
        ValueError: If `pad` is a tuple whose length is not equal to 6.
        ValueError: If element of `pad` is less than 0.
        ValueError: If `pad_mode` is not equal to 'pad' and `pad` is not equal to 0 or (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), mindspore.float16)
        >>> avg_pool3d = ops.AvgPool3D(kernel_size=2, strides=1, pad_mode="valid")
        >>> output = avg_pool3d(x)
        >>> print(output)
        [[[[[ 5.  6.]]]
          [[[17. 18.]]]]]
    """

    @prim_attr_register
    def __init__(self, kernel_size=1, strides=1, pad_mode="valid", pad=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=0, data_format="NCDHW"):
        """Initialize AvgPool3D"""
        self.init_prim_io_names(inputs=['input'], outputs=['output'])
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name)
        self.add_prim_attr('kernel_size', self.kernel_size)
        self.strides = _check_3d_int_or_tuple('strides', strides, self.name)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        self.add_prim_attr('strides', self.strides)
        if isinstance(pad, int):
            pad = (pad,) * 6
        if len(pad) != 6:
            raise ValueError(f"For '{self.name}', attr 'pad' should be an positive int number or a tuple of "
                             f"six positive int numbers, but got `{pad}`.")
        self.pad_list = pad
        self.add_prim_attr('pad_list', self.pad_list)
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME', 'PAD'], 'pad_mode', self.name)
        self.add_prim_attr('pad_mode', self.pad_mode)

        if self.pad_mode != 'PAD' and pad != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be (0, 0, 0, 0, 0, 0) when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {pad} and 'pad_mode' is {self.pad_mode}.")
        if self.pad_mode == 'PAD':
            for item in pad:
                validator.check_non_negative_int(item, 'pad or item of pad', self.name)
        self.ceil_mode = validator.check_value_type('ceil_mode', ceil_mode, bool, self.name)
        self.count_include_pad = validator.check_value_type('count_include_pad', count_include_pad, bool, self.name)
        self.divisor_override = validator.check_non_negative_int(divisor_override, 'divisor_override', self.name)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)


class Conv3D(PrimitiveWithInfer):
    r"""
    3D convolution layer.

    Applies a 3D convolution over an input tensor which is typically of shape
    :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` and output shape
    :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`. Where :math:`N` is batch size, :math:`C` is channel number,
    :math:`D` is depth, :math:`H` is height, :math:`W` is width.
    the formula is defined as:

    .. math::

        \operatorname{out}\left(N_{i}, C_{\text {out}_j}\right)=\operatorname{bias}\left(C_{\text {out}_j}\right)+
        \sum_{k=0}^{C_{in}-1} ccor(\text {weight}\left(C_{\text {out}_j}, k\right),
        \operatorname{input}\left(N_{i}, k\right))

    where :math:`k` is kernel, :math:`ccor` is the cross-correlation operator.

    If the 'pad_mode' is set to be "valid", the output depth, height and width will be
    :math:`\left \lfloor{1 + \frac{D_{in} + 2 \times \text{padding} - \text{ks_d} -
    (\text{ks_d} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{H_{in} + 2 \times \text{padding} - \text{ks_h} -
    (\text{ks_h} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` and
    :math:`\left \lfloor{1 + \frac{W_{in} + 2 \times \text{padding} - \text{ks_w} -
    (\text{ks_w} - 1) \times (\text{dilation} - 1) }{\text{stride}}} \right \rfloor` respectively. Where
    :math:`dialtion` is Spacing between kernel elements, :math:`stride` is The step length of each step,
    :math:`padding` is zero-padding added to both sides of the input.

    Args:
        out_channel (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 3 integers. Specifies the depth, height
            and width of the 3D convolution window. Single int means the value is for the depth, height and the width
            of the kernel. A tuple of 3 ints means the first value is for the depth, height and the other is for the
            width of the kernel.
        mode (int): Modes for different convolutions. It is currently not used. Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad". Default: "valid".

            - same: Adopts the way of completion. The depth, height and width of the output will be the same as
              the input. The total number of padding will be calculated in depth, horizontal and vertical
              directions and evenly distributed to head and tail, top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the tail, bottom and the right side.
              If this mode is set, `pad` must be 0.

            - valid: Adopts the way of discarding. The possible largest depth, height and width of output
              will be returned without padding. Extra pixels will be discarded. If this mode is set, `pad`
              must be 0.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of `pad` will
              be padded to the input Tensor borders. `pad` must be greater than or equal to 0.

        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six
                    integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
                    pad[3], pad[4] and pad[5] correspondingly.
        dilation (Union[int, tuple[int]]): The data type is int or a tuple of 3 integers
                                      : math:`(dilation_d, dilation_h, dilation_w)`.
                                      Currently, dilation on depth only supports the case of 1.
                                      Specifies the dilation rate to use for dilated convolution.
                                      If set :math:`k > 1`, there will be :math:`k - 1` pixels skipped
                                      for each sampling location. Its value must be greater or equal to 1 and
                                      bounded by the height and width of the input. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1. Only 1 is currently supported.
        data_format (str): The optional value for data format. Currently only support "NCDHW".

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.
          Currently input data type only support float16 and float32.
        - **weight** (Tensor) - Set size of kernel is :math:`(k_d, K_h, K_w)`, then the shape is
          :math:`(C_{out}, C_{in}//groups, k_d, K_h, K_w)`.
          Currently weight data type only support float16 and float32.
        - **bias** (Tensor) - Tensor of shape :math:`C_{in}`. Currently, only support none.

    Outputs:
        Tensor, the value that applied 3D convolution. The shape is :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `out_channel` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int nor a tuple.
        ValueError: If `out_channel`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([16, 3, 10, 32, 32]), mindspore.float16)
        >>> weight = Tensor(np.ones([32, 3, 4, 3, 3]), mindspore.float16)
        >>> conv3d = ops.Conv3D(out_channel=32, kernel_size=(4, 3, 3))
        >>> output = conv3d(x, weight)
        >>> print(output.shape)
        (16, 32, 7, 30, 30)
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCDHW"):
        """Initialize Conv3D"""
        self.init_prim_io_names(inputs=['x', 'w'], outputs=['output'])
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_3d_int_or_tuple('stride', stride, self.name, allow_five=False, ret_five=True)
        self.add_prim_attr('strides', self.stride)
        self.dilation = _check_3d_int_or_tuple('dilation', dilation, self.name, allow_five=False,
                                               ret_five=True, third_one=True)
        self.add_prim_attr('dilations', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 6
        if len(pad) != 6:
            raise ValueError(f"For '{self.name}', attr 'pad' should be an positive int number or a tuple of "
                             f"six positive int numbers, but got `{pad}`.")
        self.add_prim_attr("pad", pad)
        self.padding = pad
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.lower(), ['valid', 'same', 'pad'], 'pad_mode', self.name)
        self.add_prim_attr('pad_mode', self.pad_mode)

        if self.pad_mode != 'pad' and pad != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be (0, 0, 0, 0, 0, 0) when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {pad} and 'pad_mode' is {self.pad_mode}.")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)

        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.add_prim_attr('mode', self.mode)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.group = validator.check_equal_int(group, 1, 'group', self.name)
        self.add_prim_attr('groups', self.group)
        self.add_prim_attr('offset_x', 0)

    def infer_shape(self, x_shape, w_shape, b_shape=None):
        validator.check_equal_int(len(w_shape), 5, "weight rank", self.name)
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        if b_shape is not None:
            raise ValueError(f"For '{self.name}', the 'bias' currently only support None, but got {b_shape}.")
        validator.check(f"x_shape[1] // group", x_shape[1] // self.group, "w_shape[1]", w_shape[1], Rel.EQ, self.name)
        validator.check('out_channel', self.out_channel, 'w_shape[0]', w_shape[0], Rel.EQ, self.name)
        validator.check('kernel_size', self.kernel_size, 'w_shape[1:4]', tuple(w_shape[2:]), Rel.EQ, self.name)

        kernel_size_d = w_shape[2]
        kernel_size_h = w_shape[3]
        kernel_size_w = w_shape[4]

        stride_d = self.stride[2]
        stride_h = self.stride[3]
        stride_w = self.stride[4]

        dilation_d = self.dilation[2]
        dilation_h = self.dilation[3]
        dilation_w = self.dilation[4]

        if self.pad_mode == "valid":
            d_out = math.ceil((x_shape[2] - dilation_d * (kernel_size_d - 1)) / stride_d)
            h_out = math.ceil((x_shape[3] - dilation_h * (kernel_size_h - 1)) / stride_h)
            w_out = math.ceil((x_shape[4] - dilation_w * (kernel_size_w - 1)) / stride_w)
            pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0, 0, 0

        elif self.pad_mode == "same":
            d_out = math.ceil(x_shape[2] / stride_d)
            h_out = math.ceil(x_shape[3] / stride_h)
            w_out = math.ceil(x_shape[4] / stride_w)

            pad_needed_d = max(0, (d_out - 1) * stride_d + dilation_d * (kernel_size_d - 1) + 1 - x_shape[2])
            pad_head = math.floor(pad_needed_d / 2)
            pad_tail = pad_needed_d - pad_head

            pad_needed_h = max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[3])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[4])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left

        elif self.pad_mode == 'pad':
            pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = self.padding
            d_out = 1 + (x_shape[2] + pad_head + pad_tail - kernel_size_d - (kernel_size_d - 1)
                         * (dilation_d - 1)) / stride_d
            h_out = 1 + (x_shape[3] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1)
                         * (dilation_h - 1)) / stride_h
            w_out = 1 + (x_shape[4] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1)
                         * (dilation_w - 1)) / stride_w
            d_out = math.floor(d_out)
            h_out = math.floor(h_out)
            w_out = math.floor(w_out)

        self.pad_list = [pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]
        filter_d = (self.kernel_size[0] - 1) * dilation_d + 1
        filter_h = (self.kernel_size[1] - 1) * dilation_h + 1
        filter_w = (self.kernel_size[2] - 1) * dilation_w + 1
        validator.check_int_range(self.pad_list[0], 0, filter_d, Rel.INC_LEFT,
                                  'pad_d belonging [0, filter_d)', self.name)
        validator.check_int_range(self.pad_list[1], 0, filter_d, Rel.INC_LEFT,
                                  'pad_d belonging [0, filter_d)', self.name)
        validator.check_int_range(self.pad_list[2], 0, filter_h, Rel.INC_LEFT,
                                  'pad_h belonging [0, filter_h)', self.name)
        validator.check_int_range(self.pad_list[3], 0, filter_h, Rel.INC_LEFT,
                                  'pad_h belonging [0, filter_h)', self.name)
        validator.check_int_range(self.pad_list[4], 0, filter_w, Rel.INC_LEFT,
                                  'pad_w belonging [0, filter_w)', self.name)
        validator.check_int_range(self.pad_list[5], 0, filter_w, Rel.INC_LEFT,
                                  'pad_w belonging [0, filter_w)', self.name)
        self.add_prim_attr('pad_list', (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right))
        out_channel = self.out_channel
        out_shape = [x_shape[0], out_channel, d_out, h_out, w_out]
        _check_shape('output', out_shape, self.name)
        return out_shape

    def infer_dtype(self, x_dtype, w_dtype, b_dtype=None):
        args = {'x': x_dtype, 'w': w_dtype}
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        return x_dtype


class Conv3DBackpropInput(PrimitiveWithInfer):
    """
    Computes the gradients of convolution 3D with respect to the input.

    Args:
        out_channel (int): The dimension of the output.
        kernel_size (Union[int, tuple[int]]): The kernel size of the 3D convolution.
        mode (int): Modes for different convolutions. Not currently used.
        pad_mode (str): Modes to fill padding. It could be "valid", "same", or "pad", not case sensitive.
            Default: "valid".
        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
                    head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of four
                    integers, the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2],
                    pad[3], pad[4] and pad[5] correspondingly.
        stride (Union(int, tuple[int])): The stride to be applied to the convolution filter. Default: 1.
        dilation (Union(int, tuple[int])): Specifies the space to use between kernel elements. Default: 1.
        group (int): Splits input into groups. Default: 1.
        data_format (str): The optional value for data format. Currently only support 'NCDHW'.

    Inputs:
        - **weight** (Tensor) - Set size of kernel is :math:`(D_in, K_h, K_w)`, then the shape is
          :math:`(C_{out}, C_{in}, D_{in}, K_h, K_w)`. Currently weight data type only support float16 and float32.
        - **dout** (Tensor) - the gradients with respect to the output of the convolution.
          The shape conforms to the default.
          data_format :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`. Currently dout data type only support float16
          and float32.
        - **input_size** (tuple(int)) - A tuple describes the shape of the input which conforms to the format
          :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the gradients with respect to the input of convolution 3D. It has the same shape as the input.

    Raises:
        TypeError: If `out_channel` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `pad` or `dilation` is neither an int not a tuple.
        ValueError: If `out_channel`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> dout = Tensor(np.ones([16, 32, 10, 32, 32]), mindspore.float16)
        >>> weight = Tensor(np.ones([32, 32, 4, 6, 2]), mindspore.float16)
        >>> x = Tensor(np.ones([16, 32, 13, 37, 33]))
        >>> conv3d_backprop_input = ops.Conv3DBackpropInput(out_channel=4, kernel_size=(4, 6, 2))
        >>> output = conv3d_backprop_input(dout, weight, ops.shape(x))
        >>> print(output.shape)
        (16, 32, 13, 37, 33)
    """

    @prim_attr_register
    def __init__(self,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode="valid",
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 data_format="NCDHW"):
        """Initialize Conv3DBackpropInput"""
        self.init_prim_io_names(inputs=['filter', 'out_backprop', 'input_size'], outputs=['y'])
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_3d_int_or_tuple('stride', stride, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('strides', self.stride)
        self.dilation = _check_3d_int_or_tuple('dilation', dilation, self.name, allow_five=True, ret_five=True)
        self.add_prim_attr('dilations', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 6
        validator.check_equal_int(len(pad), 6, 'pad size', self.name)
        self.add_prim_attr("pad", pad)
        self.pad_list = pad

        self.pad_mode = validator.check_string(pad_mode.lower(), ['valid', 'same', 'pad'], 'pad_mode', self.name)
        if self.pad_mode != 'pad' and self.pad_list != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be (0, 0, 0, 0, 0, 0) "
                             f"when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {self.pad_list} and 'pad_mode' is {self.pad_mode}.")
        if self.pad_mode == 'pad':
            for item in pad:
                validator.check_non_negative_int(item, 'pad item', self.name)
        self.add_prim_attr('pad_mode', self.pad_mode)

        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.add_prim_attr('mode', self.mode)
        self.group = validator.check_positive_int(group, 'group', self.name)
        self.add_prim_attr('groups', self.group)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)

    def __infer__(self, w, doutput, x_size):
        validator.check_equal_int(len(w['shape']), 5, 'The dimension of weight ', self.name)
        validator.check_equal_int(len(doutput['shape']), 5, 'The dimension of dout', self.name)
        x_size_v = x_size['value']
        validator.check_equal_int(len(x_size_v), 5, 'The dimension of input_size', self.name)
        validator.check_value_type('x_size', x_size_v, [tuple], self.name)
        for i, dim_len in enumerate(x_size_v):
            validator.check_value_type("x_size[%d]" % i, dim_len, [int], self.name)
        args = {'doutput': doutput['dtype'], 'w': w['dtype']}
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check("filter's batch", w['shape'][0], "dout's channel", doutput['shape'][1], Rel.EQ, self.name)
        validator.check("filter's channel", w['shape'][1], "input_size's channel", x_size_v[1], Rel.EQ, self.name)
        validator.check("input_size's batch", x_size_v[0], "dout's batch", doutput['shape'][0], Rel.EQ, self.name)

        # infer shape
        dout_shape = doutput['shape']
        kernel_d = self.kernel_size[0]
        kernel_h = self.kernel_size[1]
        kernel_w = self.kernel_size[2]
        stride_d = self.stride[2]
        stride_h = self.stride[3]
        stride_w = self.stride[4]
        dilation_d = self.dilation[2]
        dilation_h = self.dilation[3]
        dilation_w = self.dilation[4]
        # The pad_mode is valid by default. If pad_mode is not valid or same, then pad.
        if self.pad_mode == "valid":
            self.pad_list = (0, 0, 0, 0, 0, 0)
        if self.pad_mode == "same":
            pad_needed_d = max(0, (dout_shape[2] - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - x_size_v[2])
            pad_head = math.floor(pad_needed_d / 2)
            pad_tail = pad_needed_d - pad_head

            pad_needed_h = max(0, (dout_shape[3] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_size_v[3])
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (dout_shape[4] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_size_v[4])
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
            self.pad_list = (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right)

        self.add_prim_attr('pad_list', self.pad_list)
        out = {
            'value': None,
            'shape': x_size_v,
            'dtype': doutput['dtype'],
        }
        return out


def _deconv_output_length(input_length, kernel_size, stride_size, dilation_size):
    filter_size = kernel_size + (kernel_size - 1) * (dilation_size - 1)
    if filter_size - stride_size > 0:
        length = input_length * stride_size + filter_size - stride_size
    else:
        length = input_length * stride_size
    return length


class CTCLossV2(Primitive):
    """
    Calculates the CTC (Connectionist Temporal Classification) loss and the gradient.

    The CTC algorithm is proposed in `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with
    Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_.

    Args:
        blank (int): The blank label. Default: 0.
        reduction (string): Apply specific reduction method to the output. Currently only support 'none',
            not case sensitive. Default: "none".
        zero_infinity (bool): Whether to set infinite loss and correlation gradient to zero. Default: False.

    Inputs:
        - **log_probs** (Tensor) - A tensor of shape (T, N, C), where T is input length, N is batch size and C is number
            of classes (including blank).
        - **targets** (Tensor) - A tensor of shape (N, S), where S is max target length, means the target sequences.
        - **input_lengths** (Union(Tuple, Tensor)) - A tuple or Tensor of shape(N). It means the lengths of the input.
        - **target_lengths** (Union(Tuple, Tensor)) - A tuple or Tensor of shape(N). It means the lengths of the target.

    Outputs:
        - **neg_log_likelihood** (Tensor) - A loss value which is differentiable with respect to each input node.
        - **log_alpha** (Tensor) - The probability of possible trace of input to target.

    Raises:
        TypeError: If `zero_infinity` is not a bool, reduction is not string.

    Supported Platforms:

    """

    @prim_attr_register
    def __init__(self, blank, reduction="none", zero_infinity=False):
        """Initialize CTCLossV2"""
        self.init_prim_io_names(inputs=["log_probs", "targets", "input_lengths", "target_lengths"],
                                outputs=["neg_log_likelihood", "log_alpha"])
        validator.check_value_type("blank", blank, [int], self.name)
        self.add_prim_attr("blank", blank)
        validator.check_value_type("reduction", reduction, [str], self.name)
        self.reduction = reduction.lower()
        validator.check_string(self.reduction, ['none'], 'reduction', self.name)
        self.add_prim_attr("reduction", self.reduction)
        validator.check_value_type("zero_infinity", zero_infinity, [bool], self.name)
        self.add_prim_attr("zero_infinity", zero_infinity)


class CTCLossV2Grad(Primitive):
    """
    Calculates the gradient of CTC (Connectionist Temporal Classification) loss.

    The CTC algorithm is proposed in `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with
    Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_.

    Args:
        blank (int): The blank label. Default: 0.
        reduction (string): Apply specific reduction method to the output. Currently only support 'none'.
          Default: "none".
        zero_infinity (bool): Whether to set infinite loss and correlation gradient to zero. Default: False.

    Inputs:
        - **grad_out** (Tenosr) - Gradient renewal codfficient, A tensor for shape (N), where N is batch size.
        - **log_probs** (Tensor) - A tensor of shape (T, N, C), where T is input length, N is batch size and C is number
            of classes (including blank).
        - **targets** (Tensor) - A tensor of shape (N, S), where S is max target length, means the target sequences.
        - **input_lengths** (Union(tuple, Tensor)) - A tuple or Tensor of shape(N). It means the lengths of the input.
        - **target_lengths** (Union(tuple, Tensor)) - A tuple or Tensor of shape(N). It means the lengths of the target.
        - **log_alpha** (Tensor) - The probability of possible trace of input to target.
        - **neg_log_likelihood** (Tensor) - A loss value which is differentiable with respect to each input node.

    Outputs:
        - **grad** (Tensor) - The grad of Connectionist Temporal Classification Loss

    Raises:
        TypeError: If `zero_infinity` is not a bool, reduction is not string.

    Supported Platforms:
        ``Ascend``
    """

    @prim_attr_register
    def __init__(self, blank, reduction="none", zero_infinity=False):
        """Initialize CTCLossV2Grad"""
        self.init_prim_io_names(inputs=["grad_out", "log_probs", "targets", "input_lengths", "target_lengths",
                                        "neg_log_likelihood", "log_alpha"],
                                outputs=["grad"])
        validator.check_value_type("blank", blank, [int], self.name)
        self.add_prim_attr("blank", blank)
        validator.check_value_type("reduction", reduction, [str], self.name)
        self.add_prim_attr("reduction", reduction)
        validator.check_value_type("zero_infinity", zero_infinity, [bool], self.name)
        self.add_prim_attr("zero_infinity", zero_infinity)


class Conv3DTranspose(PrimitiveWithInfer):
    r"""
    Computes a 3D transposed convolution, which is also known as a deconvolution
    (although it is not an actual deconvolution).

    Input is typically of shape :math:`(N, C, D, H, W)`, where :math:`N` is batch size, :math:`C` is channel number,
    :math:`D` is depth, :math:`H` is height, :math:`W` is width.

    If the 'pad_mode' is set to be "pad", the depth, height and width of output are defined as:

    .. math::
        D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{pad}[0] + \text{dilation}[0]
        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1

        H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{pad}[1] + \text{dilation}[1]
        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

        W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{pad}[2] + \text{dilation}[2]
        \times (\text{kernel\_size}[2] - 1) + \text{output\_padding}[2] + 1

    Args:
        in_channel (int): The channel of the input x.
        out_channel (int): The channel of the weight x.
        kernel_size (Union[int, tuple[int]]): The data type is int or a tuple of 3 integers.
            Specifies the depth, height and width of the 3D convolution window.
            Single int means the value is for the depth, height and the width of the kernel.
            A tuple of 3 ints means the first value is for the depth, second value is for height and the
            other is for the width of the kernel.
        mode (int): Modes for different convolutions. Default is 1. It is currently not used.
        pad_mode (str): Specifies padding mode. The optional values are
            "same", "valid", "pad", not case sensitive. Default: "valid".

            - same: Adopts the way of completion. The depth, height and width of the output will be the same as
              the input. The total number of padding will be calculated in depth, horizontal and vertical
              directions and evenly distributed to head and tail, top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the tail, bottom and the right side.
              If this mode is set, `pad` and `output_padding` must be 0.

            - valid: Adopts the way of discarding. The possible largest depth, height and width of output
              will be returned without padding. Extra pixels will be discarded. If this mode is set, `pad`
              and `output_padding` must be 0.

            - pad: Implicit paddings on both sides of the input in depth, height, width. The number of `pad` will
              be padded to the input Tensor borders. `pad` must be greater than or equal to 0.

        pad (Union(int, tuple[int])): The pad value to be filled. Default: 0. If `pad` is an integer, the paddings of
             head, tail, top, bottom, left and right are the same, equal to pad. If `pad` is a tuple of six integers,
             the padding of head, tail, top, bottom, left and right equal to pad[0], pad[1], pad[2], pad[3], pad[4]
             and pad[5] correspondingly.
        stride (Union(int, tuple[int])): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both strides, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: 1.
        dilation (Union(int, tuple[int])): Specifies the space to use between kernel elements. Default: 1.
        group (int): Splits input into groups. Default: 1. Only 1 is currently supported.
        output_padding (Union(int, tuple[int])): Add extra size to each dimension of the output. Default: 0.
        data_format (str): The optional value for data format. Currently only 'NCDHW' is supported.

    Inputs:
        - **dout** (Tensor) - The gradients with respect to the output of the convolution.
          The shape conforms to the default.
          data_format :math:`(N, C_{in}, D_{out}, H_{out}, W_{out})`. Currently dout data type only supports float16
          and float32.
        - **weight** (Tensor) - Set size of kernel is :math:`(K_d, K_h, K_w)`, then the shape is
          :math:`(C_{in}, C_{out}//group, K_d, K_h, K_w)`. Where :math:`group` is the Args parameter.
          Currently weight data type only supports float16 and float32.
        - **bias** (Tensor) - Tensor of shape :math:`C_{out}`. Currently, only support none.

    Outputs:
        Tensor, the gradients with respect to the input of convolution 3D.
        Tensor of shape :math:`(N, C_{out}//group, D_{out}, H_{out}, W_{out})`,
        where :math:`group` is the Args parameter.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Raises:
        TypeError: If `in_channel`, `out_channel` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `pad` , `dilation` or `output_padding` is neither an int not a tuple.
        ValueError: If `in_channel`, `out_channel`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `pad` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.
        ValueError: If `pad` is a tuple whose length is not equal to 6.
        ValueError: If `pad_mode` is not equal to 'pad' and `pad` is not equal to (0, 0, 0, 0, 0, 0).
        ValueError: If `data_format` is not 'NCDHW'.
        TypeError: If dout and weight data type is not float16.
        ValueError: If bias is not none. The rank of dout and weight is not 5.

    Examples:
        >>> dout = Tensor(np.ones([32, 16, 10, 32, 32]), mindspore.float16)
        >>> weight = Tensor(np.ones([16, 3, 4, 6, 2]), mindspore.float16)
        >>> conv3d_transpose = ops.Conv3DTranspose(in_channel=16, out_channel=3, kernel_size=(4, 6, 2))
        >>> output = conv3d_transpose(dout, weight)
        >>> print(output.shape)
        (32, 3, 13, 37, 33)
    """

    @prim_attr_register
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 mode=1,
                 pad_mode='valid',
                 pad=0,
                 stride=1,
                 dilation=1,
                 group=1,
                 output_padding=0,
                 data_format="NCDHW"):
        """Initialize Conv3DTranspose"""
        self.init_prim_io_names(inputs=['x', 'filter'], outputs=['output'])
        self.in_channel = validator.check_positive_int(in_channel, 'in_channel', self.name)
        self.add_prim_attr('in_channel', self.in_channel)
        self.out_channel = validator.check_positive_int(out_channel, 'out_channel', self.name)
        self.add_prim_attr('out_channel', self.out_channel)
        self.kernel_size = _check_3d_int_or_tuple('kernel_size', kernel_size, self.name)
        self.stride = _check_3d_int_or_tuple('stride', stride, self.name, allow_five=False,
                                             ret_five=True)
        self.add_prim_attr('strides', self.stride)
        self.dilation = _check_3d_int_or_tuple('dilation', dilation, self.name, allow_five=False,
                                               ret_five=True, third_one=True)
        self.add_prim_attr('dilations', self.dilation)
        validator.check_value_type('pad', pad, (int, tuple), self.name)
        if isinstance(pad, int):
            pad = (pad,) * 6
        if len(pad) != 6:
            raise ValueError(f"For '{self.name}', attr 'pad' should be an positive int number or a tuple of "
                             f"six positive int numbers, but got `{pad}`.")
        self.pad_list = pad
        validator.check_value_type('pad_mode', pad_mode, [str], self.name)
        self.pad_mode = validator.check_string(pad_mode.lower(), ['valid', 'same', 'pad'], 'pad_mode', self.name)
        self.add_prim_attr('pad_mode', self.pad_mode)

        if self.pad_mode != 'pad' and pad != (0, 0, 0, 0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'pad' must be (0, 0, 0, 0, 0, 0) when 'pad_mode' is not \"pad\", "
                             f"but got 'pad' is {pad} and 'pad_mode' is {self.pad_mode}.")

        if self.pad_mode == 'pad':
            for item in self.pad_list:
                validator.check_non_negative_int(item, 'pad item', self.name)
        self.mode = validator.check_equal_int(mode, 1, 'mode', self.name)
        self.add_prim_attr('mode', self.mode)
        self.group = validator.check_equal_int(group, 1, 'group', self.name)
        self.add_prim_attr('groups', self.group)
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.name)
        self.add_prim_attr('data_format', self.format)

        self.output_padding = _check_3d_int_or_tuple('output_padding', output_padding, self.name,
                                                     allow_five=False, ret_five=True, greater_zero=False)
        output_padding = (self.output_padding[2], self.output_padding[3], self.output_padding[4])
        if self.pad_mode != 'pad' and output_padding != (0, 0, 0):
            raise ValueError(f"For '{self.name}', the 'output_padding' must be (0, 0, 0) "
                             f"when 'pad_mode' is not \"pad\", "
                             f"but got 'output_padding' is {output_padding} and 'pad_mode' is {self.pad_mode}.")
        validator.check_int_range(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], 1, 343, Rel.INC_BOTH,
                                  'The product of height, width and depth of kernel_size belonging [1, 343]', self.name)
        validator.check_int_range(self.stride[0] * self.stride[1] * self.stride[2], 1, 343, Rel.INC_BOTH,
                                  'The product of height, width and depth of stride belonging [1, 343]', self.name)
        validator.check_int_range(self.stride[1] * self.stride[2], 1, 256, Rel.INC_BOTH,
                                  'The product of height, width and depth of stride belonging [1, 256]', self.name)
        validator.check_int_range(self.output_padding[2], 0, max(self.dilation[2], self.stride[2]), Rel.INC_LEFT,
                                  'output_padding_d belonging [0, max(stride_d, dilation_d))', self.name)
        validator.check_int_range(self.output_padding[3], 0, max(self.dilation[3], self.stride[3]), Rel.INC_LEFT,
                                  'output_padding_h belonging [0, max(stride_h,dilation_h))', self.name)
        validator.check_int_range(self.output_padding[4], 0, max(self.dilation[4], self.stride[4]), Rel.INC_LEFT,
                                  'output_padding_w belonging [0, max(stride_w,dilation_w))', self.name)

    def __infer__(self, x, w, b=None):
        args = {'x': x['dtype'], 'w': w['dtype']}
        if b is not None:
            raise ValueError(f"For '{self.name}', the 'bias' currently only support None, but got {b}.")
        valid_dtypes = [mstype.float16, mstype.float32]
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)

        # infer shape
        x_shape = x['shape']
        w_shape = w['shape']
        validator.check_equal_int(len(w_shape), 5, "weight rank", self.name)
        validator.check_equal_int(len(x_shape), 5, "x rank", self.name)
        validator.check("filter's batch", w_shape[0], "input x's channel",
                        x_shape[1], Rel.EQ, self.name)

        kernel_d, kernel_h, kernel_w = self.kernel_size
        _, _, stride_d, stride_h, stride_w = self.stride
        _, _, dilation_d, dilation_h, dilation_w = self.dilation

        if self.pad_mode == "valid":
            d_out = _deconv_output_length(x_shape[2], kernel_d, stride_d, dilation_d)
            h_out = _deconv_output_length(x_shape[3], kernel_h, stride_h, dilation_h)
            w_out = _deconv_output_length(x_shape[4], kernel_w, stride_w, dilation_w)
            self.pad_list = (0, 0, 0, 0, 0, 0)
            self.output_padding = (0, 0, 0, 0, 0)

        elif self.pad_mode == "same":
            d_out = x_shape[2] * stride_d
            h_out = x_shape[3] * stride_h
            w_out = x_shape[4] * stride_w

            pad_needed_d = max(0, (x_shape[2] - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - d_out)
            pad_head = math.floor(pad_needed_d / 2)
            pad_tail = pad_needed_d - pad_head

            pad_needed_h = max(0, (x_shape[3] - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - h_out)
            pad_top = math.floor(pad_needed_h / 2)
            pad_bottom = pad_needed_h - pad_top

            pad_needed_w = max(0, (x_shape[4] - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - w_out)
            pad_left = math.floor(pad_needed_w / 2)
            pad_right = pad_needed_w - pad_left
            self.pad_list = (pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right)
            self.output_padding = (0, 0, 0, 0, 0)

        elif self.pad_mode == 'pad':
            pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right = self.pad_list
            d_out = (x_shape[2] - 1) * self.stride[2] - (pad_head + pad_tail) + self.dilation[2] * \
                    (self.kernel_size[0] - 1) + self.output_padding[2] + 1
            h_out = (x_shape[3] - 1) * self.stride[3] - (pad_top + pad_bottom) + self.dilation[3] * \
                    (self.kernel_size[1] - 1) + self.output_padding[3] + 1
            w_out = (x_shape[4] - 1) * self.stride[4] - (pad_left + pad_right) + self.dilation[4] * \
                    (self.kernel_size[2] - 1) + self.output_padding[4] + 1

        self.add_prim_attr('pad_list', self.pad_list)
        self.add_prim_attr('output_padding', self.output_padding)
        output_shape = (x_shape[0], w_shape[1] * self.group, d_out, h_out, w_out)
        self.add_prim_attr('input_size', output_shape)
        out = {
            'value': None,
            'shape': output_shape,
            'dtype': x['dtype'],
        }
        return out


class SoftShrink(Primitive):
    r"""
    Applies the soft shrinkage function elementwise.

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd: the :math:`\lambda` must be no less than zero value for the Softshrink formulation. Default: 0.5.

    Inputs:
        - **input_x** (Tensor) - The input of SoftShrink with data type of float16 or float32.
          Any number of additional dimensions.

    Outputs:
        Tensor, has the same shape and data type as `input_x`.

    Raises:
        TypeError: If lambd is not a float.
        TypeError: If input_x is not a Tensor.
        TypeError: If dtype of input_x is neither float16 nor float32.
        ValueError: If lambd is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> input_x = Tensor(np.array([[ 0.5297,  0.7871,  1.1754], [ 0.7836,  0.6218, -1.1542]]), mindspore.float16)
        >>> softshrink = ops.SoftShrink()
        >>> output = softshrink(input_x)
        >>> print(output)
        [[ 0.02979  0.287    0.676  ]
         [ 0.2837   0.1216  -0.6543 ]]
    """

    @prim_attr_register
    def __init__(self, lambd=0.5):
        """Initialize SoftShrink"""
        validator.check_value_type("lambd", lambd, [float], self.name)
        validator.check_number("lambd", lambd, 0, Rel.GE, self.name)


class HShrink(Primitive):
    r"""
    Applies the hard shrinkage function element-wise, each element complies the follow function:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        lambd (float): The value for the HardShrink formulation. Default: 0.5

    Inputs:
        - **input_x** (Tensor) - The input of HardShrink with data type of float16 or float32.

    Outputs:
        Tensor, the same shape and data type as the input.

    Supported Platforms:
        ``Ascend``

    Raises:
        TypeError: If `lambd` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Examples:
        >>> input_x = Tensor(np.array([[ 0.5,  1,  2.0],[0.0533,0.0776,-2.1233]]),mstype.float32)
        >>> hshrink = P.HShrink()
        >>> output = hshrink(input_x)
        >>> print(output)
        [[ 0.      1.      2.    ]
        [ 0.      0.     -2.1233]]
    """

    @prim_attr_register
    def __init__(self, lambd=0.5):
        """Initialize HShrink"""
        validator.check_value_type('lambd', lambd, [float], self.name)
        if lambd < 0.0:
            lambd = 0.0
            self.add_prim_attr('lambd', lambd)


class ApplyAdagradDA(Primitive):
    r"""
    Update `var` according to the proximal adagrad scheme.

    .. math::
        \begin{array}{ll} \\
            grad_accum += grad \\
            grad_squared_accum += grad * grad \\
            tmp_val=sign(grad_accum) * max\left \{|grad_accum|-l1*global_step, 0\right \}
                    if l1>0 else grad_accum \\
            x_value = -1 * lr * tmp_val \\
            y_value = l2 * global_step * lr + \sqrt{grad_squared_accum} \\
            var = x_value / y_value
        \end{array}

    Inputs of `var`, `gradient_accumulator`, `gradient_squared_accumulator` and `grad`
    comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        use_locking (bool): If `True`, updating of the `var` and `accum` tensors will be protected by a lock.
                            Otherwise the behavior is undefined, but may exhibit less contention. Default: False.

    Inputs:
        - **var** (Parameter) - Variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **gradient_accumulator** (Parameter) - The dict of mutable tensor gradient_accumulator. Must have the same
          shape and dtype as `var`.
        - **gradient_squared_accumulator** (Parameter) - The dict of mutable tensor gradient_squared_accumulator.
          Must have the same shape and dtype as `var`.
        - **grad** (Tensor) - A tensor for gradient. Must have the same shape and dtype as `var`.
        - **lr** ([Number, Tensor]) - Scaling factor. Must be a scalar. With float32 or float16 data type.
        - **l1** ([Number, Tensor]) -  L1 regularization. Must be a scalar. With float32 or float16 data type.
        - **l2** ([Number, Tensor]) -  L2 regularization. Must be a scalar. With float32 or float16 data type.
        - **global_step** ([Number, Tensor]) - Training step number. Must be a scalar. With int32 or int64 data type.

    Outputs:
        Tuple of 3 Tensors, the updated parameters.

        - **var** (Tensor) - The same shape and data type as `var`.
        - **gradient_accumulator** (Tensor) - The same shape and data type as `gradient_accumulator`.
        - **gradient_squared_accumulator** (Tensor) - The same shape and data type as `gradient_squared_accumulator`.

    Raises:
        TypeError: If `var`, `gradient_accumulator`, `gradient_squared_accumulator` is not a Parameter.
        TypeError: If `grad` is not a Tensor.
        TypeError: If `lr`, `l1`, `l2` or `global_step` is neither a Number nor a Tensor.
        TypeError: If use_locking is not a bool.
        TypeError: If dtype of `var`, `gradient_accumulator`, `gradient_squared_accumulator`, `gradient_accumulator`,
                   `lr`, `l1`, `l2` is neither float16 nor float32.
        TypeError: If dtype of `gradient_accumulator`, `gradient_squared_accumulator`, `gradient_accumulator`
                     is not same as `var`.
        TypeError: If dtype of `global_step` is not int32 or int64.
        ValueError: If the shape size of `lr`, `l1`, `l2` and `global_step` is not 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class ApplyAdagradDANet(nn.Cell):
        ...     def __init__(self, use_locking=False):
        ...         super(ApplyAdagradDANet, self).__init__()
        ...         self.apply_adagrad_d_a = P.ApplyAdagradDA(use_locking)
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
        ...                                                                [0.1, 0.5]]).astype(np.float32)),
        ...                                               name="gradient_accumulator")
        ...         self.gradient_squared_accumulator = Parameter(Tensor(np.array([[0.2, 0.1],
        ...                                                                        [0.1, 0.2]]).astype(np.float32)),
        ...                                                       name="gradient_squared_accumulator")
        ...         self.gradient_accumulator = Parameter(Tensor(np.array([[0.1, 0.3],
        ...                                                                [0.1, 0.5]]).astype(np.float32)),
        ...                                               name="gradient_accumulator")
        ...     def construct(self, grad, lr, l1, l2, global_step):
        ...         out = self.apply_adagrad_d_a(self.var, self.gradient_accumulator,
        ...                                      self.gradient_squared_accumulator, grad, lr, l1, l2, global_step)
        ...         return out
        ...
        >>> net = ApplyAdagradDANet()
        >>> grad = Tensor(np.array([[0.3, 0.4], [0.1, 0.2]]).astype(np.float32))
        >>> lr = Tensor(0.001, mstype.float32)
        >>> l1 = Tensor(0.001, mstype.float32)
        >>> l2 = Tensor(0.001, mstype.float32)
        >>> global_step = Tensor(2, mstype.int32)
        >>> output = net(grad, lr, l1, l2, global_step)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[-7.39064650e-04, -1.36888528e-03],
         [-5.96988888e-04, -1.42478070e-03]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 4.00000006e-01,  7.00000048e-01],
         [ 2.00000003e-01,  6.99999988e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.90000021e-01,  2.60000020e-01],
         [ 1.09999999e-01,  2.40000010e-01]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('gradient_accumulator', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('gradient_squared_accumulator', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('l1', dtype=sig.sig_dtype.T2),
        sig.make_sig('l2', dtype=sig.sig_dtype.T3),
        sig.make_sig('global_step', dtype=sig.sig_dtype.T4)
    )

    @prim_attr_register
    def __init__(self, use_locking=False):
        """Initialize ApplyAdagradDA"""
        validator.check_value_type("use_locking", use_locking, [bool], self.name)


class SparseApplyRMSProp(Primitive):
    r"""
    Update relevant entries according to the rmsprop algorithm.

    .. math::
        \begin{array}{ll} \\
            ms = rho * ms_{t-1} + (1 - rho) * grad * grad \\
            mom = momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon) \\
            var = var - mom
        \end{array}

    Inputs of `var`, `ms`, `mom` and `grad` comply with the implicit type conversion rules
    to make the data types consistent.
    If they have different data types, lower priority data type will be converted to
    relatively highest priority data type.
    RuntimeError exception will be thrown when the data type conversion of Parameter is required.

    Args:
        rho (float): Decay rate. The value should between 0 and 1, otherwise the behavior is undefined.
        momentum (float): Momentum. The value should be greater or equal to 0, otherwise the behavior is undefined.
        epsilon (float): A small value added for numerical stability. The value should be greater than 0,
                         otherwise the behavior is undefined.
        use_locking (bool): If `True`, updating of the var, ms, and mom tensors is protected by a lock;
                            otherwise the behavior is undefined, but may exhibit less contention. Default: False.

    Inputs:
        - **var** (Parameter) - Variable to be updated. The data type must be float16 or float32.
          The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        - **ms** (Parameter) - The dict of mutable tensor ms. Must have the same shape and dtype as `var`.
        - **mom** (Parameter) - The dict of mutable tensor mom. Must have the same shape and dtype as `var`.
        - **lr** ([Number, Tensor]) - Learning rate. Must be a scalar. With float16 or float32 data type.
        - **grad** (Tensor) - A tensor for gradient. Must have the same shape and dtype as `var`.
        - **indices** (Tensor) - A tensor of indices in the first dimension of `var`, `ms` and `mom`.
          If there are duplicates in `indices`, the behavior is undefined. Must be one of the
          following types: int32, int64 and indices.shape[0] = var.shape[0].

    Outputs:
        Tuple of 3 Tensors, the updated parameters.

        - **var** (Tensor) -  The same shape and data type as `var`.
        - **ms** (Tensor) - The same shape and data type as `ms`.
        - **mom** (Tensor) - The same shape and data type as `mom`.

    Raises:
        TypeError: If `var`, `ms` or `mom` is not a Parameter.
        TypeError: If `grad` or `indices` is not a Tensor.
        TypeError: If dtype of `var`, `ms`, `mom`, `lr`, `grad` is neither float16 nor float32.
        TypeError: If dtype of `indices` is neither int32 nor int64.
        TypeError: If `lr` is neither a Number or a Tensor.
        TypeError: If `use_locking` is not a bool.
        TypeError: If dtype of `epsilon`, `rho`, `momentum` is not a float.
        ValueError: If shape of `ms`, `mom`, `grad` is not same as `var`.
        ValueError: If the shape size of `lr` is not 0.
        ValueError: If shape of `indices` is not same as shape of first dimension of `var`.
        ValueError: If `epsilon` is less than or equal to 0.
        ValueError: If `momentum` is less than 0.
        ValueError: If `rho` is less than 0 or greater than 1.
        ValueError: If dimension of `var` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> class SparseApplyRMSPropNet(nn.Cell):
        ...     def __init__(self, rho, momentum, epsilon, use_locking=False):
        ...         super(SparseApplyRMSPropNet, self).__init__()
        ...         self.sparse_apply_r_m_s_prop = P.SparseApplyRMSProp(rho, momentum, epsilon, use_locking)
        ...         self.var = Parameter(Tensor(np.array([[0.6, 0.3], [0.1, 0.5]]).astype(np.float32)), name="var")
        ...         self.ms = Parameter(Tensor(np.array([[0.2, 0.4], [0.1, 0.3]]).astype(np.float32)), name="ms")
        ...         self.mom = Parameter(Tensor(np.array([[0.3, 0.1], [0.3, 0.6]]).astype(np.float32)), name="mom")
        ...     def construct(self, lr, grad, indices):
        ...         out = self.sparse_apply_r_m_s_prop(self.var, self.ms, self.mom, lr, grad, indices)
        ...         return out
        ...
        >>> rho = 0.2
        >>> momentum = 0.01
        >>> epsilon = 1e-6
        >>> net = SparseApplyRMSPropNet(rho, momentum, epsilon)
        >>> lr = 0.01
        >>> grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
        >>> indices = Tensor(np.array([0, 1], dtype=np.int32))
        >>> out = net(lr, grad, indices)
        >>> print(out)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 5.88035822e-01,  2.88811117e-01],
         [ 9.10239667e-02,  4.83422279e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.12000003e-01,  4.72000003e-01],
         [ 2.80000009e-02,  5.72000027e-01]]), Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 1.19641740e-02,  1.11888833e-02],
         [ 8.97603668e-03,  1.65777095e-02]]))
    """

    __mindspore_signature__ = (
        sig.make_sig('var', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('ms', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('mom', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('lr', dtype=sig.sig_dtype.T1),
        sig.make_sig('grad', dtype=sig.sig_dtype.T),
        sig.make_sig('indices', dtype=sig.sig_dtype.T2)
    )

    @prim_attr_register
    def __init__(self, rho, momentum, epsilon, use_locking=False):
        """"Initialize SparseApplyRMSProp"""
        validator.check_value_type("rho", rho, [float], self.name)
        validator.check_value_type("momentum", momentum, [float], self.name)
        validator.check_value_type("epsilon", epsilon, [float], self.name)
        validator.check_value_type("use_locking", use_locking, [bool], self.name)
        self.epsilon = validator.check_number("epsilon", epsilon, 0.0, Rel.GT, self.name)
        self.momentum = validator.check_number("momentum", momentum, 0.0, Rel.GE, self.name)
        self.rho = validator.check_float_range(rho, 0.0, 1.0, Rel.INC_BOTH, "rho", self.name)
