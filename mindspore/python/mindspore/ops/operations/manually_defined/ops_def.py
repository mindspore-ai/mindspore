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
"""Operators for nn."""
from __future__ import absolute_import
from __future__ import division

import numbers
import math
import numpy as np
from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register, prim_arg_register, PrimitiveWithInfer
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_arg_handler as handler
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore.common._stub_tensor import _convert_stub
from mindspore._c_expression import typing
from mindspore._c_expression import Tensor as Tensor_
from mindspore._c_expression import pyboost_cast, pyboost_tile, pyboost_zeros, pyboost_ones
from mindspore.common import dtype as mstype
from mindspore.common._utils import is_shape_unknown
from mindspore import _checkparam as validator
from mindspore.ops.operations.manually_defined._inner import ScalarCast
from mindspore.ops_generate.gen_ops_inner_prim import DtypeToEnum
from mindspore.common.initializer import Zero
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate.gen_ops_prim import FlashAttentionScore


dtype_to_type_id = DtypeToEnum()


dtype_to_type_id = DtypeToEnum()


class ScalarDiv(Primitive):
    r"""
    Computes the quotient of dividing the first input scalar by the second input scalar element-wise.

    .. math::

        out_{i} = \frac{x_i}{y_i}

    .. note::
        The inputs can be constant/variable value. Usage is the same as '/' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is float.

    Raises:
        TypeError: If `x` and `y` are not scalar.
        ValueError: If `y` is 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarDiv"""

    def __call__(self, x, y):
        if y == 0:
            raise ValueError('The divisor could not be zero. But the divisor is zero now.')
        return x / y


class ScalarFloorDiv(Primitive):
    r"""
    Computes the quotient of dividing the first input scalar by the second input scalar element-wise.

    .. math::

        out_{i} = \frac{x_i}{y_i}

    .. note::
        The inputs can be constant/variable value. Usage is the same as '//' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is float.

    Raises:
        TypeError: If `x` and `y` are not scalar.
        ValueError: If `y` is 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarFloorDiv"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def __call__(self, x, y):
        if y == 0:
            raise ValueError('The divisor could not be zero. But the divisor is zero now.')
        return x // y


class ScalarAdd(Primitive):
    r"""
    Adds two input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarAdd"""

    def __call__(self, x, y):
        return x + y


class ScalarPow(Primitive):
    r"""
    Pow two input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarPow"""

    def __call__(self, x, y):
        return pow(x, y)


class ScalarLog(Primitive):
    r"""
    Log input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarAdd"""

    def __call__(self, x):
        return math.log(x)


class ScalarUadd(Primitive):
    r"""
    UAdds input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarAdd"""

    def __call__(self, x):
        return x


class ScalarUsub(Primitive):
    r"""
    usub input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarUsub"""

    def __call__(self, x):
        return -x


class ScalarSub(Primitive):
    r"""
    Subtracts the second input Scalar from the first input Scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '-' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarSub"""

    def __call__(self, x, y):
        return x - y


class ScalarMul(Primitive):
    r"""
    Muls two input scalar.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '+' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarMul"""

    def __call__(self, x, y):
        return x * y


class ScalarEq(Primitive):
    r"""
    Computes the equivalence between two Scalars.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '==' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is bool.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarEq"""

    def __call__(self, x, y):
        return x == y


class ScalarGt(Primitive):
    r"""
    Compare the value of the input scalars :math:`x,y`, and the output result is a bool value.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '>' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is bool.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize scalar_gt"""

    def __call__(self, x, y):
        return x > y


class ScalarLt(Primitive):
    r"""
    Computes the boolean value of :math:`x < y`.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '<' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is bool.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize scalar_lt"""

    def __call__(self, x, y):
        return x < y


class ScalarGe(Primitive):
    r"""
    Compare the value of the input scalars :math:`x,y`, and the output result is a bool value.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '>=' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is bool.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize scalar_ge"""

    def __call__(self, x, y):
        return x >= y


class ScalarLe(Primitive):
    r"""
    Compare the value of the input scalars :math:`x,y`, and the output result is a bool value.

    .. note::
        The inputs can be constant/variable value. Usage is the same as '<=' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type of scalar is bool.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize scalar_le"""

    def __call__(self, x, y):
        return x <= y


class ScalarMod(Primitive):
    r"""
    Computes the remainder of dividing the first input scalar by the second input scalar element-wise.

    .. math::

        out_{i} = x_{i} \text{ % } y_{i}

    .. note::
        The inputs can be constant/variable value. Usage is the same as '%' in Python.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.
        - **y** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarMod"""

    def __call__(self, x, y):
        if y == 0:
            raise ValueError('Cannot perform modulo operation on zero.')
        return x % y


class ScalarBool(Primitive):
    r"""
    Computes the input scalar true or false.

    .. note::
        The inputs can be constant/variable value.
        This primitive only have 'CPU' implementation, for other platform, it runs using heterogeneous.

    Inputs:
        - **x** (Scalar) - A constant or variable scalar.

    Outputs:
        Scalar, the type is bool.

    Raises:
        TypeError: If `x` are not scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    @prim_attr_register
    def __init__(self):
        """Initialize ScalarBool"""

    def __call__(self, x):
        return bool(x)


scalar_div = ScalarDiv()
scalar_mod = ScalarMod()
scalar_add = ScalarAdd()
scalar_mul = ScalarMul()
scalar_sub = ScalarSub()
scalar_gt = ScalarGt()
scalar_ge = ScalarGe()
scalar_le = ScalarLe()
scalar_lt = ScalarLt()
scalar_eq = ScalarEq()
scalar_bool = ScalarBool()
scalar_floordiv = ScalarFloorDiv()
scalar_log = ScalarLog()
scalar_pow = ScalarPow()
scalar_uadd = ScalarUadd()
scalar_usub = ScalarUsub()


class BatchNorm(Primitive):
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

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon,
    :math:`mean` is the mean of :math:`x`,
    :math:`variance` is the variance of :math:`x`.

    .. warning::
        - If the operation is used for inference, and outputs "reserve_space_1" and "reserve_space_2" are available,
          then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
        - For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.

    Args:
        is_training (bool): If `is_training` is ``True`` , `mean` and `variance` are computed during training.
            If `is_training` is ``False`` , they're loaded from checkpoint during inference. Default: ``False`` .
        epsilon (float): A small value added for numerical stability. Default: ``1e-5``, value must be (0, 1] .
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1]. Default: ``0.1`` .
        data_format (str): The optional value for data format, is ``'NHWC'`` or ``'NCHW'``, and the ``'NHWC'`` format
            is only supported in GPU target. Default: ``"NCHW"`` .

    Inputs:
        If `is_training` is ``False`` , inputs are Tensors.

        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Tensor) - Tensor of shape :math:`(C,)`, with float16 or float32 data type.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`, has the same data type with `scale`.

        If `is_training` is ``True`` , `scale`, `bias`, `mean` and `variance` are Parameters.

        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Parameter) - Parameter of shape :math:`(C,)`, with float16 or float32 data type.
        - **bias** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.
        - **mean** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.
        - **variance** (Parameter) - Parameter of shape :math:`(C,)`, has the same data type with `scale`.

    Outputs:
        Tuple of 5 Tensors, the normalized inputs and the updated parameters.

        - **output_x** (Tensor) - The same type and shape as the input_x. The shape is :math:`(N, C)`.
        - **batch_mean** (Tensor) - The mean calculated per-dimension over the mini-batches,
          shape is :math:`(C,)`.
        - **batch_variance** (Tensor) - The variance calculated per-dimension over the mini-batches,
          shape is :math:`(C,)`.
        - **reserve_space_1** (Tensor) - The mean that needs to be reused when calculating gradients,
          one-dimensional Tensor. The shape is :math:`(C,)`.
        - **reserve_space_2** (Tensor) - The variance that needs to be reused when calculating gradients,
          one-dimensional Tensor. The shape is :math:`(C,)`.

    Raises:
        TypeError: If `is_training` is not a bool.
        TypeError: If dtype of `epsilon` or `momentum` is not float.
        TypeError: If `data_format` is not a str.
        TypeError: If `input_x`, `scale`, `bias`, `mean` or `variance` is not a Tensor.
        TypeError: If dtype of `input_x`, `scale` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
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
    __mindspore_signature__ = (sig.make_sig('input_x', dtype=sig.sig_dtype.T1),
                               sig.make_sig('scale',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T2),
                               sig.make_sig('bias',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T2),
                               sig.make_sig('mean',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T3),
                               sig.make_sig('variance',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T3))

    @prim_arg_register
    def __init__(self,
                 is_training=False,
                 epsilon=1e-5,
                 momentum=0.1,
                 data_format="NCHW"):
        """Initialize BatchNorm."""
        if is_training is False:
            self.set_signatures(tuple())
        else:
            self.add_prim_attr('side_effect_mem', True)
        self.is_training = is_training
        self.epsilon = epsilon
        self.momentum = momentum
        self.data_format = handler.str_to_enum("BatchNorm", "data_format", data_format)

    def __call__(self, *args):
        return super().__call__(*args, self.is_training, self.epsilon,
                                self.momentum, self.data_format)


def batch_norm_(input_x,
                scale,
                bias,
                mean,
                variance,
                is_training=False,
                epsilon=1e-5,
                momentum=0.1,
                data_format="NCHW"):
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

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon,
    :math:`mean` is the mean of :math:`x`,
    :math:`variance` is the variance of :math:`x`.

    .. warning::
        - If the operation is used for inference, and outputs "reserve_space_1" and "reserve_space_2" are available,
          then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
        - For Atlas 200/300/500 inference product,
          the result accuracy fails to reach 1‰ due to the square root instruction.

    Note:
        - If `training` is `False`, `weight`, `bias`, `running_mean` and `running_var` are tensors.
        - If `training` is `True`, `weight`, `bias`, `running_mean` and `running_var` are Parameters.

    Args:
        input_x (tensor): tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        scale (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        bias (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        mean (Union[tensor, Parameter]): The shape :math:`(C,)`, with float16 or float32 data type.
        variance (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        is_training (bool, optional): If `training` is `True`, `mean` and `variance` are computed during training.
            If `training` is `False`, they're loaded from checkpoint during inference. Default: False.
        epsilon (float): A small value added for numerical stability.
            Default: ``1e-5``, value must be (0, 1] .
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1].
            Default: ``0.1`` .
        data_format (str): The optional value for data format, is ``'NHWC'`` or ``'NCHW'``,
            and the ``'NHWC'`` format is only supported in GPU target.
            Default: ``"NCHW"`` .

    Returns:
        output_x (Tensor): The same type and shape as the input_x. The shape is :math:`(N, C)`.
        batch_mean (Tensor): Tensor of shape :math:`(C,)`.
        batch_variance (Tensor): Tensor of shape :math:`(C,)`.
        reserve_space_1 (Tensor): Tensor of shape :math:`(C,)`.
        reserve_space_2 (Tensor): Tensor of shape :math:`(C,)`.

    Raises:
        TypeError: If `is_training` is not a bool.
        TypeError: If dtype of `epsilon` or `momentum` is not float.
        TypeError: If `data_format` is not a str.
        TypeError: If `input_x`, `scale`, `bias`, `mean` or `variance` is not a Tensor.
        TypeError: If dtype of `input_x`, `scale` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones([2, 2]), mindspore.float32)
        >>> scale = Tensor(np.ones([2]), mindspore.float32)
        >>> bias = Tensor(np.ones([2]), mindspore.float32)
        >>> mean = Tensor(np.ones([2]), mindspore.float32)
        >>> variance = Tensor(np.ones([2]), mindspore.float32)
        >>> output = ops.batch_norm_(input_x, scale, bias, mean, variance, is_training, epsilon, momentum, data_format)
        >>> print(output[0])
        [[1. 1.]
        [1. 1.]]
    """
    batch_norm_op = _get_cache_prim(BatchNorm)(is_training, epsilon, momentum,
                                               data_format)
    return batch_norm_op(input_x, scale, bias, mean, variance)


class Rank(Primitive):
    """
    Returns the rank of a tensor.

    Refer to :func:`mindspore.ops.rank` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> rank = ops.Rank()
        >>> output = rank(input_tensor)
        >>> print(output)
        2
        >>> print(type(output))
        <class 'int'>
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Rank"""

    def __call__(self, x):
        if not isinstance(x, (Tensor, Tensor_)):
            raise TypeError("the input x must be Tensor!")
        return len(x.shape)


def rank(input_x):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Returns:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.rank(input_tensor)
        >>> print(output)
        2
        >>> print(type(output))
        <class 'int'>

    """
    rank_op = _get_cache_prim(Rank)()
    return rank_op(input_x)


class Shape(Primitive):
    """
    Returns the shape of the input tensor.

    Refer to :func:`mindspore.ops.shape` for more details.

    Inputs:
        - **input_x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Outputs:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> shape = ops.Shape()
        >>> output = shape(input_x)
        >>> print(output)
        (3, 2, 1)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Shape"""

    def __call__(self, x):
        if isinstance(x, (Tensor, COOTensor, CSRTensor, Tensor_)):
            return x.shape
        raise TypeError(f"For primitive[{self.name}], the input argument must be Tensor, but got {type(x)}.")


def shape_(input_x):
    """
    Returns the shape of the input tensor.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        tuple[int], the output tuple is constructed by multiple integers,
        :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = ops.shape(input_x)
        >>> print(output)
        (3, 2, 1)
    """
    shape_op = _get_cache_prim(Shape)()
    return shape_op(input_x)


class ScalarToTensor(PrimitiveWithInfer):
    """
    Converts a scalar to a `Tensor`, and converts the data type to the specified type.

    Refer to :func:`mindspore.ops.scalar_to_tensor` for more details.

    Inputs:
        - **input_x** (Union[int, float]) - The input is a scalar. Only constant value is allowed.
        - **dtype** (mindspore.dtype) - The target data type. Default: ``mindspore.float32`` . Only
          constant value is allowed.

    Outputs:
        Tensor. 0-D Tensor and the content is the input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> op = ops.ScalarToTensor()
        >>> data = 1
        >>> output = op(data, mindspore.float32)
        >>> print(output)
        1.0
    """

    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['input_scalar', 'dtype'], outputs=['output_data'])

    def __call__(self, x, dtype=mstype.float32):
        validator.check_value_type("x", x, [bool, int, float], self.name)
        validator.check_subclass("dtype", dtype, mstype.number, self.name)
        data_type = mstype.dtype_to_nptype(dtype)
        return Tensor(np.array(x, data_type), dtype=dtype)


class Tile(Primitive):
    r"""
    Replicates an input tensor with given multiple times.

    Refer to :func:`mindspore.ops.tile` for more details.

    Inputs:
        - **input** (Tensor) - The tensor whose elements need to be repeated. Set the shape of input tensor as
          :math:`(x_1, x_2, ..., x_S)` .
        - **dims** (tuple[int]) - The parameter that specifies the number of replications,
          the parameter type is tuple, and the data type is int, i.e., :math:`(y_1, y_2, ..., y_S)`.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same data type as the `input`. Suppose the length of `dims` is `d`,
        the dimension of `input` is `input.dim`, and the shape of `input` is :math:`(x_1, x_2, ..., x_S)`.

        - If `input.dim = d`, then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)`.
        - If `input.dim < d`, prepend 1 to the shape of `input` until their lengths are consistent.
          Such as set the shape of `input` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and the shape of Outputs is
          :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)`.
        - If `input.dim > d`, prepend 1 to `dims` until their lengths are consistent. Such as set the
          `dims` as :math:`(1, ..., y_1, y_2, ..., y_S)`, then the shape of their corresponding positions
          can be multiplied, and the shape of Outputs is :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)`.

    Raises:
        TypeError: If `dims` is not a tuple or its elements are not all int.
        ValueError: If the elements of `dims` are not all greater than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> tile = ops.Tile()
        >>> input = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> dims = (2, 3)
        >>> output = tile(input, dims)
        >>> print(output)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
        >>> dims = (2, 3, 2)
        >>> output = tile(input, dims)
        >>> print(output)
        [[[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]
         [[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize."""

    def __call__(self, input, dims):
        return _convert_stub(pyboost_tile(self, [input, dims]))

    # pylint: disable=missing-docstring
    def check_elim(self, *args):
        base_tensor, dims = args
        if not isinstance(base_tensor, Tensor):
            raise TypeError(f"For '{self.name}', the type of 'input' must be Tensor, "
                            f"but got {type(base_tensor).__name__}.")
        if not isinstance(dims, tuple):
            raise TypeError(f"For '{self.name}', the type of 'dims' must be tuple, "
                            f"but got {type(dims).__name__}.")

        if all(v == 1 for v in dims) and len(base_tensor.shape) >= len(dims):
            from mindspore.ops.auto_generate.gen_ops_def import Identity
            ret = Identity()(base_tensor)
            return (True, ret)
        return (False, None)


def tile(input, dims):
    r"""
    Creates a new tensor by replicating `input` `dims` times. The i'th dimension of
    output tensor has `input.shape[i] * dims[i]` elements, and the values of `input`
    are replicated `dims[i]` times along the i'th dimension.

    Args:
        input (Tensor): The tensor whose elements need to be repeated. Set the shape of input tensor as
            :math:`(x_1, x_2, ..., x_S)` .

        dims (tuple[int]): The parameter that specifies the number of replications,
            the parameter type is tuple, and the data type is int, i.e., :math:`(y_1, y_2, ..., y_S)`.
            Only constant value is allowed.

    Returns:
        Tensor, has the same data type as the `input`. Suppose the length of `dims` is `d`,
        the dimension of `input` is `input.dim`, and the shape of `input` is :math:`(x_1, x_2, ..., x_S)`.

        - If `input.dim = d`, then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)`.
        - If `input.dim < d`, prepend 1 to the shape of `input` until their lengths are consistent.
          Such as set the shape of `input` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and the shape of Outputs is
          :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)`.
        - If `input.dim > d`, prepend 1 to `dims` until their lengths are consistent. Such as set the
          `dims` as :math:`(1, ..., y_1, y_2, ..., y_S)`, then the shape of their corresponding positions
          can be multiplied, and the shape of Outputs is :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)`.

    Raises:
        TypeError: If `dims` is not a tuple or its elements are not all int.
        ValueError: If the elements of `dims` are not all greater than or equal to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> dims = (2, 3)
        >>> output = ops.tile(input, dims)
        >>> print(output)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
        >>> dims = (2, 3, 2)
        >>> output = ops.tile(input, dims)
        >>> print(output)
        [[[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]
         [[1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]
          [1. 2. 1. 2.]
          [3. 4. 3. 4.]]]
    """
    tile_op = _get_cache_prim(Tile)()
    return tile_op(input, dims)


def scalar_cast(input_x, input_y):
    r"""
    The interface is deprecated from version 2.3 and will be removed in a future version,
    please use `int(x)` or `float(x)` instead.

    Casts the input scalar to another type.

    Args:
        input_x (scalar): The input scalar.
        input_y (mindspore.dtype): The type to be cast. Only constant value is allowed.
            The value should only be mindspore.int64, mindspore.float64, or mindspore.bool\_.

    Returns:
        Scalar, the type is the same as the python type corresponding to `input_y`.

    Raises:
        ValueError: if input_y's value is invalid.

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> output = ops.scalar_cast(255.0, mindspore.int64)
        >>> print(output)
        255
    """
    scalar_cast_op = _get_cache_prim(ScalarCast)()
    return scalar_cast_op(input_x, input_y)


class Cast(Primitive):
    """
    Returns a tensor with the new specified data type.

    Note:
        When converting complex numbers to boolean type, the imaginary part of the complex number is not
        taken into account. As long as the real part is non-zero, it returns True; otherwise, it returns False.

    Inputs:
        - **input_x** (Union[Tensor, Number]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The tensor to be cast.
        - **type** (dtype.Number) - The valid data type of the output tensor. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is neither Tensor nor Number.
        TypeError: If `type` is not a Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input_x = Tensor(input_np)
        >>> type_dst = mindspore.int32
        >>> cast = ops.Cast()
        >>> output = cast(input_x, type_dst)
        >>> print(output.dtype)
        Int32
        >>> print(output.shape)
        (2, 3, 4, 5)
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Cast"""
        self.init_prim_io_names(inputs=['x', 'dst_type'], outputs=['output'])

    def check_elim(self, x, dtype):
        if isinstance(x, (Tensor, numbers.Number, Parameter)):
            if isinstance(x, Parameter):
                data = x.data
                if data.dtype == dtype:
                    return (True, x)
            if isinstance(x, Tensor) and x.dtype == dtype:
                x = Tensor(x)
                x.set_cast_dtype()
                return (True, x)
            if isinstance(x, numbers.Number):
                return (True, Tensor(x, dtype=dtype))
        return (False, None)

    def __call__(self, input_x, dtype):
        should_elim, output = self.check_elim(input_x, dtype)
        if should_elim:
            return output
        return _convert_stub(pyboost_cast(self, [input_x, dtype_to_type_id('Cast', 'dtype', dtype)]))


def to_sequence(val):
    if isinstance(val, (tuple, list)):
        return val
    return (val,)


class EmbeddingTableExport(Primitive):
    """
    .. code-block::

        prim = ops.EmbeddingTableExport(embedding_dim, value_total_len,
                                        export_mode, only_var_flag, file_type,
                                        table_name, filter_export_flag)
        out = prim(file_path, ps_id, table_id)

    """

    @prim_attr_register
    def __init__(self, embedding_dim, value_total_len, export_mode="all",
                 only_var_flag=False, file_type="bin", table_name=(),
                 filter_export_flag=False, steps_to_live_list=()):
        """Initialize EmbeddingTableExport"""
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingTableImport(Primitive):
    """
    .. code-block::

        prim = ops.EmbeddingTableImport(embedding_dim, value_total_len,
                                        only_var_flag, file_type, table_name)
        out = prim(file_path, ps_id, table_id)

    """

    @prim_attr_register
    def __init__(self, embedding_dim, value_total_len,
                 only_var_flag=False, file_type="bin", table_name=()):
        """Initialize EmbeddingTableImport"""
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingComputeVarImport(Primitive):
    """
    .. code-block::

        prim = ops.EmbeddingComputeVarImport(table_name)
        out = prim(file_path, ps_id, table_id)

    """

    @prim_attr_register
    def __init__(self, table_name=()):
        """Initialize EmbeddingComputeVarImport"""
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingComputeVarExport(Primitive):
    """
    .. code-block::

        prim = ops.EmbeddingComputeVarExport(table_name)
        out = prim(file_path, ps_id, table_id)

    """

    @prim_attr_register
    def __init__(self, table_name=()):
        """Initialize EmbeddingComputeVarExport"""
        self.add_prim_attr("_process_node_engine_id", "PS")


class InitEmbeddingHashmap(Primitive):
    """
    InitEmbeddingHashmap
    """
    @prim_attr_register
    def __init__(self, value_total_len, embedding_dim, _table_id,
                 bucket_size=0, dtype=mstype.float32, initializer_mode="",
                 constant_valu=0., min=-2., max=2., mu=0., sigma=1., seed=0,
                 seed2=0, filter_mode="no_filter", optimizer_mode="",
                 optimizer_params=()):
        self.add_prim_attr("_process_node_engine_id", "PS")


def init_embedding_hashmap(table_id, value_total_len, embedding_dim, _table_id,
                           bucket_size=0, dtype=mstype.float32, initializer_mode='',
                           constant_value=0.0, min=-2.0, max=2.0, mu=0.0, sigma=1.0,
                           seed=0, seed2=0, filter_mode='no_filter',
                           optimizer_mode='', optimizer_params=()):
    """
    init_embedding_hashmap
    """
    op = _get_cache_prim(InitEmbeddingHashmap)(value_total_len, embedding_dim, _table_id,
                                               bucket_size, dtype, initializer_mode,
                                               constant_value, min, max, mu, sigma, seed,
                                               seed2, filter_mode, optimizer_mode, optimizer_params)
    return op(table_id)


class InitPartitionMap(Primitive):
    """
    InitPartitionMap
    """
    @prim_attr_register
    def __init__(self, _embedding_dim, _max_key_num,
                 _ps_num=1, partition_num=65537):
        self.add_prim_attr("_process_node_engine_id", "PS")


def init_partition_map(ps_num, ps_ids, _embedding_dim, _max_key_num,
                       _ps_num=1, partition_num=65537):
    """
    init_partition_map
    """
    op = _get_cache_prim(InitPartitionMap)(_embedding_dim, _max_key_num, _ps_num, partition_num)
    return op(ps_num, ps_ids)


class EmbeddingApplyAdam(Primitive):
    """
    EmbeddingApplyAdam
    """
    @prim_attr_register
    def __init__(self, embedding_dim, _max_key_num, mask_zero=(0,),
                 padding_key=(0,), padding_key_mask=(1,),
                 completion_key=(0,), completion_key_mask=(1,)):
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingApplyAdamW(Primitive):
    """
    EmbeddingApplyAdam
    """
    @prim_attr_register
    def __init__(self, embedding_dim, _max_key_num, amsgrad=(0,),
                 maximize=(0,), mask_zero=(0,), padding_key=(0,),
                 padding_key_mask=(1,), completion_key=(0,), completion_key_mask=(1,)):
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingApplyAdaGrad(Primitive):
    """
    EmbeddingApplyAdaGrad
    """
    @prim_attr_register
    def __init__(self, embedding_dim, _max_key_num, mask_zero=(0,),
                 padding_key=(0,), padding_key_mask=(1,),
                 completion_key=(0,), completion_key_mask=(1,)):
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingApplyFtrl(Primitive):
    """
    EmbeddingApplyFtrl
    """
    @prim_attr_register
    def __init__(self, embedding_dim, _max_key_num, mask_zero=(0,),
                 padding_key=(0,), padding_key_mask=(1,),
                 completion_key=(0,), completion_key_mask=(1,)):
        self.add_prim_attr("_process_node_engine_id", "PS")


class EmbeddingTableFind(Primitive):
    """
    EmbeddingTableFind
    """
    @prim_attr_register
    def __init__(self, embedding_dim, _embedding_dim, _max_key_num,
                 _table_id, default_value=(-1.), _use_counter_filter=0):
        self.add_prim_attr("_process_node_engine_id", "PS")
        self.add_prim_attr("_execute_times", 2)


def embedding_table_find(table_id, keys, embedding_dim, _max_key_num,
                         _table_id, default_value=(-1.0,), _use_counter_filter=0):
    r"""
    embedding_table_find
    """
    _embedding_dim = embedding_dim if isinstance(embedding_dim, int) else embedding_dim[_table_id]
    op = _get_cache_prim(EmbeddingTableFind)(to_sequence(embedding_dim), _embedding_dim,
                                             _max_key_num, _table_id,
                                             to_sequence(default_value),
                                             _use_counter_filter)
    return op(table_id, keys)


class EmbeddingTableFindAndInit(Primitive):
    """
    EmbeddingTableFindAndInit
    """
    @prim_attr_register
    def __init__(self, embedding_dim, value_total_len, _embedding_dim, _table_id,
                 _max_key_num, initializer_mode=("random_uniform",),
                 constant_value=(0.,), min=(-2.,), max=(2.,), mu=(0.,),
                 sigma=(1.,), seed=(0,), seed2=(0,),
                 filter_mode=("no_filter",), filter_freq=(0,),
                 default_key_or_value=(0,), default_key=(0,),
                 default_value=(0.,), completion_key=(0,),
                 completion_key_mask=(1,), optimizer_mode=(),
                 optimizer_params=(), _use_counter_filter=0,
                 backward_mode="adam",
                 backward_int_params=((0,), (0,), (0,), (1,)),
                 backward_float_params=(0.9, 0.99, 0.001, 0.9, 0.999, 1e-08)):
        self.add_prim_attr("_process_node_engine_id", "PS")
        self.add_prim_attr("_execute_times", 2)


def embedding_table_find_and_init(table_id, keys, max_grad_norm, parameter, embedding_dim,
                                  value_total_len, _table_id, _max_key_num,
                                  initializer_mode=('random_uniform',), constant_value=(0.,),
                                  min=(-2.,), max=(2.,), mu=(0.,), sigma=(1.,), seed=(0,),
                                  seed2=(0,), filter_mode=("no_filter",),
                                  filter_freq=(0,), default_key_or_value=(0,),
                                  default_key=(0,), default_value=(0.,),
                                  completion_key=(0,), completion_key_mask=(1,),
                                  optimizer_mode=(), optimizer_params=(), _use_counter_filter=0,
                                  backward_mode="adam", backward_int_params=((0,), (0,), (0,), (1,)),
                                  backward_float_params=(0.9, 0.99, 0.001, 0.9, 0.999, 1e-08)):
    """
    embedding_table_find_and_init

    backward_int_params (Union[tuple[tuple[int]], list[list[int]]]):
        - when the backward_mode is 'adam', 'ftrl' or 'adagrad',
          it means [[global_step], mask_zero, padding_key, padding_key_mask]
        - when the backward_mode is 'adamw', it means:
          [[global_step], amsgrad, maximize, mask_zero, padding_key, padding_key_mask]
    backward_float_params (Union[tuple[float], list[float]]):
        - when the backward_mode is 'adam', it means:
          [beta1_power, beta2_power, lr, beta1, beta2, epsilon]
        - when the backward_mode is 'ftrl', it means:
          [lr, lr_power, lambda1, lambda2]
        - when the backward_mode is 'adamw', it means:
          [beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon]
        - when the backward_mode is 'adagrad', it means [lr,]
    """
    _embedding_dim = embedding_dim if isinstance(embedding_dim, int) else embedding_dim[_table_id]
    op = _get_cache_prim(EmbeddingTableFindAndInit)(to_sequence(embedding_dim), to_sequence(value_total_len),
                                                    _embedding_dim, _table_id, _max_key_num,
                                                    to_sequence(initializer_mode),
                                                    to_sequence(constant_value), to_sequence(min),
                                                    to_sequence(max), to_sequence(mu),
                                                    to_sequence(sigma), to_sequence(seed),
                                                    to_sequence(seed2), to_sequence(filter_mode),
                                                    to_sequence(filter_freq), to_sequence(default_key_or_value),
                                                    to_sequence(default_key), to_sequence(default_value),
                                                    to_sequence(completion_key), to_sequence(completion_key_mask),
                                                    to_sequence(optimizer_mode), to_sequence(optimizer_params),
                                                    _use_counter_filter,
                                                    backward_mode, backward_int_params, backward_float_params)
    return op(table_id, keys, max_grad_norm, parameter)


class FakeRemoteLookupUniqued(Primitive):

    """
    FakeRemoteLookupUniqued
    """
    @prim_attr_register
    def __init__(self, embedding_dim, value_total_len, _embedding_dim, _table_id,
                 _max_key_num, initializer_mode=('random_uniform',), constant_value=(0.,),
                 min=(-2.,), max=(2.,), mu=(0.,), sigma=(1.,), seed=(0,), seed2=(0,),
                 filter_mode=("no_filter",), filter_freq=(0,),
                 default_key_or_value=(0,), default_key=(0,), default_value=(0.,),
                 completion_key=(0,), completion_key_mask=(1,),
                 optimizer_mode=(), optimizer_params=(), _use_counter_filter=0,
                 backward_mode="adam", backward_int_params=((0,), (0,), (0,), (1,)),
                 backward_float_params=(0.9, 0.99, 0.001, 0.9, 0.999, 1e-08)):
        self.add_prim_attr("_process_node_engine_id", "PS")
        self.add_prim_attr("_execute_times", 2)


def fake_remote_lookup_uniqued(table_id, keys, actual_keys_num, unique_indices,
                               key_count, max_grad_norm, parameter,
                               embedding_dim, value_total_len, _table_id, _max_key_num,
                               initializer_mode=('random_uniform',), constant_value=(0.,),
                               min=(-2.,), max=(2.,), mu=(0.,), sigma=(1.,), seed=(0,),
                               seed2=(0,), filter_mode=("no_filter",),
                               filter_freq=(0,), default_key_or_value=(0,),
                               default_key=(0,), default_value=(0.,),
                               completion_key=(0,), completion_key_mask=(1,),
                               optimizer_mode=(), optimizer_params=(), _use_counter_filter=0,
                               backward_mode='adam', backward_int_params=((0,), (0,), (0,), (1,)),
                               backward_float_params=(0.9, 0.99, 0.001, 0.9, 0.999, 1e-08)):
    """
    fake_remote_lookup_uniqued

    backward_mode (str): determine the optimizer used by backpropagation,
        valid values are ["adam", "adamw", "adagrad", "ftrl"]
    backward_int_params (Union[tuple[tuple[int]], list[list[int]]]):
        - when the backward_mode is 'adam', 'ftrl' or 'adagrad',
          it means [[global_step], mask_zero, padding_key, padding_key_mask]
        - when the backward_mode is 'adamw', it means:
          [[global_step], amsgrad, maximize, mask_zero, padding_key, padding_key_mask]
    backward_float_params (Union[tuple[float], list[float]]):
        - when the backward_mode is 'adam', it means:
          [beta1_power, beta2_power, lr, beta1, beta2, epsilon]
        - when the backward_mode is 'ftrl', it means:
          [lr, lr_power, lambda1, lambda2]
        - when the backward_mode is 'adamw', it means:
          [beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon]
        - when the backward_mode is 'adagrad', it means [lr,]
    """
    _embedding_dim = embedding_dim if isinstance(embedding_dim, int) else embedding_dim[_table_id]
    op = _get_cache_prim(FakeRemoteLookupUniqued)(to_sequence(embedding_dim), to_sequence(value_total_len),
                                                  _embedding_dim, _table_id, _max_key_num,
                                                  to_sequence(initializer_mode), to_sequence(constant_value),
                                                  to_sequence(min), to_sequence(max), to_sequence(mu),
                                                  to_sequence(sigma), to_sequence(seed), to_sequence(seed2),
                                                  to_sequence(filter_mode), to_sequence(filter_freq),
                                                  to_sequence(default_key_or_value), to_sequence(default_key),
                                                  to_sequence(default_value), to_sequence(completion_key),
                                                  to_sequence(completion_key_mask), to_sequence(optimizer_mode),
                                                  to_sequence(optimizer_params), _use_counter_filter,
                                                  backward_mode, backward_int_params, backward_float_params)
    return op(table_id, keys, actual_keys_num, unique_indices, key_count, max_grad_norm, parameter)


# Following is Python Infer Value.
# A valid infer value function should be:
#
# 1. named as infer_value_for_OpName
# 2. All inputs should pass without default value.
# 3. If not const input is given, return None. (for now)


def infer_value_for_Tile(input, dims):
    """Infer value for Tile op."""
    if input is None or dims is None or None in dims:
        return None
    return Tensor(np.tile(input.asnumpy(), dims))


def infer_value_for_Concat(tensors, axis):
    """Infer value for Concat op."""
    if not tensors or None in tensors or axis is None:
        return None

    tensor_to_concat = [x.asnumpy() if x.dtype != mstype.bfloat16 else x.float().asnumpy() for x in tensors]
    return Tensor(np.concatenate(tensor_to_concat, axis), dtype=tensors[0].dtype)


def infer_value_for_ReduceSum(input_x, axis, keep_dims, skip_mode):
    """Infer value for ReduceSum op."""
    value = None
    if input_x is not None and axis is not None:
        value = input_x.asnumpy()
        if isinstance(axis, int):
            pass
        elif axis:
            axis = tuple(set(axis))
        elif axis in ((), []) and skip_mode:
            return input_x
        else:
            axis = tuple(range(len(value.shape)))
        value = np.sum(value, axis, keepdims=keep_dims)
        value = np.array(value)
        value = Tensor(value)
    return value


def _infer_value_for_Reduce(input_x, axis, keep_dims, prim_name):
    """Infer value for Common Reduce op."""
    value = None
    if input_x is not None and axis is not None:
        prim_map = {
            'ReduceMax': np.max,
            'ReduceMin': np.min,
            'ReduceProd': np.prod,
            'ReduceMean': np.mean,
            'ReduceAll': np.all,
            'ReduceAny': np.any,
        }
        np_reduce_func = prim_map.get(prim_name, None)

        if np_reduce_func is not None:
            value = input_x.asnumpy()
            if isinstance(axis, int):
                pass
            elif axis:
                axis = tuple(set(axis))
            else:
                axis = tuple(range(len(value.shape)))
            value = np_reduce_func(value, axis, keepdims=keep_dims)
            value = np.array(value)
            value = Tensor(value)
    return value


def _infer_value_for_ReduceExtand(input_x, axis, keep_dims, dtype, prim_name):
    """Infer value for Common ReduceExtand op."""
    value = None
    if input_x is not None:
        prim_map = {
            'MeanExt': np.mean,
            'SumExt': np.sum,
            'ProdExt': np.prod,
        }
        np_reduce_extand_func = prim_map.get(prim_name, None)

        if np_reduce_extand_func is not None:
            value = input_x.asnumpy()
            if isinstance(axis, int):
                pass
            elif axis:
                axis = tuple(set(axis))
            else:
                axis = tuple(range(len(value.shape)))
            if dtype is not None:
                np_dtype = mstype.dtype_to_nptype(typing.type_id_to_type(dtype))
                value = np_reduce_extand_func(value, axis, dtype=np_dtype, keepdims=keep_dims)
            else:
                value = np_reduce_extand_func(value, axis, keepdims=keep_dims)

            value = np.array(value)
            value = Tensor(value)
    return value


def _infer_value_for_max_min(input_x, prim_name):
    """Infer value for Max/Min op."""
    value = None
    if input_x is not None:
        prim_map = {
            'Max': np.max,
            'Min': np.min,
        }
        np_reduce_func = prim_map.get(prim_name, None)

        if np_reduce_func is not None:
            value = input_x.asnumpy()
            value = np_reduce_func(value, None, keepdims=False)
            value = np.array(value)
            value = Tensor(value)
    return value


def infer_value_for_Cast(x, dst_type_enum=None):
    """Infer value for Cast op."""
    if x is None or dst_type_enum is None:
        return None
    dst_type = typing.type_id_to_type(dst_type_enum)
    src_type = mstype.get_py_obj_dtype(x)
    validator.check_subclass("input_x", src_type, [mstype.tensor_type, mstype.number], "Cast")
    validator.check_subclass("type", dst_type, mstype.number, "Cast")

    if isinstance(src_type, type(mstype.tensor_type)):
        src_type = src_type.element_type()
    if isinstance(dst_type, type(mstype.tensor_type)):
        dst_type = dst_type.element_type()

    value = None
    np_dst_type = mstype.dtype_to_nptype(dst_type)
    if isinstance(x, (int, float)):
        value = Tensor(np.array(x).astype(np_dst_type), dtype=dst_type)
    else:
        value = Tensor_(x.asnumpy().astype(np_dst_type), dtype=dst_type)
    return value


def infer_value_for_ReduceMax(input_x, axis, keep_dims):
    """Infer value for ReduceMax op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceMax')


def infer_value_for_Max(input_x):
    """Infer value for Max op."""
    return _infer_value_for_max_min(input_x, 'Max')


def infer_value_for_ReduceMin(input_x, axis, keep_dims):
    """Infer value for ReduceMin op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceMin')


def infer_value_for_Min(input_x):
    """Infer value for Max op."""
    return _infer_value_for_max_min(input_x, 'Min')


def infer_value_for_ReduceProd(input_x, axis, keep_dims):
    """Infer value for ReduceProd op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceProd')


def infer_value_for_ReduceMean(input_x, axis, keep_dims):
    """Infer value for ReduceMean op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceMean')


def infer_value_for_ReduceAll(input_x, axis, keep_dims):
    """Infer value for ReduceAll op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceAll')


def infer_value_for_ReduceAny(input_x, axis, keep_dims):
    """Infer value for ReduceAny op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceAny')


def infer_value_for_MeanExt(input_x, axis, keep_dims, dtype):
    """Infer value for MeanExt op."""
    return _infer_value_for_ReduceExtand(input_x, axis, keep_dims, dtype, 'MeanExt')


def infer_value_for_SumExt(input_x, axis, keep_dims, dtype):
    """Infer value for SumExt op."""
    return _infer_value_for_ReduceExtand(input_x, axis, keep_dims, dtype, 'SumExt')


def infer_value_for_ProdExt(input_x, axis, keep_dims, dtype):
    """Infer value for ProdExt op."""
    return _infer_value_for_ReduceExtand(input_x, axis, keep_dims, dtype, 'ProdExt')


def infer_value_for_Diag(input_x):
    """Infer value for Diag op."""
    if input_x is None:
        return None
    # do constant-folding only when x rank is 1
    if len(input_x.shape) != 1:
        return None
    ret = np.diag(input_x.asnumpy())
    return Tensor(ret)


def infer_value_for_BroadcastTo(x, shape):
    """Infer value for BroadcastTo op."""
    def none_in_tuple_or_list(x):
        return isinstance(x, (tuple, list)) and None in x
    if shape is None or none_in_tuple_or_list(shape) or x is None:
        return None

    if isinstance(shape, (Tensor, Tensor_)):
        validator.check_tensor_dtype_valid("shape", mstype.TensorType(shape.dtype),
                                           [mstype.int32, mstype.int64], "BroadcastTo")
        shape = shape.asnumpy().tolist()
    else:
        validator.check_value_type("shape", shape, [tuple], "BroadcastTo")
        shape = list(shape)

    np_data = np.broadcast_to(x.asnumpy(), shape)
    if 0 in shape:
        init_func = Zero()
        init_func.__enable_zero_dim__ = True
        out = Tensor(shape=shape, dtype=x.dtype, init=init_func)
        return out
    return Tensor(np_data)


def infer_value_for_Reshape(x, shape):
    """Infer value for Reshape op."""
    def none_in_tuple_or_list(x):
        return isinstance(x, (tuple, list)) and None in x
    # for shape is not constant
    if shape is None or none_in_tuple_or_list(shape) or x is None:
        return None

    if isinstance(shape, (Tensor, Tensor_)):
        validator.check_tensor_dtype_valid("shape", mstype.TensorType(shape.dtype),
                                           [mstype.int32, mstype.int64], "Reshape")
        shape = shape.asnumpy().tolist()
    else:
        validator.check_value_type("shape", shape, [tuple], "Reshape")
        shape = list(shape)

    neg_index = -1
    dim_prod = 1
    for i, shp_i in enumerate(shape):
        validator.check_value_type("shape[%d]" % i, shp_i, [int], "Reshape")
        if shp_i == -1:
            if neg_index != -1:
                raise ValueError(f"For 'Reshape', there can be at most one '-1' in 'input_shape', "
                                 f"but got {shape}.")
            neg_index = i
        else:
            dim_prod *= shp_i
    out = None
    if not is_shape_unknown(x.shape):
        x_shp = x.shape
        if dim_prod < 0:
            raise ValueError(f"For 'Reshape', the shape of 'input_x' is {x_shp}, "
                             f"the value of 'input_shape' is {shape}. "
                             f"The product of 'input_shape' should > 0, but got {dim_prod}.")
        arr_prod = np.prod(x_shp)
        if neg_index != -1:
            shape[neg_index] = int(arr_prod // dim_prod)
            dim_prod *= shape[neg_index]
        if dim_prod != arr_prod:
            raise ValueError(f"For 'Reshape', the product of the 'input_x' shape "
                             f"should be equal to product of 'input_shape', but got product of the"
                             f" shape of 'input_x': {arr_prod}, product of 'input_shape': {dim_prod}.")
        if 0 in shape:
            init_func = Zero()
            init_func.__enable_zero_dim__ = True
            out = Tensor(shape=shape, dtype=x.dtype, init=init_func)
        else:
            out = Tensor(x.asnumpy().reshape(shape))
    return out


class Ones(Primitive):
    r"""
    Creates a tensor filled with value ones.

    Refer to :func:`mindspore.ops.ones` for more details.

    .. warning::
        For argument `size`, Tensor type input will be deprecated in the future version.

    Inputs:
        - **shape** (Union[tuple[int], List[int], int, Tensor]) - The specified shape of output tensor.
        - **type** (:class:`mindspore.dtype`) - The specified type of output tensor.

    Outputs:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `shape` is neither an int nor an tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> ones = ops.Ones()
        >>> output = ones((2, 2), mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ones((3, 3), mindspore.float32)
        >>> print(output)
        [[1. 1. 1.]
         [1. 1. 1.]
         [1. 1. 1.]]
    """

    __mindspore_signature__ = (
        sig.make_sig('size'),
        sig.make_sig('type', default=None),
    )

    @prim_arg_register
    def __init__(self):
        pass

    def __call__(self, size, type=None):
        return _convert_stub(pyboost_ones(self, [size, type if type is None \
            else handler.dtype_to_type_id('Ones', 'type', type)]))


class Zeros(Primitive):
    r"""
    Zeros will be deprecated in the future. Please use class `mindspore.ops.zeros` instead.

    Creates a tensor filled with value zeros.

    Creates a tensor with shape described by the first argument and
    fills it with value zeros in type of the second argument.

    .. warning::
        For argument `size`, Tensor type input will be deprecated in the future version.

    Inputs:
        - **shape** (tuple[int], List[int], int, Tensor) - The specified shape of output tensor.
        - **type** (mindspore.dtype) - The specified type of output tensor.

    Outputs:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `shape` is neither an int nor an tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> zeros = ops.Zeros()
        >>> output = zeros((2, 2), mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]

    """

    __mindspore_signature__ = (
        sig.make_sig('size'),
        sig.make_sig('type', default=None),
    )

    @prim_arg_register
    def __init__(self):
        pass

    def __call__(self, size, type=None):
        return _convert_stub(pyboost_zeros(self, [size, type if type is None else \
            handler.dtype_to_type_id('Zeros', 'type', type)]))


def flash_attention_score(query, key, value, head_num, real_shift=None, drop_mask=None, padding_mask=None,
                          attn_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, keep_prob=1.0,
                          scalar_value=1.0, pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0,
                          input_layout='BSH', sparse_mode=0):
    r"""
    The interface is not open to the public, just for internal use,

    .. math::
        \begin{array}{ll} \\
            y = Dropout(Softmax(Mask(scale_value \mul (real_shift + query * key), attn_mask), -1), keep\_prob) \\
            \mul value \\
        \end{array}

    B -- Batch size. Value range 1 to 2k.
    S1 -- Sequence length of query. Value range 1 to 512k.
    S2 -- Sequence length of key and value. Value range 1 to 512k.
    N1 -- Num heads of query. Value range 1 to 256.
    N2 -- Num heads of key and value, and N2 must be a factor of N1.
    D -- Head size. The value ranges is a multiple of 16, with the max value of 512.
    H1 -- Hidden size of query, which equals to N1 * D.
    H2 -- Hidden size of key and value, which equals to N2 * D.

    .. warning::
        This is an experimental API that is subject to change or deletion. Only support on Atlas training series.

    Args:
        query (Tensor[float16, bfloat16]): The query tensor. Input tensor of shape :math:`(B, S1, H1)`,
            `(B, N1, S1, D)`, `(S1, B, H1)`, `(B, S1, N1, D)` or `(T1, N1, D)`.
        key (Tensor[float16, bfloat16]): The key tensor. Input tensor of shape :math:`(B, S2, H2)`,
            `(B, N2, S2, D)`, `(S2, B, H2)`, `(B, S2, N2, D)` or `(T2, N2, D)`.
        value (Tensor[float16, bfloat16]): The value tensor. Input tensor of shape :math:`(B, S2, H2)`,
            `(B, N2, S2, D)`, `(S2, B, H2)`, `(B, S2, N2, D)` or `(T2, N2, D)`. The key and value have the same shape.
        head_num (int): The head num of query, equal to N1.
        real_shift (Union[Tensor[float16, bfloat16], None]): Also known as pse. The position embedding code. If S
            is greater than 1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of
            the lower triangle for memory optimization. Input tensor of shape :math:`(B, N1, S1, S2)`,
            `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`.

            - ALiBi scenario: real_shift must meet the ALiBi rule, and sparse_mode is 2 or 3 for the lower triangle.
              In this scenario, real_shift is `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`.
            - Non-ALiBi scenario: real_shift is `(B, N1, S1, S2)`, `(1, N1, S1, S2)`.

            The shape of `real_shift` should be `(B, N1, 1024, S2)` and `(1, N1, 1024, S2)` when input_layout is
            `TND`.
        drop_mask (Union[Tensor[uint8], None]): The dropout mask tensor. Input tensor of shape :math:
            `(B, N1, S1, S2 // 8) or None`. S2 is a multiple of 8 when not None.
        padding_mask (None): Reserved parameter. Not implemented yet.
        attn_mask (Union[Tensor[uint8], Tensor[bool], None]): The attention mask tensor. For each element, 0
            indicates retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`,
            `(B, 1, S1, S2)`, `(S1, S2)` or `(2048, 2048)`. In compression scenario, sparse_mode is 2, 3, or 4,
            attn_mask must be `(2048, 2048)`. When sparse_mode is 5, attn_mask must be `(B, N1, S1, S2)`,
            `(B, 1, S1, S2)`. When sparse_mode is 0 and 1, attn_mask should be `(B, N1, S1, S2)`, `(B, 1, S1, S2)`,
            `(S1, S2)`.
        prefix (Union[List[int64], Tuple[int64] None]): N value of each Batch in the prefix sparse calculation
            scenario. Input tensor of shape :math:`(B,)`. B max value 32. Not none only when sparse_mode is 5.
            If S1 > S2, N ranges from 0 to S2. If S1 <= S2, N ranges from S2 - S1 to S2.
        actual_seq_qlen (Union[List[int64], Tuple[int64], None]): Size of query corresponding to each batch, array
            with increasing values and the last value equal to T1.
        actual_seq_kvlen (Union[List[int64], Tuple[int64], None]): Size of key and value corresponding to each batch,
            array with increasing values and the last value equal to T2.
        keep_prob (float): The keep probability of dropout. Value range is (0.0, 1.0]. Default: 1.0. when keep_prob
            is 1.0, drop_mask should be none.
        scale_value (float): The scale factor of score. Generally, the value is 1.0 / (D ** 0.5). Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
            When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
            When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
            The value of pre_tokens corresponds to S1, and the value of next_tokens corresponds to S2. They define the
            valid area on the attn_mask matrix. It must ensure that the band is not empty.
            The following values are not allowed:

            - pre_tokens < 0 and next_tokens < 0.
            - (pre_tokens < 0 and next_tokens >= 0) and (next_tokens < abs(pre_tokens) or abs(pre_tokens) >= S2).
            - (pre_tokens >= 0 and next_tokens < 0) and (abs(next_tokens) > pre_tokens or abs(next_tokens) >= S1).

        inner_precise (int): The parameter is reserved and not implemented yet. Default: 0.
        input_layout (str): Specifies the layout of input `query`, key and value. The value can be "BSH", "BNSD",
            "SBH", "BSND" or "TND". "TND" is an experimental format. Default: "BSH".
            When input_layout is "TND", the following restrictions must be met.
            There are two lists that represent the length of the input sequence: list_seq_q and list_seq_k. Each
            value in the list indicates the length of the sequence in the batch. For example, list_seq_q = [4, 2, 6],
            list_seq_k = [10, 3, 9]. The element of list indicate S. T1 is sum(list_seq_q) = 12, T2 is
            sum(list_seq_k) = 22.
            max_seqlen_q = max(list_seq_q), max_seqlen_k = max(list_seq_k).
            qk_pointer = sum(list_seq_q * list_seq_k), which is the sum of the element multiplication.

            - The lengths of two lists are the same, and size of list is batch. batch is less than or equal to 1024.
            - When input_layout is "TND", actual_seq_qlen and actual_seq_kvlen must be not none.
              Otherwise, they are none.
            - The actual_seq_qlen and actual_seq_kvlen are the cumulative sum of sequence of key/value, so they must
              be non-decreasing.
            - If real_shift is not none, list_seq_q and list_seq_k must be same. The maximum value of list_seq_q and
              list_seq_k is greater than 1024. Real_shift should be `(B, N1, 1024, S2)` and `(1, N1, 1024, S2)`, and
              S2 is equal to max_seqlen_k.
            - Attn mask must be a lower trianglar matrix, so sparse_mode should be 2 or 3. The shape of attn_mask
              should be `(2048, 2048)`.
            - The shape of drop_mask is (qk_pointer * N1 // 8,).
            - Prefix is none.
            - Next_tokens is 0, and pre_tokens is not less than max_seqlen_q.
            - When sparse_mode is 3, S1 of each batch should be less than or equal to S2.
            - 0 should not exist in list_seq_k.

        sparse_mode (int): Indicates sparse mode. Default 0.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full
              attn_mask matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and
              nextTokens needs to be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required.
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value
              of each Batch axis is different, not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.

    Returns:
        attention_out (Tensor[float16, bfloat16]), The output of attention, its shape, and data type are the same
        as the query.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import mindspore.common.dtype as mstype
        >>> import numpy as np
        >>> from mindspore import ops, Tensor
        >>> query = Tensor(np.ones([2, 4, 64]), dtype=mstype.float16)
        >>> key = Tensor(np.ones([2, 4, 64]), dtype=mstype.float16)
        >>> value = Tensor(np.ones([2, 4, 64]), dtype=mstype.float16)
        >>> head_num = 4
        >>> output = ops.flash_attention_score(query, key, value, head_num)
        >>> print(output.shape)
        (2, 4, 64)
    """
    rank_op = _get_cache_prim(FlashAttentionScore)(head_num, keep_prob, scalar_value, pre_tokens, next_tokens,
                                                   inner_precise, input_layout, sparse_mode)
    return rank_op(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen,
                   actual_seq_kvlen)[3]
