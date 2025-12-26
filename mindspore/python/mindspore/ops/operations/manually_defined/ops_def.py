# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
import types
import numpy as np
from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register, prim_arg_register, PrimitiveWithInfer
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops._utils import arg_handler as handler
from mindspore.ops._utils.arg_dtype_cast import DtypeToEnum
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore._c_expression import typing
from mindspore._c_expression import TensorPy as Tensor_
from mindspore._c_expression import pyboost_cast, pyboost_tile, pyboost_type_as
from mindspore.common import dtype as mstype
from mindspore.common._utils import is_shape_unknown
from mindspore import _checkparam as validator
from mindspore.ops.operations.manually_defined._inner import ScalarCast
from mindspore.common.initializer import Zero
from mindspore.common.parameter import Parameter
from mindspore.ops.auto_generate.gen_ops_prim import FlashAttentionScore, FusedInferAttentionScore
from mindspore.common.jit_context import jit_context


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


class ScalarMax(Primitive):
    r"""
    Return the maximum of two input scalars.

    .. note::
        The inputs can be constant/variable value. Usage is the same as 'max' in Python.
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
        """Initialize ScalarMax"""

    def __call__(self, x, y):
        return max(x, y)


class ScalarMin(Primitive):
    r"""
    Return the minimum of two input scalars.

    .. note::
        The inputs can be constant/variable value. Usage is the same as 'min' in Python.
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
        """Initialize ScalarMin"""

    def __call__(self, x, y):
        return min(x, y)


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
scalar_max = ScalarMax()
scalar_min = ScalarMin()


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
        is_training (bool, optional): If `is_training` is ``True`` ,
            `mean` and `variance` are computed during training.
            If `is_training` is ``False`` , they're loaded from checkpoint during inference. Default: ``False`` .
        epsilon (float, optional): A small value added for numerical stability.
            Default: ``1e-5``, value must be (0, 1] .
        momentum (float, optional): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1]. Default: ``0.1`` .
        data_format (str, optional): The optional value for data format, is ``'NHWC'`` or ``'NCHW'``,
            and the ``'NHWC'`` format
            is only supported in GPU target. Default: ``"NCHW"`` .

    Inputs:
        - **input_x** (Tensor) - Tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        - **scale** (Union[Parameter, Tensor]) - Tensor or Parameter of shape :math:`(C,)`,
          with float16 or float32 data type.
        - **bias** (Union[Parameter, Tensor]) - Tensor or Parameter of shape :math:`(C,)`,
          has the same data type with `scale`.
        - **mean** (Union[Parameter, Tensor]) - Tensor or Parameter of shape :math:`(C,)`,
          has the same data type with `scale`.
        - **variance** (Union[Parameter, Tensor]) - Tensor or Parameter of shape :math:`(C,)`,
          has the same data type with `scale`.

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
    Return the rank of a tensor.

    Args:
        input_x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_tensor = mindspore.tensor([[2, 2], [2, 2]], mindspore.float32)
        >>> output = mindspore.ops.rank(input_tensor)
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
        data_type = mstype._dtype_to_nptype(dtype)  # pylint:disable=protected-access
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

        .. note::
            On Ascend, the number of `dims` should not exceed 8, and currently does not support scenarios
            where more than 4 dimensions are repeated simultaneously.

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
        # Add for jit context.
        if jit_context() and jit_context().compiled:
            return jit_context().default_output()
        res = pyboost_tile(self, [input, dims])
        # Add for jit context.
        if jit_context():
            if validator.is_stub_tensor(res):
                res = res.stub_sync()
            return jit_context().run_op(self, res, input, dims)
        return res

    def check_elim(self, *args):
        """check elim"""
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
    Creates a new tensor by repeating the elements in the input tensor `dims` times.

    The i'th dimension of output tensor has `input.shape[i] * dims[i]` elements, and the values of `input`
    are repeated `dims[i]` times along the i'th dimension.

    Note:
        - On Ascend, the number of `dims` should not exceed 8, and currently does not support scenarios
          where more than 4 dimensions are repeated simultaneously.
        - If `input.dim = d`, then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)`.
        - If `input.dim < d`, prepend 1 to the shape of `input` until their lengths are consistent.
          Such as set the shape of `input` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and the shape of Outputs is
          :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)`.
        - If `input.dim > d`, prepend 1 to `dims` until their lengths are consistent. Such as set the
          `dims` as :math:`(1, ..., y_1, y_2, ..., y_S)`, then the shape of their corresponding positions
          can be multiplied, and the shape of Outputs is :math:`(x_1*1, ..., x_R*y_R, x_S*y_S)`.

    Args:
        input (Tensor): The input tensor.
        dims (tuple[int]): The specified number of repetitions in each dimension.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2], [3, 4]])
        >>> mindspore.ops.tile(input, (2, 3))
        Tensor(shape=[4, 6], dtype=Int64, value=
        [[1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4],
         [1, 2, 1, 2, 1, 2],
         [3, 4, 3, 4, 3, 4]])
        >>> mindspore.ops.tile(input, (2, 3, 2))
        Tensor(shape=[2, 6, 4], dtype=Int64, value=
        [[[1, 2, 1, 2],
          [3, 4, 3, 4],
          [1, 2, 1, 2],
          [3, 4, 3, 4],
          [1, 2, 1, 2],
          [3, 4, 3, 4]],
         [[1, 2, 1, 2],
          [3, 4, 3, 4],
          [1, 2, 1, 2],
          [3, 4, 3, 4],
          [1, 2, 1, 2],
          [3, 4, 3, 4]]])
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
            The value should only be mindspore.int64, mindspore.float64, or mindspore.bool.

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
        - **input** (Union[Tensor, Number]) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The tensor to be cast.
        - **dtype** (dtype.Number) - The valid data type of the output tensor. Only constant value is allowed.

    Outputs:
        Tensor, the shape of tensor is the same as `input`, :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input` is neither Tensor nor Number.
        TypeError: If `dtype` is not a Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input = Tensor(input_np)
        >>> dtype = mindspore.int32
        >>> cast = ops.Cast()
        >>> output = cast(input, dtype)
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
        if isinstance(x, Parameter):
            data = x.data
            if data.dtype == dtype:
                return (True, x)
        if isinstance(x, Tensor) and x.dtype == dtype:
            return (True, x)
        if isinstance(x, numbers.Number):
            return (True, Tensor(x, dtype=dtype))
        return (False, None)

    def __call__(self, input_x, dtype):
        # Add for jit context.
        if jit_context() and jit_context().compiled:
            return jit_context().default_output()
        should_elim, output = self.check_elim(input_x, dtype)
        if should_elim:
            return output
        res = pyboost_cast(self, [input_x, dtype_to_type_id('Cast', 'dtype', dtype)])
        # Add for jit context.
        if jit_context():
            if validator.is_stub_tensor(res):
                res = res.stub_sync()
            return jit_context().run_op(self, res, input_x, dtype)
        return res


class TypeAs(Primitive):
    """
    Returns first input tensor cast to the type of the with the second input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Note:
        When converting complex numbers to boolean type, the imaginary part of the complex number is not
        taken into account. As long as the real part is non-zero, it returns True; otherwise, it returns False.

    Inputs:
        - **input** (Tensor) -  The shape of tensor is :math:`(x_0, x_1, ..., x_R)`.
          The tensor whose data type is to be converted.
        - **other ** (Tensor) - The shape of tensor is :math:`(x_0, x_1, ..., x_R)`.
          The tensor whose data type is specified.

    Outputs:
        Tensor, the shape of tensor is the same as `input`, :math:`(x_0, x_1, ..., x_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `other` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
        >>> input = Tensor(input_np)
        >>> other_np = np.random.randn(2, 3, 4).astype(np.int32)
        >>> other = Tensor(other_np)
        >>> type_as = ops.TypeAs()
        >>> output = type_as(input, other)
        >>> print(output.dtype)
        Int32
        >>> print(output.shape)
        (2, 3, 4, 5)
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, input, other):
        if input.dtype == other.dtype:
            return input
        return pyboost_type_as(self, [input, other])


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
    if input.dtype == mstype.bfloat16:
        a = input.astype(mstype.float32).asnumpy()
        b = np.tile(a, dims)
        c = Tensor(b)
        return c.astype(mstype.bfloat16)
    a = input.asnumpy()
    b = np.tile(a, dims)
    c = Tensor(b)
    return c


def infer_value_for_EqualExt(x, y):
    """Infer value for EqualExt op."""
    if x is None or y is None:
        return None
    result = np.equal(x.asnumpy(), y.asnumpy())
    value = False
    if result.all():
        value = True
    return Tensor(value)


def infer_value_for_Concat(tensors, axis):
    """Infer value for Concat op."""
    if not tensors or None in tensors or axis is None:
        return None

    tensor_to_concat = [x.asnumpy() for x in tensors]
    out = np.concatenate(tensor_to_concat, axis)
    if out.dtype != np.float32:
        return Tensor(out)
    for x in tensors:
        if x.dtype in [mstype.float16, mstype.float32]:
            return Tensor(out)
    return Tensor(out, dtype=mstype.bfloat16)


def infer_value_for_GatherD(input, dim, index):
    """Infer value for GatherD op."""
    if input is None or dim is None or index is None:
        return None

    input_np = input.asnumpy()
    index_np = index.asnumpy()

    index_shape = index_np.shape
    multi_index = [np.indices(index_shape)[i] for i in range(len(index_shape))]
    multi_index[dim] = index_np

    output = input_np[tuple(multi_index)]
    return Tensor(output, dtype=input.dtype)


def infer_value_for_Softmax(input, axis):
    """Infer value for Softmax op."""
    if input is None or axis is None:
        return None

    e_input = np.exp(input.asnumpy())
    output = e_input / np.sum(e_input, axis=axis, keepdims=True)
    return Tensor(output, dtype=input.dtype)


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


def infer_value_for_Arange(start, end, step, dtype=None):
    """Infer value for Arange op."""
    if start is None or end is None or step is None:
        return None
    np_dtype = np.int64
    if dtype is None:
        has_float = any(isinstance(i, float) for i in [start, end, step])
        if has_float:
            np_dtype = np.float32
    else:
        np_dtype = mstype._dtype_to_nptype(typing.type_id_to_type(dtype))  # pylint:disable=protected-access
    return Tensor(np.arange(start, end, step, dtype=np_dtype))


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
                np_dtype = mstype._dtype_to_nptype(typing.type_id_to_type(dtype))  # pylint:disable=protected-access
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
    src_type = mstype._get_py_obj_dtype(x)  # pylint:disable=protected-access
    validator.check_subclass("input_x", src_type, [mstype.tensor_type, mstype.number], "Cast")
    validator.check_subclass("type", dst_type, mstype.number, "Cast")

    if isinstance(src_type, type(mstype.tensor_type)):
        src_type = src_type.element_type()
    if isinstance(dst_type, type(mstype.tensor_type)):
        dst_type = dst_type.element_type()

    value = None
    np_dst_type = mstype._dtype_to_nptype(dst_type)  # pylint:disable=protected-access
    if isinstance(x, (int, float)):
        value = Tensor(np.array(x).astype(np_dst_type), dtype=dst_type)
    else:
        value = Tensor_(x.asnumpy().astype(np_dst_type), dtype=dst_type)
    return value


def infer_value_for_LinalgVectorNorm(input_x, ord, dim, keepdim, dtype):
    """Infer value for linalg_vector_norm op.
       Current version numpy is not support numpy.linalg.vector_norm.
       So using numpy.linalg.norm.
    """
    if input_x is None or ord is None:
        return None
    if ord != 0:
        out = np.power(np.sum(np.power(np.abs(input_x.asnumpy()), ord), axis=dim, keepdims=keepdim), 1/ord)
    else:
        out = np.sum(input_x.asnumpy() != 0, axis=dim, keepdims=keepdim)
    if dtype is None:
        return Tensor(out)
    dtype_for_ms = typing.type_id_to_type(dtype)
    return Tensor(out, dtype=dtype_for_ms)


def infer_value_for_LpNormV2(input_x, p=2, dim=None, keepdim=False, eps=1e-12):
    """Infer value for linalg_vector_norm op.
       Current version numpy is not support numpy.linalg.vector_norm.
       So using numpy.linalg.norm.
    """
    if input_x is None:
        return None
    return Tensor(np.linalg.norm(input_x.asnumpy(), axis=dim, keepdims=keepdim,
                                 ord=p))


def infer_value_for_Svd(input_x, full_matrices, compute_uv):
    """Infer value for Svd op."""
    if input_x is None:
        return None
    if bool(compute_uv):
        s, u, v = np.linalg.svd(input_x.asnumpy(), full_matrices=full_matrices, compute_uv=True)
        return Tensor(s), Tensor(u), Tensor(v)
    s = np.linalg.svd(input_x.asnumpy(), full_matrices=full_matrices, compute_uv=False)
    return Tensor(s), np.zeros(1), np.zeros(1)


def infer_value_for_Div(input_x, other_x):
    """Infer value for Div op."""
    if input_x is None or other_x is None:
        return None
    # NumPy does not support bfloat16 arithmetic well.
    if (input_x.dtype == mstype.bfloat16) or (other_x.dtype == mstype.bfloat16):
        return None
    out = np.true_divide(input_x.asnumpy(), other_x.asnumpy())
    return Tensor(out)


def infer_value_for_Divs(input_x, other_x):
    """Infer value for Divs op."""
    if input_x is None or other_x is None:
        return None
    if input_x.dtype == mstype.bfloat16:
        return None
    tmp = np.true_divide(input_x.asnumpy(), other_x)
    if not input_x.shape:
        # tensor scalar has a special rule for data type promote
        if input_x.dtype in (mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64, mstype.int8, mstype.int16,
                             mstype.int32, mstype.int64):
            res = Tensor(tmp, dtype=mstype.float32)
        else:
            res = Tensor(tmp, dtype=input_x.dtype)
    else:
        res = Tensor(tmp)
    return res


def infer_value_for_DivMod(input_x, other_x, rounding_mode):
    """Infer value for DivMod op."""
    if input_x is None or other_x is None:
        return None
    if input_x.dtype == mstype.bfloat16 or other_x.dtype == mstype.bfloat16:
        return None
    if rounding_mode == 1:
        # trunc
        return Tensor(np.trunc(np.true_divide(input_x.asnumpy(), other_x.asnumpy())))
    if rounding_mode == 2:
        # floor
        return Tensor(np.floor_divide(input_x.asnumpy(), other_x.asnumpy()))
    return None


def infer_value_for_DivMods(input_x, other_x, rounding_mode):
    """Infer value for DivMods op."""
    if input_x is None or other_x is None:
        return None
    if input_x.dtype == mstype.bfloat16:
        return None
    if rounding_mode == 1:
        # trunc
        return Tensor(np.trunc(np.true_divide(input_x.asnumpy(), other_x)))
    if rounding_mode == 2:
        # floor
        return Tensor(np.floor_divide(input_x.asnumpy(), other_x))
    return None


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

    # Resolve -1 entries and support input rank < target rank.
    input_shape = list(x.shape)
    target_shape = list(shape)
    in_rank = len(input_shape)
    out_rank = len(target_shape)
    for k in range(1, out_rank + 1):
        t = target_shape[-k]
        if t == -1:
            if k <= in_rank:
                target_shape[-k] = input_shape[-k]
            else:
                pass

    resolved_shape = target_shape

    np_data = np.broadcast_to(x.asnumpy(), resolved_shape)
    if 0 in resolved_shape:
        init_func = Zero()
        init_func.__enable_zero_dim__ = True
        out = Tensor(shape=resolved_shape, dtype=x.dtype, init=init_func)
        out.init_data()
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
            out.init_data()
        else:
            out = Tensor(x.asnumpy().reshape(shape))
    return out


def flash_attention_score(query, key, value, head_num, real_shift=None, drop_mask=None, padding_mask=None,
                          attn_mask=None, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, keep_prob=1.0,
                          scalar_value=1.0, pre_tokens=2147483647, next_tokens=2147483647, inner_precise=0,
                          input_layout='BSH', sparse_mode=0):
    r"""
    Implement self-attention calculations in training scenarios.

    - B: Batch size. Value range 1 to 2k.
    - S1: Sequence length of `query`. Value range 1 to 512k.
    - S2: Sequence length of `key` and `value`. Value range 1 to 512k.
    - N1: Num heads of `query`. Value range 1 to 256.
    - N2: Num heads of `key` and `value`, and N2 must be a factor of N1.
    - D: Head size. The value ranges is a multiple of 16, with the max value of 512.
    - H1: Hidden size of `query`, which equals to N1 * D.
    - H2: Hidden size of `key` and `value`, which equals to N2 * D.

    The self attention calculation formula is defined as:

    .. math::
        \begin{array}{ll} \\
            \text { attention_out }=\operatorname{Dropout}\left(\operatorname{Softmax}\left(\text
            { Mask(scale } *\left(\text { query } * \mathrm{key}^{\top}\right)+\text { pse }\right)\text
            {, atten_mask), keep_prob) } *\right. \text { value }
        \end{array}

    .. warning::
        - This is an experimental API that is subject to change or deletion.
        - Only support on Atlas A2 training series.

    Args:
        query (Tensor): The query tensor. Input tensor of shape :math:`(B, S1, H1)`,
            :math:`(B, N1, S1, D)`, :math:`(S1, B, H1)`, :math:`(B, S1, N1, D)` or :math:`(T1, N1, D)`.
            The supported dtype is float16 and bfloat16.
        key (Tensor): The key tensor with the same dtype as `query`. Supported shape: :math:`(B, S2, H2)`,
            :math:`(B, N2, S2, D)`, :math:`(S2, B, H2)`, :math:`(B, S2, N2, D)` or :math:`(T2, N2, D)`.
        value (Tensor): The value tensor with the same dtype and shape as `key`.
        head_num (int): The head num of `query`, equal to N1.
        real_shift (Tensor, optional): The position embedding code which is also known as pse, it has the same
            dtype as `query`.
            Default: ``None``.
            If S is greater than 1024 and the mask of the lower triangle is used, only the inverse 1024 lines of
            the lower triangle is used for memory optimization. Input tensor of shape :math:`(B, N1, S1, S2)`,
            :math:`(1, N1, S1, S2)`, :math:`(B, N1, 1024, S2)`, :math:`(1, N1, 1024, S2)`.

            - ALiBi scenario: `real_shift` must meet the ALiBi rule, and sparse_mode is 2 or 3 for the lower triangle.
              In this scenario, `real_shift` is :math:`(B, N1, 1024, S2)`, :math:`(1, N1, 1024, S2)`.
            - Non-ALiBi scenario: `real_shift` is :math:`(B, N1, S1, S2)`, :math:`(1, N1, S1, S2)`.
            - input_layout is TND: shape should be :math:`(B, N1, 1024, S2)` and :math:`(1, N1, 1024, S2)`.

        drop_mask (Tensor, optional): The dropout mask tensor of uint8. Input tensor of shape
            :math:`(B, N1, S1, S2 // 8) or None`. `S2` is a multiple of 8 when not None. Default: ``None``.
        padding_mask (Tensor, optional): Reserved parameter. Not implemented yet. Default: ``None``.
        attn_mask (Tensor, optional): The attention mask tensor of bool or uint8. For each element, 0/False
            indicates retention and 1/True indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`,
            :math:`(B, 1, S1, S2)`, :math:`(S1, S2)` or :math:`(2048, 2048)`.
            Default: ``None``.

            - In compression scenario, `sparse_mode` is 2, 3, or 4, `attn_mask` must be :math:`(2048, 2048)`.
            - When `sparse_mode` is 5, `attn_mask` should be :math:`(B, N1, S1, S2)`, :math:`(B, 1, S1, S2)`.
            - When `sparse_mode` is 0 and 1, `attn_mask` should be :math:`(B, N1, S1, S2)`, :math:`(B, 1, S1, S2)`,
              :math:`(S1, S2)`.

        prefix (Union[Tensor, tuple[int], list[int]], optional): N value of each Batch in the prefix sparse calculation
            scenario. Input tensor of shape :math:`(B,)`. B max value 32. This parameter takes effect only when
            `sparse_mode` is 5. Default: ``None``.
            If S1 > S2, N ranges from 0 to S2. If S1 <= S2, N ranges from S2 - S1 to S2.
        actual_seq_qlen (Union[Tensor, tuple[int], list[int]], optional): Size of query corresponding to each batch,
            array with increasing values and the last value equal to T1.
            Default: ``None``.
        actual_seq_kvlen (Union[Tensor, tuple[int], list[int]], optional): Size of key and value corresponding
            to each batch, array with increasing values and the last value equal to T2.
            Default: ``None``.
        keep_prob (double, optional): The keep probability of dropout. Value range is (0.0, 1.0]. When `keep_prob`
            is 1.0, `drop_mask` should be None.
            Default: ``1.0``.
        scalar_value (double, optional): The scale value indicating the scale coefficient, which is used as the
            scalar of Muls in the calculation. Generally, the value is 1.0 / (D ** 0.5).
            Default: ``1.0``.
        pre_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted forward.
            When `sparse_mode` is set to 1, 2, 3, or 5, this parameter does not take effect.
            Default: ``2147483647``.
        next_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted backward.
            When `sparse_mode` is set to 1, 2, 3, or 5, this parameter does not take effect. Default: ``2147483647``.
            The value of `pre_tokens` corresponds to S1, and the value of `next_tokens` corresponds to S2.
            They define the valid area on the `attn_mask` matrix. It must ensure that the band is not empty.
            The following values are not allowed:

            - pre_tokens < 0 and next_tokens < 0.
            - (pre_tokens < 0 and next_tokens >= 0) and (next_tokens < abs(pre_tokens) or abs(pre_tokens) >= S2).
            - (pre_tokens >= 0 and next_tokens < 0) and (abs(next_tokens) > pre_tokens or abs(next_tokens) >= S1).

        inner_precise (int, optional): The parameter is reserved and not implemented yet. Default:``0``.
        input_layout (str, optional): Specifies the layout of input `query`, `key` and `value`. The value can be
            "BSH", "BNSD", "SBH", "BSND" or "TND". "TND" is an experimental format. Default: ``"BSH"``.
            When input_layout is "TND", the following restrictions must be met.
            Assume there are two lists that represent the length of the input sequence: list_seq_q and list_seq_k. Each
            value in the list indicates the length of the sequence in the batch. For example, list_seq_q = [4, 2, 6],
            list_seq_k = [10, 3, 9]. The element of list indicate S. T1 is sum(list_seq_q) = 12, T2 is
            sum(list_seq_k) = 22.
            max_seqlen_q = max(list_seq_q), max_seqlen_k = max(list_seq_k).
            qk_pointer = sum(list_seq_q * list_seq_k), which is the sum of the element multiplication.

            - The lengths of two lists must be the same, and size of list is batch. batch is less than or equal to
              1024.
            - When `input_layout` is "TND", `actual_seq_qlen` and `actual_seq_kvlen` must be not none.
              Otherwise, they are none.
            - The `actual_seq_qlen` and `actual_seq_kvlen` are the cumulative sum of sequence of key/value, so they must
              be non-decreasing.
            - If `real_shift` is not none, list_seq_q and list_seq_k must be same. The maximum value of list_seq_q and
              list_seq_k is greater than 1024. `real_shift` should be :math:`(B, N1, 1024, S2)` and
              :math:`(1, N1, 1024, S2)`, and S2 is equal to max_seqlen_k.
            - `attn_mask` must be a lower trianglar matrix, so `sparse_mode` should be 2 or 3. The shape of `attn_mask`
              should be :math:`(2048, 2048)`.
            - The shape of `drop_mask` is :math:`(qk\_pointer * N1 // 8,)`.
            - `prefix` is none.
            - `next_tokens` is 0, and `pre_tokens` is not less than max_seqlen_q.
            - When `sparse_mode` is 3, S1 of each batch should be less than or equal to S2.
            - 0 should not exist in list_seq_k.

        sparse_mode (int, optional): Indicates sparse mode. Default: ``0``.

            - 0: Indicates the defaultMask mode. If `attn_mask` is not passed, the mask operation is not performed,
              `next_tokens` and `pre_tokens` (internally assigned as INT_MAX) are ignored. If passed in, the full
              `attn_mask` matrix (S1 * S2) needs to be passed in, indicating that the part between `next_tokens` and
              `pre_tokens` needs to be calculated.
            - 1: Represents allMask, that is, passing in the complete `attn_mask` matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized `attn_mask` matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized `attn_mask` matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting `next_tokens` and `pre_tokens`,
              and the optimized `attn_mask` matrix (2048*2048) is required.
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input `prefix`, and the N value
              of each Batch axis is different. `prefix` takes effect only when `sparse_mode` is 5.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.

    Returns:
        attention_out (Tensor) - The output of attention, it has the same shape and dtype as `query`.

    Raises:
        TypeError: Dtype of `query` is not float16 or bfloat16.
        TypeError: `query`, `key` and `value` don't have the same dtype.
        TypeError: Dtype of `attn_mask` is not bool or uint8.
        TypeError: Dtype of `real_shift` has a different dtype as `query`.
        TypeError: `scalar_value` or `keep_prob` is not a double number.
        TypeError: `input_layout` is not a string.
        TypeError: `num_key_value_heads` is not an int.
        TypeError: `sparse_mode` is not an int.
        TypeError: `real_shift` is not Tensor type.
        TypeError: `drop_mask` is not Tensor type.
        TypeError: `padding_mask` is not Tensor type.
        TypeError: `attn_mask` is not Tensor type.
        ValueError: `input_layout` is a string but not valid.
        RuntimeError: `head_num` is not divisible by `N2`.
        RuntimeError: `head_num` is not greater than 0.
        RuntimeError: `attn_mask` shape is not valid.
        RuntimeError: The specified value of `sparse_mode` is invalid.
        RuntimeError: D-axis of `query`, `key` and `value` is not the same.

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


def fused_infer_attention_score(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None,
                                actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None,
                                quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None,
                                key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None,
                                value_antiquant_offset=None, block_table=None, query_padding_size=None,
                                kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None,
                                actual_shared_prefix_len=None, num_heads=1, scale=1.0, pre_tokens=2147483647,
                                next_tokens=2147483647, input_layout='BSH', num_key_value_heads=0, sparse_mode=0,
                                inner_precise=1, block_size=0, antiquant_mode=0, key_antiquant_mode=0,
                                value_antiquant_mode=0, softmax_lse_flag=False):
    r"""
    This is a FlashAttention function designed for both incremental and full inference scenarios. It supports full
    inference scenarios (PromptFlashAttention) as well as incremental inference scenarios (IncreFlashAttention).
    When the S dimension of the query tensor (Q_S) equals 1, it enters the IncreFlashAttention branch; otherwise,
    it enters the PromptFlashAttention branch.

    .. math::

        Attention(Q,K,V) = Softmax(\frac{QK^{T}}{\sqrt{d}})V

    .. warning::
        - This is an experimental API that is subject to change or deletion.
        - For Ascend, only the Atlas A2 training series products and Atlas 800I A2 inference products are currently
          supported.

    Note:
        - The data layout formats of query, key and value can be interpreted from multiple dimensions, as shown below:

          - B, Batch size. Represents the batch size of the input samples.
          - S, Sequence length. Represents the sequence length of the input samples. S1 represents the sequence length
            of the query, and S2 represents the sequence length of the key/value.
          - H, Head size. Represents the size of the hidden layer.
          - N, Head nums. Represents the number of attention heads.
          - D, Head dims. Represents the smallest unit size of the hidden layer, satisfying :math:`D = H / N`.

    Args:
        query (Tensor): The query input of the attention structure, with data type of float16, bfloat16 or int8.
            Input tensor of shape :math:`(B, S, H)`, :math:`(B, N, S, D)`, or :math:`(B, S, N, D)`.
        key (Union[Tensor, tuple[Tensor], list[Tensor]]): The key input of the attention structure, with data type
            of float16, bfloat16 or int8. Input tensor of shape :math:`(B, S, H)`, :math:`(B, N, S, D)`, or
            :math:`(B, S, N, D)`.
        value (Union[Tensor, tuple[Tensor], list[Tensor]]): The value input of the attention structure, with data
            type of float16, bfloat16 or int8. Input tensor of shape :math:`(B, S, H)`, :math:`(B, N, S, D)`, or
            :math:`(B, S, N, D)`.

    Keyword Args:
        pse_shift (Tensor, optional): The padding mask tensor with data type of float16 or bfloat16.
            Default: ``None``.

            - When Q_S is not 1, if pse_shift is of type float16, the query must be of type float16 or int8.
              If pse_shift is of type bfloat16, the query must also be of type bfloat16. The input shape
              must be either :math:`(B, N, Q\_S, KV\_S)` or :math:`(1, N, Q\_S, KV\_S)`, where Q_S corresponds to the
              S dimension of the query shape, and KV_S corresponds to the S dimension of the key and value shapes.
              For scenarios where the KV_S of pse_shift is not 32-aligned, it is recommended to pad it
              to 32 bytes to improve performance. The padding values for the extra portions are not restricted.
            - When Q_S is 1, if pse_shift is of type float16, the query must also be of type float16.
              If pse_shift is of type bfloat16, the query must be of type bfloat16. The input shape must be
              :math:`(B, N, 1, KV\_S)` or :math:`(1, N, 1, KV\_S)`, where KV_S corresponds to the S dimension of the
              key/value shapes. For scenarios where the KV\_S of pse_shift is not 32-aligned, it is recommended
              to pad it to 32 bytes to improve performance. The padding values for the extra portions are not
              restricted.

        atten_mask (Tensor, optional): The attention mask tensor for the result of query*key with data type of int8,
            uint8 or bool. For each element, 0 indicates retention and 1 indicates discard.
            Default: ``None``.

            - When Q_S is not 1, the recommended input shapes are Q_S,KV_S; B,Q_S,KV_S; 1,Q_S,KV_S; B,1,Q_S,KV_S
              or 1,1,Q_S,KV_S.
            - When Q_S is 1, the recommended input shapes are B,KV_S; B,1,KV_S or B,1,1,KV_S.

        actual_seq_lengths (Union[tuple[int], list[int], Tensor], optional): Describe actual sequence length of the
            query with data type of int64. If this parameter is not specified, it can be set to None, indicating that
            it matches the S dimension of the query shape. Constraint: The effective sequence length for each batch in
            this parameter should not exceed the corresponding batch's sequence length in the query. When Q_S is 1, this
            parameter is ignored.
            Default: ``None``.
        actual_seq_lengths_kv (Union[tuple[int], list[int], Tensor], optional): Describe actual sequence length of the
            key and value with data type of int64. If this parameter is not specified, it can be set to None,
            indicating that it matches the S dimension of the key and value shape. Constraint: The effective sequence
            length for each batch in this parameter should not exceed the corresponding batch's sequence length in the
            key and value.
            Default: ``None``.
        dequant_scale1 (Tensor, optional): Quantization factors for inverse quantization after BMM1 with data type of
            uint64. Supports per-tensor mode. If not used, set it to None.
            Default: ``None``.
        quant_scale1 (Tensor, optional): Quantization factors for quantization before BMM2 with data type of float32.
            Supports per-tensor mode. If not used, set it to None.
            Default: ``None``.
        dequant_scale2 (Tensor, optional): Quantization factors for inverse quantization after BMM2 with data type of
            uint64. Supports per-tensor mode. If not used, set it to None.
            Default: ``None``.
        quant_scale2 (Tensor, optional): Quantization factors for output quantization with data type of float32,
            bfloat16. Supports per-tensor and per-channel modes. If not used, set it to None.
            Default: ``None``.
        quant_offset2 (Tensor, optional): Quantization offset for output quantization with data type of float32,
            bfloat16. Supports per-tensor and per-channel modes. If not used, set it to None.
            Default: ``None``.

            For scenarios where the input is int8 and the output is int8: the parameters dequant_scale1, quant_scale1,
            dequant_scale2, and quant_scale2 must all be provided. The parameter quant_offset2 is optional and defaults
            to 0 if not specified.

            - When the output is int8 and quant_scale2 and quant_offset2 are per-channel, left padding, Ring Attention,
              or D-axis misalignment (not 32-aligned) scenarios are not supported.
            - When the output is int8, scenarios with sparse_mode as band and pre_tokens/next_tokens being negative are
              not supported.
            - When the output is int8, if quant_offset2 is not None and empty tensor, and the sparse_mode, pre_tokens,
              and next_tokens meet the following conditions, certain rows of the matrix may not participate in
              calculations, leading to errors. This scenario will be intercepted (solution: if this scenario should
              not be intercepted, quantization should be performed outside the FIA interface, not enabled inside the
              FIA interface):

              - sparse_mode = 0, if atten_mask is a not None and each batch's
                actual_seq_lengths - actual_seq_lengths_kv - pre_tokens > 0 or next_tokens < 0, it will meet the
                interception condition.
              - sparse_mode = 1 or 2, no interception condition will occur.
              - sparse_mode = 3, if each batch's actual_seq_lengths - actual_seq_lengths_kv < 0, it will meet the
                interception condition.
              - sparse_mode = 4, if pre_tokens < 0 or each batch's
                next_tokens + actual_seq_lengths - actual_seq_lengths_kv < 0, it will meet the interception
                condition.

            For scenarios where the input is int8 and the output is float16: the parameters dequant_scale1,
            quant_scale1, and dequant_scale2 must all be provided.

            For scenarios where the input is entirely float16 or bfloat16 and the output is int8: the parameter
            quant_scale2 must be provided. The parameter quant_offset2 is optional and defaults to 0 if not specified.

            The parameters quant_scale2 and quant_offset2 support both per-tensor and per-channel modes and two data
            types: float32 and bfloat16. If quant_offset2 is provided, its type and shape must match those of
            quant_scale2. When the input is bfloat16, both float32 and bfloat16 are supported; otherwise, only float32
            is supported. For per-channel mode: When the output layout is BSH, the product of all dimensions in
            quant_scale2 must equal H. For other layouts, the product must equal N * D. When the output layout is BSH,
            it is recommended to set the shape of quant_scale2 as :math:`(1, 1, H)` or :math:`(H)`. When the output
            layout is BNSD, it is recommended to set the shape as :math:`(1, N, 1, D)` or :math:`(N, D)`. When the
            output layout is BSND, it is recommended to set the shape as :math:`(1, 1, N, D)` or :math:`(N, D)`.

        antiquant_scale (Tensor, optional): Inverse quantization factors with data type of float16, float32 or bfloat16.
            Only support float16 when Q_S > 1. Supports per-tensor, per-channel and per-token modes.
            Default: ``None``.
        antiquant_offset (Tensor, optional): Inverse quantization offset with data type of float16, float32 or bfloat16.
            Only support float16 when Q_S > 1. Supports per-tensor, per-channel and per-token modes.
            Default: ``None``.

            Constraints for antiquant_scale and antiquant_offset parameters:

            - Supports three modes: per-channel, per-tensor, and per-token:

              - Per-channel mode: The shape of both parameters in the BNSD layout is :math:`(2, N, 1, D)`, the shape
                in the BSND layout is :math:`(2, N, D)`, and the shape in the BSH layout is :math:`(2, H)`, where 2
                corresponds to the key and value, and N represents num_key_value_heads. The parameter data type is
                the same as the query data type, and antiquant_mode should be set to 0.
              - Per-tensor mode: The shape of both parameters is :math:`(2)`, the data type is the same as the query
                data type, and antiquant_mode should be set to 0.
              - Per-token mode: The shape of both parameters is :math:`(2, B, S)`, the data type is fixed to float32,
                and antiquant_mode should be set to 1.

            - Supports both symmetric and asymmetric quantization:

              - Asymmetric quantization mode: Both antiquant_scale and antiquant_offset must be provided.
              - Symmetric quantization mode: antiquant_offset can be empty (``None``). If antiquant_offset is empty,
                symmetric quantization is performed. If antiquant_offset is provided, asymmetric quantization is
                performed.

        key_antiquant_scale (Tensor, optional): Inverse quantization factors for the key, with data type of float16,
            float32 or bfloat16, when the KV fake quantization parameters are separated.
            Supports per-tensor, per-channel and per-token modes.
            Default: ``None``. Invalid when Q_S > 1.
        key_antiquant_offset (Tensor, optional): Inverse quantization offset for the key, with data type of float16,
            float32 or bfloat16, when the KV fake quantization parameters are separated.
            Supports per-tensor, per-channel and per-token modes.
            Default: ``None``. Invalid when Q_S > 1.
        value_antiquant_scale (Tensor, optional): Inverse quantization factors for the value, with data type of
            float16, float32 or bfloat16, when the KV fake quantization parameters are separated.
            Supports per-tensor, per-channel and per-token modes.
            Default: ``None``. Invalid when Q_S > 1.
        value_antiquant_offset (Tensor, optional): Inverse quantization offset for the value, with data type of
            float16, float32 or bfloat16, when the KV fake quantization parameters are separated.
            Supports per-tensor, per-channel and per-token modes.
            Default: ``None``. Invalid when Q_S > 1.
        block_table (Tensor, optional): Block mapping table in KV cache for PageAttention, with data type of int32.
            If not used, set it to None.
            Default: ``None``. Invalid when Q_S > 1.
        query_padding_size (Tensor, optional): The query padding size with data type of int64. Indicates whether the
            data in each batch of the query is right-aligned, and how many elements are right-aligned.
            Default: ``None``. Invalid when Q_S is 1.
        kv_padding_size (Tensor, optional): The key and value padding size with data type of int64. Indicates whether
            the data in each batch of the key and value is right-aligned, and how many elements are right-aligned.
            Default: ``None``. Invalid when Q_S is 1.
        key_shared_prefix (Tensor, optional): Shared prefix of the key. This is a reserved parameter and is not yet
            enabled. Default: ``None``.
        value_shared_prefix (Tensor, optional): Shared prefix of the value. This is a reserved parameter and is not yet
            enabled. Default: ``None``.
        actual_shared_prefix_len (Union[tuple[int], list[int], Tensor], optional): Describe the actual length of shared
            prefix. This is a reserved parameter and is not yet enabled.
            Default: ``None``.
        num_heads (int, optional): The number of heads in the query, equal to N when input_layout is BNSD.
            Default: ``1``.
        scale (double, optional): The scale value indicating the scale coefficient, which serves as the scalar value for
            the Muls in the calculation. Generally, the value is :math:`1.0 / \sqrt{d}`. Default: ``1.0``.
        pre_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted forward.
            Default: ``2147483647``. Invalid when Q_S is 1.
        next_tokens (int, optional): Parameter for sparse computation, represents how many tokens are counted backward.
            Default: ``2147483647``. Invalid when Q_S is 1.
        input_layout (str, optional): Specifies the layout of input query, key and value. BSH, BNSD, BSND or
            BNSD_BSND is supported. When the layout is BNSD_BSND, it means the input is in the BNSD format and
            the output is in the BSND format, this is only supported when Q_S > 1.
            Default: ``BSH``.
        num_key_value_heads (int, optional): Head numbers of key/value which are used in GQA (Grouped-Query Attention)
            scenario. Default: ``0``. A value of 0 means it is equal to the number of key/value heads. The num_heads
            must be divisible by num_key_value_heads, and the ratio of num_heads to num_key_value_heads must not be
            greater than 64. When the layout is BNSD, the num_key_value_heads must also equals to the N dimension of
            the key/value shapes, otherwise, an execution error will occur.
        sparse_mode (int, optional): Indicates sparse mode. Default ``0``. Invalid when Q_S is 1.

            - 0: Indicates the defaultMask mode. If atten_mask is not passed, the mask operation is not performed,
              and pre_tokens and next_tokens(internally assigned as INT_MAX) are ignored. If passed in, the complete
              atten_mask matrix (S1 * S2) also must be passed in, indicating that the part between pre_tokens and
              next_tokens needs to be calculated.
            - 1: Represents allMask. The complete atten_mask matrix (S1 * S2) is required.
            - 2: Represents the mask in leftUpCausal mode. The optimized atten_mask matrix (2048*2048) is required.
            - 3: Represents the mask in rightDownCausal mode, corresponding to the lower triangular scenario divided by
              the right vertex. The optimized atten_mask matrix (2048*2048) is required.
            - 4: Represents the mask in band mode, that is, the part between counting pre_tokens and next_tokens. The
              optimized atten_mask matrix (2048*2048) is required.
            - 5: Represents the prefix scenario, not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.

        inner_precise (int, optional): There are four modes: 0, 1, 2, and 3, represented by 2 bits: bit 0 (bit0)
            represents the choice for high precision or high performance, and bit 1 (bit1) indicates whether row-wise
            invalidity correction is applied.

            - 0: Enable high-precise mode, without row-wise invalidity correction.
            - 1: High-performance mode, without row-wise invalidity correction.
            - 2: Enable high-precise mode, with row-wise invalidity correction.
            - 3: High-performance mode, with row-wise invalidity correction.

            When Q_S > 1, if sparse_mode is 0 or 1 and a user-defined mask is provided, it is recommended to enable
            row-wise invalidity correction. Only support 0 and 1 when Q_S is 1. Default: ``1``.

            High-precise and high-performance are only effective for float16 inputs; Row invalidity correction
            is effective for float16, bfloat16, and int8 inputs.
            Currently, 0 and 1 are reserved configuration values. If there is a situation where an entire row in the
            "mask portion involved in computation" is all 1s, precision may degrade. In such cases, you can try
            setting this parameter to 2 or 3 to enable row invalidity correction for improved precision. However,
            this configuration will result in decreased performance.
            If the function can detect the presence of invalid row scenarios, e.g. in cases where sparse_mode is 3
            and S_q > S_kv, it will automatically enable row invalidity computation.

        block_size (int, optional): Maximum number of tokens per block in the KV cache block for PageAttention.
            Default: ``0``. Invalid when Q_S > 1.
        antiquant_mode (int, optional): Fake-quantization mode, 0: per-channel (per-channel includes per-tensor),
            1: per-token. The per-channel and per-tensor modes can be distinguished by the dimension of the input
            shape. When the dimension is 1, it runs in per-tensor mode; otherwise, it runs in per-channel mode.
            Default: ``0``. Invalid when Q_S > 1.
        key_antiquant_mode (int, optional): Fake-quantization mode for the key. 0: per-channel (per-channel includes
            per-tensor), 1: per-token. Default: ``0``. Invalid when Q_S > 1.
        value_antiquant_mode (int, optional): Fake-quantization mode for the value. 0: per-channel (per-channel includes
            per-tensor), 1: per-token. Default: ``0``. Invalid when Q_S > 1.
        softmax_lse_flag (bool, optional): Whether to output softmax_lse. Default: ``False``.

    Returns:
        attention_out (Tensor), the attention score with data type of float16, bfloat16 or int8. When the input_layout
        is BNSD_BSND, the shape is :math:`(B, S, N, D)`. In all other cases, the shape is consistent with the
        input query shape.

        softmax_lse (Tensor), the softmax_lse with data type of float32, obtained by taking the lse (log, sum and exp)
        of the result of query*key. Specifically, the Ring Attention algorithm first takes the max of the result of
        query*key, obtaining softmax_max. The result of query*key is then subtracted by softmax_max, followed by
        taking exp, and then the sum is computed to obtain softmax_sum. Finally, the log of softmax_sum is taken,
        and softmax_max is added to obtain softmax_lse. The softmax_lse is only calculated when softmax_lse_flag
        is True, and the shape would be :math:`(B, N, Q\_S, 1)`. If softmax_lse_flag is False, then a tensor with
        shape :math:`(1)` filled with zeros would be returned. In GE backend, please ensure that the softmax_lse_flag
        is enabled before using softmax_lse; otherwise, an exception will occur.

    Constraints:
        - Full Inference Scenario (Q_S > 1):

          - Query, key, and value inputs functional usage restrictions:

            - The B axis supports values less than or equal to 65535. If the input type includes int8, or
              if the input type is float16 or bfloat16 and the D axis is not 16-aligned, the B axis is only
              supported up to 128.
            - The N axis supports values less than or equal to 256, and the D axis supports values less than
              or equal to 512.
            - The S axis supports values less than or equal to 20,971,520 (20M). In some long sequence
              scenarios, if the computation load is too large, it may cause a timeout in the PFA operator
              (AICore error type with errorStr: "timeout or trap error"). In this case, it is recommended to
              perform an S split. Note: The computational load is affected by B, S, N, D, etc.; the larger the
              values, the greater the computational load. Typical long sequence timeout scenarios (where the
              product of B, S, N, and D is large) include, but are not limited to:

              1. B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152;
              2. B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520;
              3. B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152;
              4. B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152.

            - When the query, key, value, or attention_out type includes int8, the D axis must be 32-aligned.
              If all types are float16 or bfloat16, the D axis must be 16-aligned.

          - The sparse_mode parameter currently only supports values 0, 1, 2, 3, and 4. Using any other values
            will result in an error.

            - When sparse_mode = 0, if the atten_mask is None, or if the atten_mask is provided in the left
              padding scenario, the input parameters pre_tokens and next_tokens are ignored.
            - When sparse_mode = 2, 3, or 4, the shape of the atten_mask must be S,S or 1,S,S or 1,1,S,S, where
              S must be fixed at 2048, and the user must ensure the atten_mask is a lower triangular matrix. If
              no atten_mask is provided or if the shape is incorrect, an error will occur.
            - In sparse_mode = 1, 2, 3 scenarios, the pre_tokens and next_tokens inputs are ignored and assigned
              according to the relevant rules.

          - The KV cache de-quantization only supports queries of type float16, where int8 keys and values are
            de-quantized to float16. The data range of the input key/value and the antiquant_scale must have a
            product within the range of (-1, 1). High-performance mode can guarantee precision; otherwise,
            high-precision mode should be enabled to ensure accuracy.

          - Query left padding scenario:

            - In the query left padding scenario, the formula for calculating the starting point of the query
              transport is: Q_S - query_padding_size - actual_seq_lengths. The formula for the
              ending point of the query transport is: Q_S - query_padding_size. The query transport
              starting point must not be less than 0, and the ending point must not exceed Q_S; otherwise,
              the results will be incorrect.
            - If the kv_padding_size in the query left padding scenario is less than 0, it will be set to 0.
            - The query left padding scenario must be enabled together with the actual_seq_lengths parameter,
              otherwise, the default is the query right padding scenario.
            - The query left padding scenario does not support PageAttention and cannot be enabled together with
              the block_table parameter.

          - KV left padding scenario:

            - In the KV left padding scenario, the formula for calculating the starting point of the key and
              value transport is: KV_S - kv_padding_size - actual_seq_lengths_kv. The formula
              for the ending point of the key and value transport is: KV_S - kv_padding_size. The
              key and value transport starting point must not be less than 0, and the ending point must not
              exceed KV_S; otherwise, the results will be incorrect.
            - If the kv_padding_size in the KV left padding scenario is less than 0, it will be set to 0.
            - The KV left padding scenario must be enabled together with the actual_seq_lengths_kv parameter,
              otherwise, the default is the KV right padding scenario.
            - The KV left padding scenario does not support PageAttention and cannot be enabled together with
              the block_table parameter.

          - pse_shift functional usage restrictions:

            - This function is supported when the query data type is float16, bfloat16, or int8.
            - If the query data type is float16 and pse_shift is enabled, it will force high-precision mode,
              inheriting the limitations of high-precision mode.
            - Q_S must be greater than or equal to the length of the query S, and KV_S must be greater than
              or equal to the length of the key S.

          - KV fake quantization parameter separation is not currently supported.

        - Incremental Inference Scenario (Q_S is 1):

          - Query, key, and value inputs functional usage restrictions:

            - The B axis supports values less than or equal to 65,536.
            - The N axis supports values less than or equal to 256.
            - The D axis supports values less than or equal to 512.
            - Scenarios where the input types of query, key, and value are all int8 are not supported.

          - Page attention scenario:

            - The necessary condition to enable page attention is that the block_table exists and is valid.
              The key and value are arranged in contiguous memory according to the indices in the block_table.
              The key and value dtypes supported are float16, bfloat16, and int8. In this scenario, the
              input_layout parameter for key and value is invalid.
            - block_size is a user-defined parameter, and its value will affect the performance of page
              attention. When enabling page attention, a non-zero value for block_size must be provided, and
              the maximum value for block_size is 512.
            - If the input types of key and value are float16 or bfloat16, they must be 16-aligned. If the
              input types are int8, they must be 32-aligned, with 128 being recommended. In general, page
              attention can increase throughput but may lead to a performance decrease.
            - In the page attention enabled scenario, when the KV cache layout is (blocknum, block_size, H) and
              num_key_value_heads * D exceeds 64K, an error will be reported due to hardware
              instruction constraints. This can be resolved by enabling GQA (reducing num_key_value_heads) or
              adjusting the KV cache layout to (blocknum, num_key_value_heads, block_size, D).
            - The product of all dimensions of the shape of the key and value tensors in the page attention
              scenario must not exceed the representable range of int32.

          - In the page attention enabled scenario, the input S must be greater than or equal to
            max_block_num_per_seq * block_size.

            - Enabling attention mask (e.g., mask shape = (B, 1, 1, S))
            - Enabling pse_shift (e.g., pse_shift shape = (B, N, 1, S))
            - Enabling fake quantization in per-token mode (e.g., antiquant_scale and antiquant_offset shapes =
              (2, B, S)) are also supported.

          - KV left padding scenario:

            - In the KV left padding scenario, the formula for calculating the starting point of the KV cache
              transport is: KV_S - kv_padding_size - actual_seq_lengths. The formula for the endpoint of the
              KV cache transport is: KV_S - kv_padding_size. If the starting point or endpoint of the KV cache
              is less than 0, the returned data result will be all zeros.
            - If kv_padding_size is less than 0 in the KV left padding scenario, it will be set to 0.
            - The KV left padding scenario must be enabled together with the actual_seq_lengths parameter,
              otherwise, it defaults to the KV right padding scenario.
            - The KV left padding scenario must be enabled together with the atten_mask parameter, and the
              atten_mask must be correctly applied to hide invalid data. Otherwise, accuracy issues may arise.

          - pse_shift functional usage restrictions:

            - The data type of pse_shift must match the data type of the query.
            - Only the D axis alignment is supported, meaning the D axis must be divisible by 16.

          - KV fake quantization parameter separation:

            - key_antiquant_mode and value_antiquant_mode must be consistent.
            - key_antiquant_scale and value_antiquant_scale must either both be empty or both non-empty.
            - key_antiquant_offset and value_antiquant_offset must either both be empty or both non-empty.
            - When both key_antiquant_scale and value_antiquant_scale are non-empty, their shapes must be
              consistent.
            - When both key_antiquant_offset and value_antiquant_offset are non-empty, their shapes must be
              consistent.


    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> B, N, S, D = 1, 8, 1024, 128
        >>> query = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        >>> key = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        >>> value = Tensor(np.random.rand(B, N, S, D).astype(np.float16))
        >>> out = ops.fused_infer_attention_score(query, key, value, num_heads=N, input_layout='BNSD')
        >>> print(out[0].shape)
        (1, 8, 1024, 128)
    """
    fias_op = _get_cache_prim(FusedInferAttentionScore)(num_heads, scale, pre_tokens, next_tokens, input_layout,
                                                        num_key_value_heads, sparse_mode, inner_precise, block_size,
                                                        antiquant_mode, softmax_lse_flag, key_antiquant_mode,
                                                        value_antiquant_mode)
    key_list = key if isinstance(key, (tuple, list)) else [key]
    value_list = value if isinstance(value, (tuple, list)) else [value]
    return fias_op(query, key_list, value_list, pse_shift, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
                   dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale,
                   antiquant_offset, block_table, query_padding_size, kv_padding_size, key_antiquant_scale,
                   key_antiquant_offset, value_antiquant_scale, value_antiquant_offset, key_shared_prefix,
                   value_shared_prefix, actual_shared_prefix_len)


class WhileLoop(Primitive):
    """
    Provide a useful op for reducing compilation times of while loop.
    The execution logic of the WhileLoop operator can be roughly represented by the following code:

    .. code-block:: python

        def WhileLoop(cond_func, loop_func, init_val):
            while(cond_func(init_val)):
                init_val = loop_func(init_val)
            return init_val

    The current WhileLoop operator has the following syntactic limitations:

    - Using a side-effect function as `loop_func` is currently not support,
      such as operations that modify parameters, global variables, etc.
    - The return value of `loop_func` being of a different type or shape
      from the `init_val` is currently not support.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **cond_func** (Function) - The condition function.
        - **loop_func** (Function) - The loop function, take one argument and
          return value has the same type with input argument.
        - **init_val** (Union[Tensor, number, str, bool, list, tuple, dict]) - The initial value.

    Outputs:
        Union[Tensor, number, str, bool, list, tuple, dict], the final result of the while loop,
        has same type and shape with input `init_val` .

    Raises:
        TypeError: If `cond_func` is not a function.
        TypeError: If `loop_func` is not a function.
        ValueError: If `loop_func` cannot take `init_val` as input or has different
                    output type or shape with `init_val` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> def loop_while_fun(init_val):
        ...     val = init_val
        ...     val = val + 1
        ...     return val
        ...
        >>> init_state = 10
        >>> while_loop = ops.WhileLoop()
        >>> result = while_loop(lambda x : x < 100, loop_while_fun, init_state)
        >>> print(result)
        100
    """

    @prim_attr_register
    def __init__(self):
        """Initialize WhileLoop."""

    def __call__(self, cond_func, loop_func, init_val):
        validator.check_value_type("cond_func", cond_func,
                                   [types.FunctionType, types.MethodType], "WhileLoop")
        validator.check_value_type("loop_func", loop_func,
                                   [types.FunctionType, types.MethodType], "WhileLoop")
        val = init_val
        try:
            while cond_func(val):
                val = loop_func(val)
        except Exception as e:
            raise ValueError(f"Invalid loop_func, please check input arguments and "
                             f"return value, error info: {e}") from e
        return val


class Scan(Primitive):
    """
    Scan a function over an array while the processing of the current element
    depends on the execution result of the previous element.
    The execution logic of the Scan operator can be roughly represented by the following code:

    .. code-block:: python

        def Scan(loop_func, init, xs, length=None):
            if xs is None:
                xs = [None] * length
            carry = init
            ys = []
            for x in xs:
                carry, y = loop_func(carry, x)
                ys.append(y)
            return carry, ys

    The current Scan operator has the following syntactic limitations:

    - Using a side-effect function as `loop_func` is currently not support,
      such as operations that modify parameters, global variables, etc.
    - The first element of the return value of `loop_func` being of a different
      type or shape from the `init_val` is currently not support.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **loop_func** (Function) - The loop function.
        - **init** (Union[Tensor, number, str, bool, list, tuple, dict]) - An initial loop carry value.
        - **xs** (Union[tuple, list, None]) - The value over which to scan.
        - **length** (Union[int, None], optional) - The size of xs. Default: ``None`` .
        - **unroll** (bool, optional) - The flag for whether to perform loop unrolling in compile process.
          Default: ``True`` .

    Outputs:
        Tuple(Union[Tensor, number, str, bool, list, tuple, dict], list). Output of scan loop,
        a tuple with two elements, the first element has same type and shape with init argument,
        and the second is a list containing the results of each loop.

    Raises:
        TypeError: If `loop_func` is not a function.
        TypeError: If `xs` is not a tuple, a list or None.
        TypeError: If `length` is not an int or None.
        TypeError: If `unroll` is not a bool.
        ValueError: If `loop_func` cannot take `init` and element of `xs` as inputs.
        ValueError: If the return value of `loop_func` is not a tuple with two elements,
                    or the first element of the tuple has different type or shape from `init` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> def cumsum(res, el):
        ...     res = res + el
        ...     return res, res
        ...
        >>> a = [1, 2, 3, 4]
        >>> result_init = 0
        >>> scan_op = ops.Scan()
        >>> result = scan_op(cumsum, result_init, a)
        >>> print(result == (10, [1, 3, 6, 10]))
        True
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Scan."""

    def __call__(self, loop_func, init, xs, length=None, unroll=True):
        validator.check_value_type("loop_func", loop_func,
                                   [types.FunctionType, types.MethodType], "Scan")
        validator.check_value_type("xs", xs, [list, tuple, None], "Scan")
        if xs is None:
            validator.check_value_type("length", length, [int], "Scan")
            xs = [None] * length
        carry = init
        length = len(xs)
        if not length:
            return init, []
        try:
            carry, y = loop_func(carry, xs[0])
            ys = [y]
            i = 1
            while i < length:
                carry, y = loop_func(carry, xs[i])
                ys.append(y)
                i = i + 1
        except Exception as e:
            raise ValueError(f"Invalid loop_func, please check input arguments and "
                             f"return value, error info: {e}") from e
        return carry, ys


class ForiLoop(Primitive):
    """
    Performs a loop operation within the specified range.
    The execution logic of the ForiLoop operator can be roughly represented by the following code:

    .. code-block:: python

        def ForiLoop(lower, upper, loop_func, init_val):
            for i in range(lower, upper):
                init_val = loop_func(i, init_val)
            return init_val

    The current ForiLoop operator has the following syntactic limitations:

    - Using a side-effect function as `loop_func` is currently not support,
      such as operations that modify parameters, global variables, etc.
    - The return value of `loop_func` being of a different type or shape
      from the `init_val` is currently not support.
    - Negative numbers or custom increments is currently not support.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Inputs:
        - **lower** (Union[int, Tensor]) - The start index of loop.
        - **upper** (Union[int, Tensor]) - The end index of loop.
        - **loop_func** (Function) - The loop function, takes two arguments.
        - **init_val** (Union[Tensor, number, str, bool, list, tuple, dict]) - The init value.
        - **unroll** (bool, optional) - The flag for whether unroll in compile process,
          only valid when the number of loop iterations is determined. Default: ``True`` .

    Outputs:
        Union[Tensor, number, str, bool, list, tuple, dict], the final result of the loop,
        has same type and shape with input `init_val` .

    Raises:
        TypeError: If `lower` is not an int or a Tensor.
        TypeError: If `upper` is not an int or a Tensor.
        TypeError: If `loop_func` is not a function.
        ValueError: If `loop_func` cannot take index and `init_val` as arguments or if the type
                    of output it produces is different from the type or shape of `init_val` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> def cumsum(index, res):
        ...     return index + res
        ...
        >>> result_init = 0
        >>> fori_loop = ops.ForiLoop()
        >>> result = fori_loop(0, 4, cumsum, result_init)
        >>> print(result)
        6
    """

    @prim_attr_register
    def __init__(self):
        """Initialize ForiLoop."""

    def __call__(self, lower, upper, loop_func, init_val, unroll=True):
        validator.check_value_type("lower", lower, [int, Tensor], "ForiLoop")
        validator.check_value_type("upper", upper, [int, Tensor], "ForiLoop")
        validator.check_value_type("loop_func", loop_func,
                                   [types.FunctionType, types.MethodType], "ForiLoop")
        val = init_val
        try:
            for i in range(lower, upper):
                val = loop_func(i, val)
        except Exception as e:
            raise ValueError(f"Invalid loop_func, please check input arguments and "
                             f"return value, error info: {e}") from e
        return val
