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

from math import log
import numpy as np
from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register, prim_arg_register
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_arg_handler as handler
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops._tracefunc import PackFunc
from mindspore.common._utils import is_shape_unknown
from mindspore import _checkparam as validator
from mindspore.ops.operations.manually_defined._inner import ScalarCast


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
        return log(x)


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
    .. code-block::

        prim = ops.BatchNorm(is_training, epsilon, momentum, data_format)
        out = prim(input_x, scale, bias, mean, variance)

    is equivalent to

    .. code-block::

        ops.batch_norm_(input_x, scale, bias, mean, variance, is_training, epsilon, momentum, data_format)

    Refer to :func:`mindspore.ops.batch_norm_` for more details.
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


def rank(x):
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
    return rank_op(x)


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


class Tile(Primitive):
    """
    Tile.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize."""

    def check_elim(self, *args):
        """Function for checking eliminate."""
        base_tensor, multiplier = args
        if not isinstance(base_tensor, Tensor):
            raise TypeError(f"For '{self.name}', the type of 'input_x' must be Tensor, "
                            f"but got {type(base_tensor).__name__}.")
        if PackFunc.is_tracing() and not PackFunc.current.is_pynative_mode:
            return (False, None)
        if not isinstance(multiplier, tuple):
            raise TypeError(f"For '{self.name}', the type of 'multiplier' must be tuple, "
                            f"but got {type(multiplier).__name__}.")

        if all(v == 1 for v in multiplier) and len(base_tensor.shape) >= len(multiplier):
            from mindspore.ops.auto_generate.gen_ops_def import Identity
            ret = Identity()(base_tensor)
            return (True, ret)
        return (False, None)


def tile(input, multiples):
    r"""
    Replicates an input tensor with given multiples times.

    Creates a new tensor by replicating `input` `multiples` times. The i'th dimension of
    output tensor has `input.shape[i] * multiples[i]` elements, and the values of `input`
    are replicated `multiples[i]` times along the i'th dimension.

    Note:
        The length of `multiples` must be greater or equal to the length of dimension in `input`.

    Args:
        input (Tensor): 1-D or higher dimensional Tensor. Set the shape of input tensor as
            :math:`(x_1, x_2, ..., x_S)` .

        multiples (tuple[int]): The parameter that specifies the number of replications,
            the parameter type is tuple, and the data type is int, i.e., :math:`(y_1, y_2, ..., y_S)`.
            The length of `multiples` cannot be smaller than the length of the shape of `input`.
            Only constant value is allowed.

    Returns:
        Tensor, has the same data type as the `input`. Suppose the length of `multiples` is `d`,
        the dimension of `input` is `input.dim`, and the shape of `input` is :math:`(x_1, x_2, ..., x_S)`.

        - If `input.dim = d`, then the shape of their corresponding positions can be multiplied, and
          the shape of Outputs is :math:`(x_1*y_1, x_2*y_2, ..., x_S*y_S)`.
        - If `input.dim < d`, fill in multiple 1 in the length of the shape of `input` until their
          lengths are consistent. Such as set the shape of `input` as :math:`(1, ..., x_1, x_2, ..., x_S)`,
          then the shape of their corresponding positions can be multiplied, and the shape of Outputs is
          :math:`(1*y_1, ..., x_R*y_R, x_S*y_S)`.

    Raises:
        TypeError: If `multiples` is not a tuple or its elements are not all int.
        ValueError: If the elements of `multiples` are not all greater than 0.
        ValueError: If the length of `multiples` are smaller than the length of dimension in `input`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)
        >>> multiples = (2, 3)
        >>> output = ops.tile(input, multiples)
        >>> print(output)
        [[1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]
         [1.  2.  1.  2.  1.  2.]
         [3.  4.  3.  4.  3.  4.]]
        >>> multiples = (2, 3, 2)
        >>> output = ops.tile(input, multiples)
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
    return tile_op(input, multiples)


def scalar_cast(input_x, input_y):
    r"""
    Casts the input scalar to another type.

    Args:
        input_x (scalar): The input scalar. Only constant value is allowed.
        input_y (mindspore.dtype): The type to be cast. Only constant value is allowed.
            The value should only be mindspore.int64, mindspore.float64, or mindspore.bool_.

    Returns:
        Scalar, the type is the same as the python type corresponding to `input_y`.

    Raises:
        ValueError: if input_y's value is invalid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import ops
        >>> output = ops.scalar_cast(255.0, mindspore.int64)
        >>> print(output)
        255
    """
    scalar_cast_op = _get_cache_prim(ScalarCast)()
    return scalar_cast_op(input_x, input_y)

# Following is Python Infer Value.
# A valid infer value function should be:
#
# 1. named as infer_value_for_OpName
# 2. All inputs should pass without default value.
# 3. If not const input is given, return None. (for now)

def infer_value_for_Tile(x, multiples):
    """Infer value for Tile op."""
    if x is None or multiples is None or None in multiples:
        return None
    return Tensor(np.tile(x.asnumpy(), multiples))


def infer_value_for_Concat(input_x, axis):
    """Infer value for Concat op."""
    if input_x is None or None in input_x or axis is None:
        return None
    return Tensor(np.concatenate([x.asnumpy() for x in input_x], axis))


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


def infer_value_for_ReduceMax(input_x, axis, keep_dims):
    """Infer value for ReduceMax op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceMax')


def infer_value_for_ReduceMin(input_x, axis, keep_dims):
    """Infer value for ReduceMin op."""
    return _infer_value_for_Reduce(input_x, axis, keep_dims, 'ReduceMin')


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


def infer_value_for_Diag(input_x):
    """Infer value for Diag op."""
    if input_x is None:
        return None
    # do constant-folding only when x rank is 1
    if len(input_x.shape) != 1:
        return None
    ret = np.diag(input_x.asnumpy())
    return Tensor(ret)


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
        if dim_prod <= 0:
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
        out = Tensor(x.asnumpy().reshape(shape))
    return out
