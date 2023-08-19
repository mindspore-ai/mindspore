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

import numpy as np

from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_arg_handler as handler
from mindspore.common import Tensor, CSRTensor, COOTensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops._tracefunc import PackFunc

class BatchNorm(Primitive):
    """
    BatchNorm.
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

    @prim_attr_register
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
        self.data_format = handler.py_format_to_enum(data_format)

    def __call__(self, *args):
        return super().__call__(self, *args, self.is_training, self.epsilon,
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
        - For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.

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


def infer_value_for_Tile(x, multiples):
    """Infer value for Tile op."""
    if x is None or multiples is None or None in multiples:
        return None
    return Tensor(np.tile(x.asnumpy(), multiples))
