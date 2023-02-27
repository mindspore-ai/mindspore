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
"""Hypercomplex Dense"""
import numbers
from typing import TypeVar, Type, Union

import mindspore
import mindspore.nn as nn
from mindspore._checkparam import Validator
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import ops as P
from mindspore.hypercomplex.hypercomplex._hc_dense_impl import _DenseImpl as DenseImpl
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel


TDenseImpl = TypeVar('TDenseImpl', bound=DenseImpl)


class Dense(nn.Cell):
    r"""
    The abstract part of dense connected layer.

    Applies dense connected layer for the second-order hypercomplex input. This layer implements the operation as:

    .. math::
        \text{out} = \text{linear}(\text{inp}, \text{kernel}) + \text{bias},

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which is defined and provided by the implementor part of the dense connected layer, :math:`\text{kernel}` is
    a hypercomplex weight matrix with the same data type as the :math:`inp` created by the layer, and
    :math:`\text{bias}` is a hypercomplex bias vector with the same data type as the :math:`inp` created by the layer
    (only if has_bias is True).

    This is not a self-sufficient class. In order to construct a fully connected layer, one should instantiate this
    class and an implementor class, which acts like a strategy pattern and determine the exact set of hypercomplex
    numbers. That implies the rules of multiplication and therefore affects how a linear transformation works.

    Args:
        dense_impl(DenseImpl): The implementor object of the dense connected layer. Essentially, the concrete class
            name of this argument defines the algebra that the dense layer will operate on.
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `inp`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, *, ..., *, in\_channels)`, with float16 or float32 data type,
          or :math:`(*, ..., *, in\_channels)`, with complex64 data type. In the former case '2' denotes that the input
          tensor belongs to the hypercomplex  domain and has got a real and an imaginary parts. The `in_channels` in
          `Args` has to be equal to :math:`in\_channels` in `Inputs`. The count of mediator dimensions denoted by '*'
          is arbitrary but must be at least one.

    Outputs:
        Tensor of the same data type as 'inp' and of shape :math:`(2, *, ..., *, out\_channels)`, with float16 or
        float32 data type, or :math:`(*, ..., *, out\_channels)`, with complex64 data type. The count of mediator
        dimensions is the same as one in 'inp'.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If any two of `inp`, `weight_init` and `bias_init` are Tensors of different data type.
        ValueError: If length of shape of `weight_init` is not equal to 3,
                    or shape[0] of 'weight_init' is not equal to 2,
                    or shape[1] of `weight_init` is not equal to `out_channels`,
                    or shape[2] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 2,
                    or shape[0] of 'bias_init' is not equal to 2,
                    or shape[1] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 dense_impl: Type[TDenseImpl],
                 in_channels: int,
                 out_channels: int,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number] = 'normal',
                 bias_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 has_bias: bool = True) -> None:
        """Initialize Dense."""
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)
        self.dtype = None

        self.weight_x = None
        self.weight_y = None
        if isinstance(weight_init, Tensor):
            self.dtype = weight_init.dtype
            if self.dtype in [mindspore.float16, mindspore.float32] and ( \
                    weight_init.ndim != 3
                    or weight_init.shape[0] != 2 \
                    or weight_init.shape[1] != out_channels \
                    or weight_init.shape[2] != in_channels):
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 3, and the first dim must be equal to 2, and the second dim must be "
                                 f"equal to 'out_channels', and the third dim must be equal to 'in_channels'. But got "
                                 f"'weight_init': {weight_init}, 'out_channels': {out_channels}, 'in_channels': "
                                 f"{in_channels}.")
            if self.dtype == mindspore.complex64 and ( \
                    weight_init.ndim != 2 \
                    or weight_init.shape[0] != out_channels \
                    or weight_init.shape[1] != in_channels):
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 2, and the first dim must be equal to 'out_channels', "
                                 f"and the second dim must be equal to 'in_channels'. But got "
                                 f"'weight_init': {weight_init}, 'out_channels': {out_channels}, 'in_channels': "
                                 f"{in_channels}.")

        self.dense_impl = dense_impl(weight_init, [out_channels, in_channels])

        self.bias_x = None
        self.bias_y = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if self.dtype is None:
                    self.dtype = bias_init.dtype
                elif self.dtype != bias_init.dtype:
                    raise TypeError("Data type of weight init tensor and the bias init tensor must be equal, "
                                    f"but got weight_init.dtype={self.dtype} and bias_init.dtype={bias_init.dtype}")
                if self.dtype in [mindspore.float16, mindspore.float32] and ( \
                        bias_init.ndim != 2 \
                        or bias_init.shape[0] != 2 \
                        or bias_init.shape[1] != out_channels):
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 2, and the second dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
                if self.dtype == mindspore.complex64 and ( \
                        bias_init.ndim != 1 \
                        or bias_init.shape[0] != out_channels):
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the only dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
                bias_init_x, bias_init_y = get_x_and_y(bias_init)
            else:
                bias_init_x = bias_init_y = bias_init
            self.bias_x = Parameter(initializer(bias_init_x, [out_channels]), name="bias_x")
            self.bias_y = Parameter(initializer(bias_init_y, [out_channels]), name="bias_y")

    def check_dense_input_shape(self, x: Tensor, x_dtype):
        msg_prefix = f"For '{self.cls_name}', the" if self.cls_name else "The"
        if x_dtype in [mindspore.float32, mindspore.float64] and (len(x) < 3 or x[0] != 2):
            raise ValueError(f"{msg_prefix} dimension of 'x' should not be less than 3, and the first dimension "
                             f"should be 2, but got {x}.")
        if x_dtype == mindspore.complex64 and len(x) < 2:
            raise ValueError(f"{msg_prefix} dimension of 'x' should not be less than 2, but got {x}.")
        return None

    def construct(self, u: Tensor) -> Tensor:
        """Construct"""
        if self.dtype is not None and self.dtype != u.dtype:
            raise TypeError("dtype must be equal to the data type of the inputs tensor, but got: "
                            f"dtype={self.dtype} and inputs.dtype={u.dtype}")
        u_shape = P.shape(u)
        self.check_dense_input_shape(u_shape, u.dtype)
        u_reshape = [-1, u_shape[-1]]
        if u.dtype in [mindspore.float32, mindspore.float64]:
            u_reshape = [2] + u_reshape
        if len(u_reshape) < len(u_shape):
            u = P.reshape(u, tuple(u_reshape))
        x, y = get_x_and_y(u)
        out_x, out_y = self.dense_impl(x, y)
        if self.has_bias:
            out_x = P.bias_add(out_x, self.bias_x)
            out_y = P.bias_add(out_y, self.bias_y)
        out = to_2channel(out_x, out_y, u.dtype)
        if len(u_reshape) < len(u_shape):
            out_shape = u_shape[:-1] + (-1,)
            out = P.reshape(out, out_shape)
        return out

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        return s
