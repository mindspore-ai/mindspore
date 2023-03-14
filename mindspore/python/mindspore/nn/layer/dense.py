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

"""basic"""
from __future__ import absolute_import

import math

import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer, Uniform
from mindspore.common.parameter import Parameter
from mindspore.ops.primitive import constexpr
from mindspore._checkparam import Validator
from mindspore.nn.cell import Cell

__all__ = ['BiDense']


@constexpr(check=False)
def _check_is_tensor(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(P.typeof(input_data), mstype.tensor_type):
        raise TypeError(f"For '{cls_name}', the '{param_name}' must be '{mstype.tensor_type}', "
                        f"but got '{P.typeof(input_data)}'")


class BiDense(Cell):
    r"""
    The bilinear dense connected layer.

    Applies dense connected layer for two inputs. This layer implements the operation as:

    .. math::
        y = x_1^T A x_2 + b,

    where :math:`x_{1}` is the first input tensor, :math:`x_{2}` is the second input tensor
    , :math:`A` is a weight matrix with the same data type as the :math:`x_{*}` created by the layer
    , and :math:`b` is a bias vector with the same data type as the :math:`x_{*}` created by the layer
    (only if has_bias is True).

    Args:
        in1_channels (int): The number of channels in the input1 space.
        in2_channels (int): The number of channels in the input2 space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter.
            The values of str refer to the function `initializer`. Default: None.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter.
            The values of str refer to the function `initializer`. Default: None.
        has_bias (bool): Specifies whether the layer uses :math:`\text{bias}` vector. Default: True.

    Shape:
        - **input1** - :math:`(*, H_{in1})` where :math:`H_{in1}=\text{in1_channels}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - **input2** - :math:`(*, H_{in2})` where :math:`H_{in2}=\text{in2_channels}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - **output** - :math:`(*, H_{out})` where :math:`H_{out}=\text{out_channels}` and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          are the same shape as the inputs.

    Dtype:
        - **input1** (Tensor) - The dtype must be float16 or float32 and be same as **input2**.
        - **input1** (Tensor) - The dtype must be float16 or float32 and be same as **input1**.
        - **output** (Tensor) - With the same dtype as the inputs.

    Weights:
        - **weight** (Parameter) - The learnable weights with shape
          :math:`(\text{out_channels}, \text{in1_channels}, \text{in2_channels})`.
          When `weight_init` is `None`, the values are initialized from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in1_channels}}`.
        - **bias** (Parameter) - The learnable bias of shape :math:`(\text{out_channels})`.
          If `has_bias` is `True` and `bias_init` is `None`, the values are initialized from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in1_channels}}`.

    Raises:
        TypeError: If `in1_channels`, `in2_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If length of shape of `weight_init` is not equal to 3 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in1_channels`
                    or shape[2] of `weight_init` is not equal to `in2_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.random.randn(128, 20), mindspore.float32)
        >>> x2 = Tensor(np.random.randn(128, 30), mindspore.float32)
        >>> net = nn.BiDense(20, 30, 40)
        >>> output = net(x1, x2)
        >>> print(output.shape)
        (128, 40)
    """

    def __init__(self,
                 in1_channels,
                 in2_channels,
                 out_channels,
                 weight_init=None,
                 bias_init=None,
                 has_bias=True):
        super().__init__()
        self.in_channels = Validator.check_positive_int(in1_channels, "in1_channels", self.cls_name)
        self.in_channels = Validator.check_positive_int(in2_channels, "in2_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)

        self.in1_channels = in1_channels
        self.in2_channels = in2_channels
        self.out_channels = out_channels
        self.has_bias = has_bias
        bound = 1 / math.sqrt(in1_channels)
        if weight_init is None:
            weight_init = Uniform(bound)
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 3 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in1_channels or weight_init.shape[2] != in2_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 3, the first dim must be equal to 'out_channels', the "
                                 f"second dim must be equal to 'in1_channels', and the third dim must be "
                                 f"equal to 'in2_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in1_channels}, "
                                 f"'in2_channels': {in2_channels}")
        self.weight = Parameter(initializer(weight_init, (out_channels, in1_channels, in2_channels)), 'weight')

        if self.has_bias:
            if bias_init is None:
                bias_init = Uniform(bound)
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' should "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()
        self.matmul = P.MatMul()

    def construct(self, input1, input2):
        _check_is_tensor("input1", input1, self.cls_name)
        _check_is_tensor("input2", input2, self.cls_name)
        input1_shape = input1.shape
        input2_shape = input2.shape
        if len(input1_shape) != 2:
            input1 = input1.reshape((-1, input1_shape[-1]))
            input2 = input2.reshape((-1, input2_shape[-1]))
        batch_size = input1.shape[0]
        output = self.matmul(input1, self.weight.transpose(1, 2, 0).view(self.in1_channels, -1))
        output = output.view(batch_size, self.in2_channels, self.out_channels)
        output = output.transpose(2, 0, 1) * input2
        output = output.sum(2).swapaxes(0, 1)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if len(input1_shape) != 2:
            out_shape = input1_shape[:-1] + (-1,)
            output = output.reshape(out_shape)
        return output

    def extend_repr(self):
        s = 'in1_channels={}, in2_channels={}, output_channels={}'.format(
            self.in1_channels, self.in2_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        return s
