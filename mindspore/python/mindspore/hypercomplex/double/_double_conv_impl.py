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
"""Double convolution implementation"""
from typing import Callable, Tuple

from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_conv_impl import _BaseConvImpl as BaseConvImpl


class _ConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for double numbers in regular representation.

    Applies double-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Db(kernel)}, \text{Db(inp)})\\
        \text{Du(ccor)} = \text{ccor}(\text{Db(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Db(inp)}),
        \end{align}

    where and :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the double input tensors, :math:`\text{kernel}` is a double weight matrix with the same
    data type as the :math:`inp` created by the layer. :math:`\text{Re(...)}` and :math:`\text{Db(...)}`
    are respectively the first and the second parts of the double-valued expression inside the parentheses in the
    regular form.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and double parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the double convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the first part of the input.
        - **double** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the second part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents both the first and the second parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  conv_fn: Callable,
                  real: Tensor,
                  double: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:

        u1 = real + double
        u2 = real - double

        out1 = conv_fn(u1, self.weight_x, pad_mode=pad_mode, padding=padding,
                       stride=stride, dilation=dilation, group=group)
        out2 = conv_fn(u2, self.weight_y, pad_mode=pad_mode, padding=padding,
                       stride=stride, dilation=dilation, group=group)

        out_r = out1 + out2
        out_d = out1 - out2
        return out_r, out_d


class _J1J2ConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for double numbers in diagonal representation.

    Applies double-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(ccor)} = \text{ccor}(\text{X(kernel)}, \text{X(inp)}) + \text{X(bias)}\\
        \text{Du(ccor)} = \text{ccor}(\text{Y(kernel)}, \text{Y(inp)}) + \text{Y(bias)},
        \end{align}

    where and :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the double input tensors, :math:`\text{kernel}` is a double weight matrix with the same
    data type as the :math:`inp` created by the layer. :math:`\text{X(...)}` and :math:`\text{Y(...)}`
    are respectively the first and the second parts of the double-valued expression inside the parentheses in the
    diagonal form.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and double parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the double convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **u1** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the first part of the input.
        - **u2** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the second part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents both the first and the second parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  conv_fn: Callable,
                  u1: Tensor,
                  u2: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:

        out1 = conv_fn(u1, self.weight_x, pad_mode=pad_mode, padding=padding,
                       stride=stride, dilation=dilation, group=group)
        out2 = conv_fn(u2, self.weight_y, pad_mode=pad_mode, padding=padding,
                       stride=stride, dilation=dilation, group=group)

        return out1, out2
