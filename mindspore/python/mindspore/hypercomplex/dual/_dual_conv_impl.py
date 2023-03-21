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
"""Dual Convolution Implementation"""
import numbers
from typing import Callable, Tuple, Union

from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Initializer
from mindspore.hypercomplex.hypercomplex._hc_conv_impl import _BaseConvImpl as BaseConvImpl
from mindspore import ops as P


class _ConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for dual numbers.

    Applies dual-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})\\
        \text{Du(ccor)} = \text{ccor}(\text{Du(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Du(inp)}),
        \end{align}

    where and :math:`cccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer. :math:`\text{Re(...)}` and :math:`\text{Du(...)}`
    are respectively real and dual parts of the dual-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and dual parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the dual convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **dual** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the dual part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the dual parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  conv_fn: Callable,
                  real: Tensor,
                  dual: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:

        out_r = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        out_rd = conv_fn(real, self.weight_y, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)
        out_dr = conv_fn(dual, self.weight_x, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)

        out_d = out_rd + out_dr
        return out_r, out_d


class _ReDuConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for dual numbers.

    Applies dual-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{inp_cat} = \text{cat}(\text{Re(inp)}, \text{Du(inp)}) \\
        \text{K} = \text{cat}(\text{Du(kernel)}, \text{Re(kernel)}) \\
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})\\
        \text{Du(ccor)} = \text{ccor}(\text{K}, \text{Re(inp_cat)})
        \end{align}

    where and :math:`cccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{cat}` is concatenation along the channel axis.
    :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression
    inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and dual parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the dual convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **dual** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the dual part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the dual parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        super(_ReDuConvImpl, self).__init__(weight_init, weight_shape, **factory_kwargs)
        data_format = factory_kwargs.get('data_format')
        if data_format is None:
            data_format = "NCHW"
        self.c_idx = data_format.lower().find('c')
        if self.c_idx < 0:
            raise ValueError(f"Data format {data_format} is unsupported")

    def construct(self,
                  conv_fn: Callable,
                  real: Tensor,
                  dual: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:

        out_r = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        inp = P.concat([real, dual], axis=self.c_idx)
        w = P.concat([self.weight_y, self.weight_x], axis=self.c_idx)
        out_d = conv_fn(inp, w, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        return out_r, out_d
