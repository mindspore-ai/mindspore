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
"""Hypercomplex Convolution Implementation"""
import numbers
from typing import Callable, Union, Tuple

from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import nn
from mindspore.hypercomplex.utils import get_x_and_y


class _ConvImpl(nn.Cell):
    r"""
    The interface of the implementor part of convolution layer on second-order hypercomplex numbers.

    Defines the API for unbiased convolution transformation, which is used by the '_ConvNd' class. The API must
    be implemented separately for every hypercomplex algebra:

    .. math::
       \text{out} = \text{conv}(\text{inp}, \text{kernel})

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{conv}` is the convolution transformation
    operation, which is provided by subclasses, :math:`\text{kernel}` is a hypercomplex weight matrix with the same
    data type as the :math:`inp` created by the layer.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and hypercomplex parts of
            the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used with decomposition
          of the hypercomplex convolution transformation. For example, mindspore.ops.operations.Conv2D(...)
          may be passed for a 2D convolution.
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input. The exact shape depends on data format and the number of spatial
          dimensions.
        - **y** (Tensor)  - Tensor of the same shape as `x`, which defines the real part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`, which
        represent the real and the hypercomplex parts of the output respectively. Data format and the count of spatial
        dimensions are the same as in `x` and `y`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        super(_ConvImpl, self).__init__()

    def construct(self,
                  conv_op: Callable,
                  x: Tensor,
                  y: Tensor) -> Tuple[Tensor, Tensor]:
        pass


class _BaseConvImpl(_ConvImpl):
    r"""
    The base implementor part of the convolution layer for all the hypercomplex numbers of the second order.

    Contains initialization of the kernel tensors, which is shared by all specific implementations of the 'ConvImpl'
    interface for dual, double, and complex numbers.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and hypercomplex parts of
            the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used with decomposition
          of the hypercomplex convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be
          passed for a 2D convolution.
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input. The exact shape depends on data format and the number of spatial
          dimensions.
        - **y** (Tensor)  - Tensor of the same shape as `x`, which defines the real part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`, which
        represent the real and the hypercomplex parts of the output respectively. Data format and the count of spatial
        dimensions are the same as in `x` and `y`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        super(_BaseConvImpl, self).__init__(weight_init,
                                            weight_shape,
                                            **factory_kwargs)

        if isinstance(weight_init, Tensor):
            weight_init_x, weight_init_y = get_x_and_y(weight_init)
        else:
            weight_init_x = weight_init_y = weight_init
        self.weight_x = Parameter(initializer(weight_init_x, shape=weight_shape), name='weight_x')
        self.weight_y = Parameter(initializer(weight_init_y, shape=weight_shape), name='weight_y')
