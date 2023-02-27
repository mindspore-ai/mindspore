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
"""hypercomplex dense implementation"""
import numbers
from typing import Union, Tuple

from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore import nn
from mindspore.hypercomplex.utils import get_x_and_y


class _DenseImpl(nn.Cell):
    r"""
    The interface of the implementor part of dense connected layer on second-order hypercomplex numbers.

    Defines the API for linear transformation, which is used by the 'Dense' class. The API must be implemented
    seprarately for every hypercomplex algebra:

    .. math::
        \text{out} = \text{linear}(\text{inp}, \text{kernel})

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which is provided by subclasses, :math:`\text{kernel}` is a hypercomplex weight matrix with the same data type as
    the :math:`inp` created by the layer.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and hypercomplex parts of
        the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the hypercomplex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **y** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the hypercomplex part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the hypercomplex
        part of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        super(_DenseImpl, self).__init__()

    def construct(self,
                  x: Tensor,
                  y: Tensor) -> Tuple[Tensor, Tensor]:
        pass


class _BaseDenseImpl(_DenseImpl):
    r"""
    The base implementor part of the dense connected layer for all the hypercomplex numbers of the second order.

    Contains initialization of the kernel tensors, which is shared by all specific implementations of the 'DenseImpl'
    interface for dual, double, and complex numbers.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and hypercomplex parts of
        the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the hypercomplex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **y** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the hypercomplex part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the hypercomplex
        part of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        super(_BaseDenseImpl, self).__init__(weight_init,
                                             weight_shape,
                                             **factory_kwargs)
        if isinstance(weight_init, Tensor):
            weight_init_x, weight_init_y = get_x_and_y(weight_init)
        else:
            weight_init_x = weight_init_y = weight_init
        self.weight_x = Parameter(initializer(weight_init_x, shape=weight_shape), name='weight_x')
        self.weight_y = Parameter(initializer(weight_init_y, shape=weight_shape), name='weight_y')
