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
"""Double dense implementation"""
from typing import Tuple

from mindspore import ops as P
from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_dense_impl import _BaseDenseImpl as BaseDenseImpl


class _DenseImpl(BaseDenseImpl):
    r"""
    The implementor class of the dense connected layer for double numbers in normal representation.

    Applies double-valued matrix multiplication for dense connected layer. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{X(out)} = \text{X(inp)} * \text{X(kernel)} + \text{Y(inp)} * \text{Y(kernel)}\\
        \text{Y(out)} = \text{X(inp)} * \text{Y(kernel)} + \text{Y(inp)} * \text{X(kernel)},
        \end{align}

    where :math:`inp` is the double input tensors, :math:`\text{kernel}` is a double weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a double bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{X(...)}` and
    :math:`\text{Y(...)}` are respectively the first and the second parts of the double-valued expression
    inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the complex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **real** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **double** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the imaginary part of the
          input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the imaginary
        parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  real: Tensor,
                  double: Tensor) -> Tuple[Tensor, Tensor]:
        u1 = real + double
        u2 = real - double

        out1 = P.matmul(u1, self.weight_x.transpose())
        out2 = P.matmul(u2, self.weight_y.transpose())

        out_r = out1 + out2
        out_d = out1 - out2

        return out_r, out_d


class _J1J2DenseImpl(BaseDenseImpl):
    r"""
    The implementor class of the dense connected layer for double numbers in the diagonal representation.

    Applies double-valued matrix multiplication for dense connected layer. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{X(out)} = \text{X(inp)} * \text{X(kernel)}\\
        \text{Y(out)} = \text{Y(inp)} * \text{Y(kernel)},
        \end{align}

    where :math:`inp` is the double input tensors in the diagonal form, :math:`\text{kernel}` is a double weight matrix
    in the diagonal form with the same data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is
    a double bias vector in the diagonal form with the same data type as the :math:`inp` created by the layer
    (only if has_bias is True). :math:`\text{X(...)}` and :math:`\text{Y(...)}` are respectively the first and the
    second parts of the double-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the complex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **u1** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **u2** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the imaginary
        parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  u1: Tensor,
                  u2: Tensor) -> Tuple[Tensor, Tensor]:

        out1 = P.matmul(u1, self.weight_x.transpose())
        out2 = P.matmul(u2, self.weight_y.transpose())

        return out1, out2
