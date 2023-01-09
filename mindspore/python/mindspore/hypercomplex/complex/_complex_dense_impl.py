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
"""complex dense implementation"""
from typing import Callable, Tuple

from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_dense_impl import _BaseDenseImpl as BaseDenseImpl


class _DenseImpl(BaseDenseImpl):
    r"""
    The implementor class of the dense connected layer for complex numbers.

    Applies complex-valued matrix multiplication for dense connected layer. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(out)} = \text{Re(inp)} * \text{Re(kernel)} - \text{Im(inp)} * \text{Im(kernel)}\\
        \text{Im(out)} = \text{Re(inp)} * \text{Im(kernel)} + \text{Im(inp)} * \text{Re(kernel)},
        \end{align}

    where :math:`inp` is the complex input tensors, :math:`\text{kernel}` is a complex weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{Re(...)}` and :math:`\text{Im(...)}`
    are respectively real and imaginary parts of the complex-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the complex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **real** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **imag** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the imaginary
        parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  matmul_op: Callable,
                  real: Tensor,
                  imag: Tensor) -> Tuple[Tensor, Tensor]:
        out_rr = matmul_op(real, self.weight_x)
        out_ii = matmul_op(imag, self.weight_y)
        out_ri = matmul_op(real, self.weight_y)
        out_ir = matmul_op(imag, self.weight_x)

        out_r = out_rr - out_ii
        out_i = out_ri + out_ir

        return out_r, out_i


class _KaratsubaDenseImpl(BaseDenseImpl):
    r"""
    The implementor class of the dense connected layer for complex numbers.

    Applies complex-valued matrix multiplication for dense connected layer. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{L1} = \text{Re(inp)} * \text{Re(kernel)}\\
        \text{L2} = \text{Im(inp)} * \text{Im(kernel)}\\
        \text{L3} = (\text{Re(inp)} + \text{Im(inp)}) * (\text{Re(kernel)} + \text{Im(kernel)})\\
        \text{Re(out)} = L1 - L2\\
        \text{Im(out)} = L3 - L1 - L2,
        \end{align}

    where :math:`inp` is the complex input tensors, :math:`\text{kernel}` is a complex weight matrix with the same
    data type as the :math:`inp` created by the layer, and :math:`\text{bias}` is a complex bias vector with the same
    data type as the :math:`inp` created by the layer (only if has_bias is True). :math:`\text{Re(...)}` and
    :math:`\text{Im(...)}` are respectively real and imaginary parts of the complex-valued expression inside
    the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the complex linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **real** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **imag** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the imaginary
        parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  matmul_op: Callable,
                  real: Tensor,
                  imag: Tensor) -> Tuple[Tensor, Tensor]:
        l1 = matmul_op(real, self.weight_x)
        l2 = matmul_op(imag, self.weight_y)
        l3 = matmul_op(real + imag, self.weight_x + self.weight_y)

        out_r = l1 - l2
        out_i = l3 - l1 - l2

        return out_r, out_i
