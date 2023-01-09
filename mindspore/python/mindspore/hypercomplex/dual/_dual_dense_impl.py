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
"""Dual Dense Implementation"""
from typing import Callable, Tuple

from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_dense_impl import _BaseDenseImpl as BaseDenseImpl


class _DenseImpl(BaseDenseImpl):
    r"""
    The implementor class of the dense connected layer for dual numbers.

    Applies dual-valued matrix multiplication for dense connected layer. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(out)} = \text{Re(inp)} * \text{Re(kernel)}\\
        \text{Du(out)} = \text{Re(inp)} * \text{Du(kernel)} + \text{Du(inp)} * \text{Re(kernel)},
        \end{align}

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{kernel}` is
    a hypercomplex weight matrix with the same data type as the :math:`inp` created by the layer,
    :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued
    expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and dual parts of the kernel.
        **factory_kwargs (dict): Extra parameters which may be needed by specific subclasses.

    Inputs:
        - **matmul_op** (Callable) - the function of the real-valued matrix multiplication to be used for decomposition
          of the dual linear transformation. Usually, mindspore.ops.operations.MatMul(...) is passed
        - **real** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the real part of the input.
        - **dual** (Tensor) - Tensor of shape :math:`(*, in\_channels)`, which defines the dual part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(*, out\_channels)`, which represents the real and the dual
        parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self,
                  matmul_op: Callable,
                  real: Tensor,
                  dual: Tensor) -> Tuple[Tensor, Tensor]:

        out_r = matmul_op(real, self.weight_x)
        out_rd = matmul_op(real, self.weight_y)
        out_dr = matmul_op(dual, self.weight_x)

        out_d = out_rd + out_dr
        return out_r, out_d
