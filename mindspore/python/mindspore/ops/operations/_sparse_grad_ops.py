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

"""Grad operators for sparse operators."""

from __future__ import absolute_import
from mindspore.ops.primitive import Primitive, prim_attr_register


class SparseAddGrad(Primitive):
    """
    Computes gradients for sparse add operation.

    Inputs:
        - **backprop_val_grad** (Tensor) - 1D Tensor, the gradient with respect to
            the non-empty values of the sum.
        - **x1_indices** (Tensor) - 2D Tensor, the indices of the x1 in forward.
        - **x2_indices** (Tensor) - 2D Tensor, the indices of the x2 in forward.
        - **sum_indices** (Tensor) - 2d Tensor the indices of sum in forward.

    Outputs:
        - **x1_val_grad** (Tensor) - the gradient with respect to x1 in forward.
        - **x2_val_grad** (Tensor) - the gradient with respect to x2 in forward.

    Raises:
        ValueError: If (x1_indices/x2_indices/sum_indices)'s dim is not equal to 2.
        ValueError: If backprop_val_grad's dim is not equal to 1.
        ValueError: If (x1_shape/x2_shape)'s dim is not equal to 1.
        ValueError: If backprop_val_grad's length is not equal to sum_indices' length.
        TypeError: If (x1_indices/x2_indices/sum_indices)'s type is not equal to int64.
        TypeError: If backprop_val_grad's type is not equal to anf of
                   (int8/int16/int32/int64/float32/float64/complex64/complex128).

    Supported Platforms:
        ``GPU``

    Examples:
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(
            inputs=['backprop_val_grad', 'x1_indices', 'x2_indices', 'sum_indices'],
            outputs=['x1_val_grad', 'x2_val_grad'])
