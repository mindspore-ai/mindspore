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
"""complex ReLU implementation"""
import mindspore
from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.hypercomplex.utils import get_x_and_y as get_real_and_imag, to_2channel as to_complex


class ReLU(nn.Cell):
    r"""
    Rectified Linear Unit activation function for complex-valued input.

    Applies ReLU activation layer for the complex-valued input. This layer applies the element-wise
    :math:`\max(0, x)` for both real and imaginary parts of the input tensor independently:

     .. math::
        \begin{align}
        \text{Re(out)} = (Re(inp))^+ = \max(0, Re(inp))\\
        \text{Im(out)} = (Im(inp))^+ = \max(0, Im(inp)),
        \end{align}

    Inputs:
        - **inp** (Tensor) - The input of ReLU is a Tensor of shape (2, *, ..., *), with float16 or float32 data type,
          or (*, ..., *), with complex64 data type.

    Outputs:
        Tensor, with the same data type and shape as the `inp`.

    Raises:
        TypeError: If dtype of `inp` is not float16, float32, or complex64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        """Initialize ReLU."""
        super(ReLU, self).__init__()

    def construct(self, u: Tensor) -> Tensor:
        if u.dtype == mindspore.complex64:
            real, imag = get_real_and_imag(u)
            real = P.relu(real)
            imag = P.relu(imag)
            out = to_complex(real, imag, u.dtype)
        else:
            out = P.relu(u)
        return out
