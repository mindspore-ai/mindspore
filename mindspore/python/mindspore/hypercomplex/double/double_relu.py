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
"""Double relu operators"""
from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.hypercomplex.utils import get_x_and_y as get_u1_and_u2, \
                                to_2channel as to_double


class J1J2ReLU(nn.Cell):
    r"""
    Rectified Linear Unit activation function for double-valued input in the diagonal representation.

    Applies ReLU activation layer for the double-valued input. This layer first converts the input to the regular form:

    .. math::
        \begin{align}
        \text{Re(inp)} = 0.5 * (\text{X(inp)} + \text{Y(inp)})\\
        \text{Db(inp)} = 0.5 * (\text{X(inp)} - \text{Y(inp)}),
        \end{align}

     then applies the element-wise :math:`\max(0, x)` for both real and double parts of the input tensor independently:

     .. math::
        \begin{align}
        \text{Re(out)} = (Re(inp))^+ = \max(0, Re(inp))\\
        \text{Db(out)} = (Db(inp))^+ = \max(0, Db(inp)),
        \end{align}

     and finally transfers the result back to the diagonal representation:

    .. math::
        \begin{align}
        \text{X(out)} = \text{Re(out)} + \text{Db(out)}\\
        \text{Y(out)} = \text{Re(out)} - \text{Db(out)}
        \end{align}

    Inputs:
        - **inp** (Tensor) - The input of ReLU is a Tensor of shape (2, *, ..., *). The data type is
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ .

    Outputs:
        Tensor, with the same type and shape as the `inp`.

    Raises:
        TypeError: If dtype of `inp` is not a number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self):
        """Initialize J1J2ReLU."""
        super(J1J2ReLU, self).__init__()

    def construct(self, u: Tensor) -> Tensor:
        u = u / 2
        u1, u2 = get_u1_and_u2(u)
        x = P.relu(u1 + u2)
        y = P.relu(u1 - u2)
        out1 = x + y
        out2 = x - y
        out = to_double(out1, out2)
        return out
