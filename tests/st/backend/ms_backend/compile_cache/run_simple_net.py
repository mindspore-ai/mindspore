# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
test compile cache with simple net
"""
import numpy as np
from mindspore import context, nn, Tensor, Parameter, jit
from mindspore import dtype as mstype
from mindspore.ops import operations as P


@jit(backend="ms_backend")
class NetWithWeights(nn.Cell):
    """
    NetWithWeights
    Args:
        None
    Inputs:
        x (Tensor): Input tensor of integer type.

    Returns:
        Tensor, output tensor

    Examples:
        >>> net = NetWithWeights()
        >>> x = Tensor(np.array(1), mstype.int32)
        >>> y = Tensor(np.array(1), mstype.int32)
        >>> output = net(x, y)
    """
    def __init__(self):
        super().__init__()
        self.matmul = P.MatMul()
        self.a = Parameter(Tensor(np.array([2.0], np.float32)), name='a')
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        y = y * self.a
        out = self.matmul(x, y)
        return out


if __name__ == "__main__":
    context.set_context(jit_config={"jit_level": "O1"})
    input_x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    input_y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
    net = NetWithWeights()
    output = net(input_x, input_y)
    print("RUNTIME_COMPILE", output, "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", output.asnumpy().shape, "RUNTIME_CACHE")
