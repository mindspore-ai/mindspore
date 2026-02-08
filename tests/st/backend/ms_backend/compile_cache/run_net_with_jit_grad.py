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
test compile cache with grad net in single compile cache in JIT mode.
"""
import numpy as np
from mindspore import nn, jit, Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common import Parameter
from tests.st.pynative.utils import GradOfAllParams


class ReluSquareNetWithParameter(Cell):
    """ReluSquareNetWithParameter
        Args:
            param (Tensor): Initial value for the learnable parameter.

        Inputs:
            None

        Returns:
            Tensor, output tensor after ReLU and square operations.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> param = Tensor(np.array([-1, 0, 2]), ms.int32)
            >>> net = ReluSquareNetWithParameter(param)
            >>> output = net()
            >>> print(output)
            [0 0 4]
    """
    def __init__(self, param):
        super().__init__()
        self.param = Parameter(param, name="weight")
        self.relu = nn.ReLU()
        self.square = P.Square()

    @jit
    def construct(self):
        return self.square(self.relu(self.param))


if __name__ == "__main__":
    net = ReluSquareNetWithParameter(Tensor(np.ones((3, 3), dtype=np.float32)))
    forward_out = net()
    grad_net = GradOfAllParams(net)
    grad_net.set_train()
    grad_out = grad_net(forward_out)
    print("RUNTIME_COMPILE", forward_out[0], "RUNTIME_CACHE")
    print("RUNTIME_COMPILE", forward_out[0].asnumpy().shape, "RUNTIME_CACHE")
