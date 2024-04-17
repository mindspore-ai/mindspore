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
import mindspore
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import operations as P
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_ge_dynamic_input():
    """
    Description: Test GE dynamic input.
    Description: Support dynamic inputs.
    Expectation: Run without errors.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.addn = P.AddN()

        def construct(self, *x):
            return self.addn(x)

    net = Net()
    x1 = Tensor(np.array([4, 1, 3]), mindspore.int32)
    x2 = Tensor(np.array([2, 1, 5]), mindspore.int32)
    x3 = Tensor(np.array([9, 0, 2]), mindspore.int32)
    net(x1, x2, x3, x3, x2, x1)
