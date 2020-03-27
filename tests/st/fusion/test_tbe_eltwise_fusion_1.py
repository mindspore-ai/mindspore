# Copyright 2020 Huawei Technologies Co., Ltd
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
import pytest
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.common.api import ms_function
import mindspore.common.dtype as mstype
import numpy as np
import mindspore.context as context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.softmax = P.Softmax(axis=1)
        self.relu = P.ReLU()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.relu(x)
        x = self.relu(x)
        return x

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    relu_relu = Net()
    output = relu_relu(Tensor(x))
    print(x)
    print(output.asnumpy())
