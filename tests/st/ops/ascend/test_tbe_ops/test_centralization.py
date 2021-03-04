# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops.operations import _inner_ops as inner

class Net(nn.Cell):
    def __init__(self, axis=()):
        super(Net, self).__init__()
        self.centralization = inner.Centralization()
        self.axis = axis

    @ms_function
    def construct(self, inputs):
        return self.centralization(inputs, self.axis)

def test_net():
    np.random.seed(1)
    x1 = np.random.randn(2, 2).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    centralization = Net(-1)
    output = centralization(Tensor(x1))
    print(x1)
    print(output.asnumpy())

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    centralization = Net(-1)
    output = centralization(Tensor(x1))
    print(x1)
    print(output.asnumpy())
