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
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)

def get_output(x, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(x)
    return output

def test_basic(dtype):
    expect_np = np.random.normal(0, 10, (16, 32)).astype(dtype)
    x = Tensor(expect_np)
    output = get_output(x, True)
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_1():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_basic(np.float16)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_2():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_basic(np.float32)
