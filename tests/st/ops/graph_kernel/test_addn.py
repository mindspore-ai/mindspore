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
        self.addn = P.AddN()

    def construct(self, *args):
        return self.addn(*args)


def get_output(*tensors):
    net = Net()
    output = net(tensors)
    return output


def test_basic():
    np.random.seed(0)
    tensors = []
    expect = np.array([0], np.float32)
    for _ in range(10):
        t = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
        expect = t + expect
        tensors.append(Tensor(t))

    output = get_output(*tensors).asnumpy()

    assert np.allclose(expect, output, 1.e-4, 1.e-7)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", enable_graph_kernel=True)
    test_basic()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", enable_graph_kernel=True)
    test_basic()
