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
from tests.mark_utils import arg_mark
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


def run_basic():
    np.random.seed(0)
    tensors = []
    expect = np.array([0], np.float32)
    for _ in range(10):
        t = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
        expect = t + expect
        tensors.append(Tensor(t))

    output = get_output(*tensors).asnumpy()

    assert np.allclose(expect, output, 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_basic_gpu():
    """
    Feature: test graph kernel AddN
    Description: run test case on GPU
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_basic()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_basic_ascend():
    """
    Feature: test graph kernel AddN
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_basic()
