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
        self.matmul = P.MatMul(transpose_a=False, transpose_b=False)

    def construct(self, x, y):
        return self.matmul(x, y)

class Net1(Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.bmm = P.BatchMatMul(transpose_a=False, transpose_b=False)

    def construct(self, x, y):
        return self.bmm(x, y)

def get_output(i0, i1, net_cls, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = net_cls()
    output = net(i0, i1)
    return output

def test_matmul():
    i0 = Tensor(np.random.normal(1, 0.01, [96, 1]).astype(np.float32))
    i1 = Tensor(np.random.normal(1, 0.01, [1, 128]).astype(np.float32))
    expect = get_output(i0, i1, Net, False)
    output = get_output(i0, i1, Net, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)

def test_batchmatmul():
    i0 = Tensor(np.random.normal(1, 0.01, [16, 96, 1]).astype(np.float32))
    i1 = Tensor(np.random.normal(1, 0.01, [16, 1, 128]).astype(np.float32))
    expect = get_output(i0, i1, Net1, False)
    output = get_output(i0, i1, Net1, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 6.e-4, 6.e-4)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_matmul_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_matmul()

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batchmatmul_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_batchmatmul()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_matmul_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_matmul()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchmatmul_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_batchmatmul()
