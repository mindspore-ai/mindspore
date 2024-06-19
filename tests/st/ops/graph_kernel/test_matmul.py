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
from mindspore.common import dtype as mstype

class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = P.MatMul(transpose_a=True, transpose_b=True)

    def construct(self, x, y):
        return self.matmul(x, y)

class Net1(Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.matmul = P.MatMul(transpose_a=True, transpose_b=True)
        self.add = P.BiasAdd()

    def construct(self, x, y, bias):
        res = self.matmul(x, y)
        return self.add(res, bias)

def get_output(i0, i1, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(i0, i1)
    return output

def get_output1(i0, i1, i2, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net1()
    output = net(i0, i1, i2)
    return output

def test_basic():
    i0 = Tensor(np.random.normal(1, 0.01, [800, 96]).astype(np.float16))
    i1 = Tensor(np.random.normal(1, 0.01, [128, 800]).astype(np.float16))
    expect = get_output(i0, i1, False)
    output = get_output(i0, i1, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)

def test_basic1():
    i0 = Tensor(np.random.normal(1, 0.01, [800, 96]).astype(np.float16))
    i1 = Tensor(np.random.normal(1, 0.01, [128, 800]).astype(np.float16))
    i2 = Tensor(np.random.normal(100, 0.01, [128,]).astype(np.float16))
    expect = get_output1(i0, i1, i2, False)
    output = get_output1(i0, i1, i2, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 6.e-4, 6.e-4)

def basic_bfloat16():
    i0 = Tensor(np.random.normal(1, 0.01, [512, 256]).astype(np.float32), mstype.bfloat16)
    i1 = Tensor(np.random.normal(1, 0.01, [128, 512]).astype(np.float32), mstype.bfloat16)
    expect = get_output(i0, i1, False)
    output = get_output(i0, i1, True)
    expect_np = expect.float().asnumpy().copy()
    output_np = output.float().asnumpy().copy()
    assert np.allclose(expect_np, output_np, 4.e-3, 4.e-3)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_basic()

@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_basic_ascend1():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_basic1()

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_basic()

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_basic_ascend_bfloat16():
    """
    Feature: graph kernel ascend bfloat16 test
    Description: test dvm matmul bfloat16
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level='O0')
    context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    basic_bfloat16()
