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
""" test_pynative_mixed_precision_cells """
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context
from mindspore.nn import Cell
from mindspore.nn import ReLU
from mindspore.common.tensor import Tensor

class MetaFactory:
    def __init__(self):
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None

class ReluTanhSoftmax(Cell, MetaFactory):
    def __init__(self):
        super().__init__()
        MetaFactory.__init__(self)
        self.relu = ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def construct(self, x):
        x = self.relu(x)
        y = self.tanh(x)
        z = self.softmax(x)
        return x, y, z

class Add(Cell, MetaFactory):
    def __init__(self):
        super().__init__()
        MetaFactory.__init__(self)
        self.add = P.Add()

    def construct(self, x, y):
        return self.add(x, y)

class ReluTanhAdd(Cell, MetaFactory):
    def __init__(self):
        super().__init__()
        MetaFactory.__init__(self)
        self.relu = ReLU()
        self.tanh = nn.Tanh()
        self.add = Add()

    def construct(self, x):
        x_1 = self.relu(x)
        y = self.tanh(x)
        x = self.add(x_1, y)
        return x

def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me)*rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count/total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])

def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True

def mixed_precision_multiple_cells_temp_01():
    np.random.seed(1)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    net = ReluTanhSoftmax()
    net.to_float(ms.float16)
    net.relu.to_float(ms.float32)
    net.softmax.to_float(ms.float16)
    out_me_relu_01, out_me_tanh_01, out_me_softmax_01 = net(Tensor(x))
    return out_me_relu_01, out_me_tanh_01, out_me_softmax_01

def mixed_precision_multiple_cells_temp_02():
    np.random.seed(1)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    net = ReluTanhSoftmax()
    net.relu.to_float(ms.float32)
    net.softmax.to_float(ms.float16)
    net.to_float(ms.float16)
    out_me_relu_02, out_me_tanh_02, out_me_softmax_02 = net(Tensor(x))
    return out_me_relu_02, out_me_tanh_02, out_me_softmax_02

def mixed_precision_multiple_cells_temp_03():
    np.random.seed(1)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    net = ReluTanhAdd()
    net.to_float(ms.float16)
    net.relu.to_float(ms.float32)
    net.add.to_float(ms.float32)
    out_me = net(Tensor(x))
    return out_me

def mixed_precision_multiples_cell_01():
    context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
    graph_relu_01, graph_tanh_01, graph_softmax_01 = mixed_precision_multiple_cells_temp_01()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
    pynative_relu_01, pynative_tanh_01, pynative_softmax_01 = mixed_precision_multiple_cells_temp_01()

    allclose_nparray(graph_relu_01.asnumpy(), pynative_relu_01.asnumpy(), 0.001, 0.001)
    allclose_nparray(graph_tanh_01.asnumpy(), pynative_tanh_01.asnumpy(), 0.001, 0.001)
    allclose_nparray(graph_softmax_01.asnumpy(), pynative_softmax_01.asnumpy(), 0.001, 0.001)

def mixed_precision_multiples_cell_02():
    context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
    graph_relu_02, graph_tanh_02, graph_softmax_02 = mixed_precision_multiple_cells_temp_02()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
    pynative_relu_02, pynative_tanh_02, pynative_softmax_02 = mixed_precision_multiple_cells_temp_02()

    allclose_nparray(graph_relu_02.asnumpy(), pynative_relu_02.asnumpy(), 0.001, 0.001)
    allclose_nparray(graph_tanh_02.asnumpy(), pynative_tanh_02.asnumpy(), 0.001, 0.001)
    allclose_nparray(graph_softmax_02.asnumpy(), pynative_softmax_02.asnumpy(), 0.001, 0.001)

def mixed_precision_multiples_cell_03():
    context.set_context(mode=context.GRAPH_MODE, device_target=context.get_context('device_target'))
    graph_output_03 = mixed_precision_multiple_cells_temp_03()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=context.get_context('device_target'))
    pynative_output_03 = mixed_precision_multiple_cells_temp_03()

    allclose_nparray(graph_output_03.asnumpy(), pynative_output_03.asnumpy(), 0.001, 0.001)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_ascend_01():
    context.set_context(device_target="Ascend")
    mixed_precision_multiples_cell_01()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_gpu_01():
    context.set_context(device_target="GPU")
    mixed_precision_multiples_cell_01()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_ascend_02():
    context.set_context(device_target="Ascend")
    mixed_precision_multiples_cell_02()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_gpu_02():
    context.set_context(device_target="GPU")
    mixed_precision_multiples_cell_02()

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_ascend_03():
    context.set_context(device_target="Ascend")
    mixed_precision_multiples_cell_03()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mixed_precision_multiples_cell_gpu_03():
    context.set_context(device_target="GPU")
    mixed_precision_multiples_cell_03()
