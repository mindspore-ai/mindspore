# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

class GreaterEqualNet(nn.Cell):
    def construct(self, x, y):
        return ops.greater_equal(x, y)

def call_ge(input_tensor, other):
    """call_greater_equal"""
    out = ops.greater_equal(input_tensor, other)
    return out

def GenInputData(np_data_type, shape=(3, 4, 5)):
    """GenInputData"""
    size = 1
    for s in shape:
        size *= s
    data = np.arange(size).reshape(*shape).astype(np_data_type)
    return Tensor(data)

@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_equal_op(mode):
    """
    Feature: test greater equal op
    Description: test greater equal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([2, 1]).astype(np.float32))
    net = GreaterEqualNet()
    output = net(x, y)
    assert np.allclose(output.asnumpy(), [False, True])


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_equal_op_backward(mode):
    """
    Feature: test greater equal op
    Description: test greater equal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1, 2]).astype(np.float32))
    y = Tensor(np.array([2, 1]).astype(np.float32))
    net = GreaterEqualNet()
    grads = ops.grad(net)(x, y)
    expect_out = np.array([0., 0., 0.]).astype(np.float32)
    assert np.allclose(grads[0].asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_equal_op_vmap(mode):
    """
    Feature: test greater equal op
    Description: test greater equal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[1, 2, 3], [3, 2, 1]]).astype(np.float32))
    y = Tensor(np.array([[2, 2, 2], [2, 2, 2]]).astype(np.float32))
    net = GreaterEqualNet()
    out = ops.vmap(net, in_axes=0, out_axes=0)(x, y)
    expect_out = np.array([[False, True, True], [True, True, False]]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_equal_op_dynamic_shape(mode):
    """
    Feature: test greater equal op
    Description: test greater equal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    x = Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = Tensor(np.array([2, 4, 3]).astype(np.float32))
    net = GreaterEqualNet()
    expect_out = net(x, y)
    net.set_inputs(x_dyn, y_dyn)
    output = net(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_greater_equal_op_dynamic_rank(mode):
    """
    Feature: test greater equal op
    Description: test greater equal run by pyboost
    Expectation: expect correct forward result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.random.randn(10, 2, 20, 59, 19), ms.float32)
    y = Tensor(np.random.randn(10, 2, 20, 59, 19), ms.float32)
    x_dyn = Tensor(shape=None, dtype=x.dtype)
    y_dyn = Tensor(shape=None, dtype=y.dtype)
    net = GreaterEqualNet()
    expect_out = net(x, y)
    net.set_inputs(x_dyn, y_dyn)
    output = net(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_greater_equal_dynamic_shape():
    """
    Feature: Test greater equal with dynamic shape in graph mode.
    Description: call ops.greater_equal with valid input and index.
    Expectation: return the correct value.
    """
    ms_data1 = GenInputData(np.float32, (3, 4, 5))
    ms_data2 = GenInputData(np.float32, (3, 4, 5))

    ms_data3 = GenInputData(np.float32, (5, 5, 5, 5))
    ms_data4 = GenInputData(np.float32, (5, 5, 5, 5))
    TEST_OP(call_ge, [[ms_data1, ms_data2], [ms_data3, ms_data4]], 'greater_equal', disable_grad=True)
