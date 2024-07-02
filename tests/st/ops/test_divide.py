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
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y, rounding_mode):
    if rounding_mode == 'floor':
        return np.floor_divide(x, y)
    if rounding_mode == 'trunc':
        return np.trunc(np.divide(x, y))
    return np.divide(x, y)


class NetNone(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.div = ms.ops.div

    def construct(self, x, y):
        return self.div(x, y)


class NetFloor(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.div = ms.ops.div

    def construct(self, x, y):
        return self.div(x, y, rounding_mode="floor")


class NetTrunc(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.div = ms.ops.div

    def construct(self, x, y):
        return self.div(x, y, rounding_mode="trunc")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function div vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array([7, 8, 9], dtype=np.float32)
    y = np.array([14, 6, 12], dtype=np.float32)
    output = ms.ops.vmap(ms.ops.div, in_axes=-1, out_axes=0)(ms.Tensor(x), ms.Tensor(y))
    expect = generate_expect_forward_output(x, y, None)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('rounding_mode', [None, 'floor', 'trunc'])
def test_ops_div_std(mode, rounding_mode):
    """
    Feature: pyboost function.
    Description: test function div forward/backward.
    Expectation: expect correct result.
    """
    # forward test
    ms.context.set_context(mode=mode)
    x, y = generate_random_input((4, 5, 6), np.float32)
    if rounding_mode == 'floor':
        net = NetFloor()
    elif rounding_mode == 'trunc':
        net = NetTrunc()
    else:
        net = NetNone()
    output = net(ms.Tensor(x, dtype=ms.float32), ms.Tensor(y, dtype=ms.float32))
    expect = generate_expect_forward_output(x, y, rounding_mode)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    # backward test
    x, y = np.array([1.0, 5.0, 7.5]), np.array([4.0, 2.0, 3.0])
    net = NetNone()
    output = ms.ops.grad(net, (0,))(ms.Tensor(x, dtype=ms.float32), ms.Tensor(y, dtype=ms.float32))
    expect = [0.25, 0.5, 0.33333333]
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_div_forward_case01(mode):
    """
    Feature: pyboost function.
    Description: test function div.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.random.randn(64, 32, 3578).astype(np.float32)
    y = np.random.randn(64, 32, 1).astype(np.float32)
    rounding_mode = None
    net = NetNone()
    output = net(ms.Tensor(x, dtype=ms.float32), ms.Tensor(y, dtype=ms.float32))
    expect = generate_expect_forward_output(x, y, rounding_mode)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_div_forward_case02(mode):
    """
    Feature: pyboost function.
    Description: test function div.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.random.randn(64, 32, 1).astype(np.float32)
    y = 7168
    rounding_mode = None
    net = NetNone()
    output = net(ms.Tensor(x, dtype=ms.float32), y)
    expect = generate_expect_forward_output(x, y, rounding_mode)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@test_utils.run_with_cell
def div_forward_dyn(x, y):
    return ms.ops.div(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_div_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    ms_x0, ms_y0 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32), ms.Tensor(np.array([[1, 2, 3, 4]]),
                                                                                            ms.float32)
    ms_x1, ms_y1 = ms.Tensor(np.array([[1, 2, 3], [5, 6, 7]]), ms.float32), ms.Tensor(np.array([[1, 2, 3]]), ms.float32)
    TEST_OP(div_forward_dyn, [[ms_x0, ms_y0], [ms_x1, ms_y1]], '', disable_input_check=True, disable_yaml_check=True)
