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
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, other):
        return x.mul(other)


class Net1(nn.Cell):
    def construct(self, x, other):
        return x * other


class Net2(nn.Cell):
    def construct(self, x, other):
        return other * x


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_mul(mode):
    """
    Feature: tensor.mul
    Description: Verify the result of tensor.mul
    Expectation: success
    """
    # test 1: tensor(number) mul tensor(number)
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    net = Net()
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 10, 18]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 2: tensor(number) mul tensor(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([True, False, True]))
    output_x = net(x, y)
    expect_x = Tensor(np.array([1, 0, 3]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 3: tensor(number) mul scalar(number)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = 4.0
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 8, 12]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 4: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = False
    output_x = net(x, y)
    expect_x = Tensor(np.array([0, 0, 0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 5: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = True
    output_x = net(x, y)
    expect_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor___mul__(mode):
    """
    Feature: tensor.__mul__
    Description: Verify the result of tensor.__mul__
    Expectation: success
    """
    # test 1: tensor(number) mul tensor(number)
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    net = Net1()
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 10, 18]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 2: tensor(number) mul tensor(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([True, False, True]))
    output_x = net(x, y)
    expect_x = Tensor(np.array([1, 0, 3]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 3: tensor(number) mul scalar(number)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = 4.0
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 8, 12]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 4: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = False
    output_x = net(x, y)
    expect_x = Tensor(np.array([0, 0, 0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 5: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = True
    output_x = net(x, y)
    expect_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor___rmul__(mode):
    """
    Feature: tensor.__mul__
    Description: Verify the result of tensor.__mul__
    Expectation: success
    """
    # test 1: tensor(number) mul tensor(number)
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([4.0, 5.0, 6.0]), ms.float32)
    net = Net2()
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 10, 18]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 2: tensor(number) mul tensor(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = Tensor(np.array([True, False, True]))
    output_x = net(x, y)
    expect_x = Tensor(np.array([1, 0, 3]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 3: tensor(number) mul scalar(number)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = 4.0
    output_x = net(x, y)
    expect_x = Tensor(np.array([4, 8, 12]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 4: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = False
    output_x = net(x, y)
    expect_x = Tensor(np.array([0, 0, 0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test 5: tensor(number) mul scalar(bool)
    x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    y = True
    output_x = net(x, y)
    expect_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
