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
import mindspore.context as context
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_memory_stats():
    """
    Feature: Hal memory api.
    Description: Test hal.memory_stats api.
    Expectation: hal.memory_stats api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.memory_stats()
    assert not res is None
    assert isinstance(res, dict)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_memory_reserved():
    """
    Feature: Hal memory api.
    Description: Test hal.memory_reserved api.
    Expectation: hal.memory_reserved api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.memory_reserved()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_memory_allocated():
    """
    Feature: Hal memory api.
    Description: Test hal.memory_allocated api.
    Expectation: hal.memory_allocated api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.memory_allocated()
    assert not res is None
    assert isinstance(res, int)

@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_max_memory_reserved():
    """
    Feature: Hal memory api.
    Description: Test hal.max_memory_reserved api.
    Expectation: hal.max_memory_reserved api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.max_memory_reserved()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_max_memory_allocated():
    """
    Feature: Hal memory api.
    Description: Test hal.max_memory_allocated api.
    Expectation: hal.max_memory_allocated api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.max_memory_allocated()
    assert not res is None
    assert isinstance(res, int)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_memory_summary():
    """
    Feature: Hal memory api.
    Description: Test hal.memory_summar api.
    Expectation: hal.memory_summar api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    res = ms.hal.memory_summary()
    assert not res is None
    assert isinstance(res, str)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_reset_peak_memory_stats():
    """
    Feature: Hal memory api.
    Description: Test hal.reset_peak_memory_stats api.
    Expectation: hal.reset_peak_memory_stats api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    ms.hal.reset_peak_memory_stats()
    res = ms.hal.memory_stats()
    assert (res["max_reserved_memory"] == 0 and res["max_allocated_memory"] == 0)


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_reset_max_memory_reserved():
    """
    Feature: Hal memory api.
    Description: Test hal.reset_max_memory_reserved api.
    Expectation: hal.reset_max_memory_reserved api performs as expected in grad.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    ms.hal.reset_max_memory_reserved()
    res = ms.hal.max_memory_reserved()
    assert res == 0


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_hal_reset_max_memory_allocated():
    """
    Feature: Hal memory api.
    Description: Test hal.reset_max_memory_allocated api.
    Expectation: hal.reset_max_memory_allocated api performs as expected.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    net = Net()
    net(Tensor(2.0))
    ms.hal.reset_max_memory_allocated()
    res = ms.hal.max_memory_allocated()
    assert res == 0
