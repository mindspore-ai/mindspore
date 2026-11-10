# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Device queries retained in HAL and the device_context availability interfaces."""
import pytest
from tests.mark_utils import arg_mark

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.hal import is_initialized, device_count, get_device_properties, get_device_name

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='allcards')
def test_hal_device_gpu():
    """
    Feature: Hal device api.
    Description: Test hal.device api on GPU platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    assert not is_initialized("GPU")
    assert ms.device_context.gpu.is_available()
    assert ms.device_context.cpu.is_available()
    assert not ms.device_context.ascend.is_available()
    net = Net()
    net(Tensor(2.0))
    assert not is_initialized("CPU")
    assert is_initialized("GPU")
    try:
        ms.device_context.ascend.device_count()
    except ValueError as e:
        assert str(e).find('not available') != -1

    dev_cnt = device_count()
    assert dev_cnt > 0
    print("Device count is", dev_cnt)
    prop = get_device_properties(dev_cnt - 1)
    print("Device properties is", prop)
    print("Device properties attributes", prop.name, prop.major, prop.minor, prop.is_multi_gpu_board,
          prop.is_integrated, prop.multi_processor_count, prop.total_memory, prop.warp_size)
    print("Device name is", get_device_name(dev_cnt - 1))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_hal_device_ascend():
    """
    Feature: Hal device api.
    Description: Test hal.device api on Ascend platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', jit_level="O0")
    assert not is_initialized("Ascend")
    assert ms.device_context.ascend.is_available()
    assert get_device_properties(0).total_memory == 0
    net = Net()
    net(Tensor(2.0))
    assert not is_initialized("CPU")
    assert is_initialized("Ascend")
    dev_cnt = ms.device_context.ascend.device_count()
    assert dev_cnt > 0
    assert get_device_properties(dev_cnt - 1).total_memory > 0
    try:
        get_device_name(-1)
    except ValueError as e:
        assert str(e).find('negative') != -1
    print("Device count is", dev_cnt)
    print("Device properties is", get_device_properties(dev_cnt - 1))
    print("Device name is", get_device_name(dev_cnt - 1))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard')
def test_hal_device_cpu():
    """
    Feature: Hal device api.
    Description: Test hal.device api on CPU platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    assert ms.device_context.cpu.is_available()
    net = Net()
    net(Tensor(2.0))
    assert is_initialized("CPU")


@pytest.mark.skip("PyNative cases need to be extracted to a single file.")
def test_hal_device_pynative():
    """
    Feature: Hal device api.
    Description: Test hal.device api on in pynative mode.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(device_target='Ascend')
    assert not is_initialized("Ascend")
    t = Tensor(2.0)
    P.Abs()(t)
    assert is_initialized("Ascend")
