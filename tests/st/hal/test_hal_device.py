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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.hal import is_initialized, is_available, device_count,\
                      get_device_capability, get_device_properties, get_device_name, get_arch_list

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_hal_device_gpu():
    """
    Feature: Hal device api.
    Description: Test hal.device api on GPU platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    assert not is_initialized("GPU")
    assert is_available("GPU")
    assert is_available("CPU")
    assert not is_available("Ascend")
    net = Net()
    net(Tensor(2.0))
    assert not is_initialized("CPU")
    assert is_initialized("GPU")
    try:
        device_count("Ascend")
    except ValueError as e:
        assert str(e).find('not available') != -1

    dev_cnt = device_count()
    assert dev_cnt > 0
    print("Device count is", dev_cnt)
    prop = get_device_properties(dev_cnt - 1)
    print("Device properties is", prop)
    print("Device properties attributes", prop.name, prop.major, prop.minor, prop.is_multi_gpu_board,
          prop.is_integrated, prop.multi_processor_count, prop.total_memory, prop.warp_size)
    print("Device capability is", get_device_capability(dev_cnt - 1))
    print("Device name is", get_device_name(dev_cnt - 1))
    print("Arch list is", get_arch_list())
    assert not get_arch_list("CPU")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_hal_device_ascend():
    """
    Feature: Hal device api.
    Description: Test hal.device api on Ascend platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    assert not is_initialized("Ascend")
    assert is_available("Ascend")
    assert get_device_properties(0).total_memory == 0
    net = Net()
    net(Tensor(2.0))
    assert not is_initialized("CPU")
    assert is_initialized("Ascend")
    dev_cnt = device_count()
    assert dev_cnt > 0
    assert get_device_properties(dev_cnt - 1).total_memory > 0
    try:
        get_device_name(-1)
    except ValueError as e:
        assert str(e).find('negative') != -1
    assert get_arch_list() is None
    print("Device count is", dev_cnt)
    print("Device properties is", get_device_properties(dev_cnt - 1))
    print("Device name is", get_device_name(dev_cnt - 1))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_hal_device_cpu():
    """
    Feature: Hal device api.
    Description: Test hal.device api on CPU platform.
    Expectation: hal.device api performs as expected.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    assert not is_initialized("CPU")
    assert is_available("CPU")
    net = Net()
    net(Tensor(2.0))
    assert is_initialized("CPU")
    assert get_arch_list() is None


@pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
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
