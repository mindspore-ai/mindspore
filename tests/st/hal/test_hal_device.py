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
                      get_device_properties, get_device_name

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_hal_device():
    """
    Feature: Hal device api.
    Description: Test hal.device api.
    Expectation: hal.device api performs as expected.
    """
    assert not is_initialized("GPU")
    assert is_available("GPU")
    assert is_available("CPU")
    net = Net()
    net(Tensor(2.0))
    assert not is_initialized("CPU")
    assert is_initialized("GPU")
    print("Device count is", device_count())
    print("Device properties is", get_device_properties(0))
    print("Device name is", get_device_name(0))
