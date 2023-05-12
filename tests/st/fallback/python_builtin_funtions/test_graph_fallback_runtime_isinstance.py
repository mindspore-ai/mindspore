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
import numpy as np
from mindspore import Tensor, context
import mindspore.nn as nn

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support type yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_isinstance():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            x_is_tensor = isinstance(x, Tensor)
            x_is_sequence = isinstance(x, (tuple, list))
            y_is_sequence = isinstance(y, (tuple, list))
            return x_is_tensor, x_is_sequence, y_is_sequence

    x = Tensor([-1, 2, 4])
    y = (1, 2)
    net = Net()
    out = net(x, y)
    assert out[0] == out[2] == True
    assert not out[1]


@pytest.mark.skip(reason="No support type yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_isinstance_numpy():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            return isinstance(x.asnumpy(), np.ndarray)

    x = Tensor(np.array([-1, 2, 4]))
    net = Net()
    out = net(x)
    assert out


@pytest.mark.skip(reason="No support type yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_isinstance_numpy_type():
    """
    Feature: JIT Fallback
    Description: Test isinstance() in fallback runtime
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            y = np.array([-1, 2, 4])
            y_type = type(y)
            return isinstance(x.asnumpy(), y_type)

    x = Tensor(np.array([-1, 2, 4]))
    net = Net()
    out = net(x)
    assert out
