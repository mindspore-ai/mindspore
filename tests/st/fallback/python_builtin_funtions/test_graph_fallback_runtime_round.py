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
import mindspore as ms
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_round_cust_class():
    """
    Feature: JIT Fallback
    Description: Test round() in fallback runtime
    Expectation: No exception
    """

    class GetattrClass():
        def __init__(self):
            self.attr1 = 99.909
            self.attr2 = 1

        def method1(self, x):
            return x + self.attr2

    class GetattrClassNet(ms.nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            return round(self.cls.method1(self.cls.attr1), 2), round(100.909, 2)

    net = GetattrClassNet()
    out = net()
    assert out[0] == 100.91, out[1] == 100.91


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_round_var_tensor_set_digit():
    """
    Feature: JIT Fallback
    Description: Test round() in fallback runtime
    Expectation: No exception
    """
    @ms.jit
    def foo(x):
        return round(x, 1)

    x = Tensor([-1, -2, -3])
    with pytest.raises(TypeError) as e:
        foo(x)
    assert "When applying round() to tensor, only one tensor is supported as input" in str(e.value)
