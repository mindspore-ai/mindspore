# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x):
        return x.is_complex()


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_is_complex(mode):
    """
    Feature: tensor.is_complex
    Description: Verify the result of is_complex
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = ms.Tensor([complex(2, 3), complex(1, 3), complex(2.2, 3)], ms.complex64)
    b = ms.Tensor(complex(2, 3), ms.complex128)
    c = ms.Tensor([1, 2, 3], ms.float32)
    out1 = net(a)
    out2 = net(b)
    out3 = net(c)
    assert out1
    assert out2
    assert not out3
