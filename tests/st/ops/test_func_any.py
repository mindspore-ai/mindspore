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
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.any(x, axis=1, keep_dims=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_any(mode):
    """
    Feature: ops.any
    Description: Verify the result of any
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[True, True],
                [False, True],
                [True, False],
                [False, False]], ms.bool_)
    net = Net()
    output = net(x)
    expect_output = Tensor([[True],
                            [True],
                            [True],
                            [False]], ms.bool_)
    assert all(output == expect_output)
