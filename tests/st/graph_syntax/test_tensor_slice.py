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
from mindspore import Tensor, context, nn
from mindspore.common import dtype as mstype


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_slice():
    """
    Feature: Test tensor slice.
    Description: Tensor getitem by a single bool value.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, input_x):
            return input_x[True] + 1.0

    x = Tensor(2.0, mstype.float32)
    net = Net()
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(x)
    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_out = net(x)
    assert graph_out == pynative_out
