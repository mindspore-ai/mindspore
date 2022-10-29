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

""" test ops unbind """
import numpy as np
import pytest
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore import nn


class UnbindNet(nn.Cell):
    def construct(self, x, dim):
        return ops.unbind(x, dim)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_unbind(mode):
    """
    Feature: unbind
    Description: Verify the result of unbind
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    dim = 0
    unbind = UnbindNet()
    output = unbind(x, dim)
    for i in range(len(x)):
        assert np.allclose(output[i].asnumpy(), x[i].asnumpy(), rtol=0.0001)
