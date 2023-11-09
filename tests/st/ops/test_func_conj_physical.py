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

import mindspore as ms
from mindspore import Tensor, nn, ops
import mindspore.context as context
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def construct(self, x):
        output = ops.conj_physical(x)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conj_physical(mode):
    """
    Feature: test conj_physical functional API.
    Description: test case for conj_physical functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array((1.3 + 0.4j)), mstype.complex64)
    net = Net()
    out = net(x)
    expected = np.array((1.3 - 0.4j), np.complex64)
    assert np.allclose(out.asnumpy(), expected)

    x1 = Tensor([[1, 2], [3, 4]], mstype.float32)
    net = Net()
    out1 = net(x1)
    expected = [[1, 2], [3, 4]]
    assert np.allclose(out1.asnumpy(), expected)

    x2 = Tensor([[True, False], [False, True]], mstype.bool_)
    net = Net()
    out2 = net(x2)
    expected = [[True, False], [False, True]]
    assert np.allclose(out2.asnumpy(), expected)
