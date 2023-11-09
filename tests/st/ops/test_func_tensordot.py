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
from mindspore import ops
import mindspore.nn as nn
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self, axes):
        super(Net, self).__init__()
        self.axes = axes

    def construct(self, x, y):
        return ops.tensordot(x, y, self.axes)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_tensordot(mode):
    """
    Feature: test tensordot functional API.
    Description: test case for tensordot functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x1 = ms.Tensor(np.arange(0, 6).reshape(1, 2, 3), ms.float32)
    x2 = ms.Tensor(np.arange(0, 6).reshape(3, 1, 2), ms.float32)
    axes = ((0, 1), (1, 2))

    network = Net(axes)
    ms_result_np = network(x1, x2).asnumpy()
    np_result = [[3, 9, 15],
                 [4, 14, 24],
                 [5, 19, 33]]
    assert np.allclose(ms_result_np, np_result)
