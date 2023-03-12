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
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn
import mindspore.ops.function as F


class Net(nn.Cell):
    def __init__(self, batch_dims=0):
        super().__init__()
        self.batch_dims = batch_dims

    def construct(self, params, indices, axis):
        return F.gather(params, indices, axis, self.batch_dims)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_with_batch_dims(mode):
    """
    Feature: test gather op
    Description: verify the result of gather
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    x = np.arange(27).reshape(3, 3, 3).astype(np.int32)
    indices = np.array([[0, 0], [1, 1], [1, 1]]).astype(np.int32)
    axis = 1
    batch_dims = 1
    gather = Net(batch_dims)
    out = gather(Tensor(x), Tensor(indices), axis)
    expect = np.array([[[0, 1, 2], [0, 1, 2]],
                       [[12, 13, 14], [12, 13, 14]],
                       [[21, 22, 23], [21, 22, 23]]]).astype(np.int32)
    assert np.allclose(out.asnumpy(), expect)
