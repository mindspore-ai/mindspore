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
import mindspore.nn as nn


class NetScatter(nn.Cell):
    def construct(self, x, axis, index, src):
        return x.scatter(axis, index, src)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_scatter(mode):
    """
    Feature: Tensor.scatter
    Description: Verify the result of scatter
    Expectation: success.
    """
    ms.set_context(mode=mode)

    scatter_ops = NetScatter()
    x = ms.Tensor(np.zeros((4, 4)), dtype=ms.float32)
    src = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
    index = ms.Tensor(np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]), dtype=ms.int64)
    axis = 1
    out = scatter_ops(x=x, axis=axis, index=index, src=src)
    expect = np.array([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]])
    assert np.allclose(out.asnumpy(), expect)
