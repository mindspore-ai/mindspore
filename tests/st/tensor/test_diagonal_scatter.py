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


class Net(nn.Cell):
    def construct(self, x, src):
        return x.diagonal_scatter(src, offset=1, dim1=1, dim2=2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_diagonal_scatter(mode):
    """
    Feature: tensor.diagonal_scatter
    Description: Verify the result of diagonal_scatter
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]],

                   [[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]]], ms.float32)
    src = ms.Tensor([[1., 1.],
                     [1., 1.]], ms.float32)
    net = Net()
    output = net(x, src)
    expect_output = [[[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]],

                     [[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]]]
    assert np.allclose(output.asnumpy(), expect_output)
