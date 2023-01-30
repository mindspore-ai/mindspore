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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x, y):
        output = ops.inner(x, y)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_normal(mode):
    """
    Feature: ops.inner
    Description: Verify the result of ops.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], ms.float32)
    y = ms.Tensor([[2, 3, 4], [4, 3, 2]], ms.float32)
    out = net(x, y)
    expect_out = np.array([[[20, 16], [16, 20]], [[47, 43], [47, 43]]], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_with_scalar(mode):
    """
    Feature: ops.inner
    Description: Verify the result of ops.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], ms.float32)
    y = ms.Tensor(2, ms.float32)
    out = net(x, y)
    expect_out = np.array([[[2, 4, 6], [6, 4, 2]], [[8, 10, 12], [8, 10, 12]]], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_1d(mode):
    """
    Feature: ops.inner
    Description: Verify the result of ops.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([1, 2, 3], ms.float32)
    y = ms.Tensor([4, 5, 6], ms.float32)
    out = net(x, y)
    expect_out = np.array(32, dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inner_multidims(mode):
    """
    Feature: ops.inner
    Description: Verify the result of ops.inner
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[0.8173, 1.0874, 1.1784], [0.3279, 0.1234, 2.7894]], dtype=ms.float32)
    y = ms.Tensor([[[-0.4682, -0.7159, 0.1506], [0.4034, -0.3657, 1.0387],
                    [0.9892, -0.6684, 0.1774], [0.9482, 1.3261, 0.3917]],
                   [[0.4537, 0.7493, 1.1724], [0.2291, 0.5749, -0.2267],
                    [-0.7920, 0.3607, -0.3701], [1.3666, -0.5850, -1.7242]]], dtype=ms.float32)
    out = net(x, y)
    expect_out = np.array([[[-0.9836624, 1.1560408, 0.29070318, 2.6785443],
                            [2.567154, 0.54524636, -0.6912023, -1.5510042]],
                           [[0.17821884, 2.9844973, 0.7367177, 1.5671635],
                            [3.5115247, -0.48629245, -1.2475433, -4.433564]]], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect_out)
