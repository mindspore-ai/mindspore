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
    def construct(self, a, norm_ord=None, dim=None, keepdim=False, dtype=None):
        output = a.norm(norm_ord, dim, keepdim, dtype=dtype)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_norm_normal(mode):
    """
    Feature: norm
    Description: Verify the result of norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = ops.arange(9, dtype=ms.float32) - 4
    b = a.reshape((3, 3))
    output1 = net(a)
    expect_output1 = np.array(7.745967)
    assert np.allclose(output1.asnumpy(), expect_output1)

    output2 = net(b)
    expect_output2 = np.array(7.745967)
    assert np.allclose(output2.asnumpy(), expect_output2)

    output3 = net(a, float('inf'))
    expect_output3 = np.array(4.0)
    assert np.allclose(output3.asnumpy(), expect_output3)

    output4 = net(b, float('inf'))
    expect_output4 = np.array(9.0)
    assert np.allclose(output4.asnumpy(), expect_output4)

    output5 = net(a, -float('inf'))
    expect_output5 = np.array(0.0)
    assert np.allclose(output5.asnumpy(), expect_output5)

    output6 = net(b, -float('inf'))
    expect_output6 = np.array(2.0)
    assert np.allclose(output6.asnumpy(), expect_output6)

    output7 = net(a, 1)
    expect_output7 = np.array(20.0)
    assert np.allclose(output7.asnumpy(), expect_output7)

    output8 = net(b, 1)
    expect_output8 = np.array(7.0)
    assert np.allclose(output8.asnumpy(), expect_output8)

    output9 = net(a, 2)
    expect_output9 = np.array(7.745967)
    assert np.allclose(output9.asnumpy(), expect_output9)

    output10 = net(b, 2)
    expect_output10 = np.array(7.3484707)
    assert np.allclose(output10.asnumpy(), expect_output10)

    output11 = net(a, -1)
    expect_output11 = np.array(0.0)
    assert np.allclose(output11.asnumpy(), expect_output11)

    output12 = net(b, -1)
    expect_output12 = np.array(6.0)
    assert np.allclose(output12.asnumpy(), expect_output12)

    output13 = net(a, -2)
    expect_output13 = np.array(0.0)
    assert np.allclose(output13.asnumpy(), expect_output13)

    output15 = net(a, 3)
    expect_output15 = np.array(5.848036)
    assert np.allclose(output15.asnumpy(), expect_output15)

    output16 = net(a, -3)
    expect_output16 = np.array(0.0)
    assert np.allclose(output16.asnumpy(), expect_output16)

    c = ms.Tensor([[1., 2., 3.], [-1, 1, 4]])
    output17 = net(c, dim=0)
    expect_output17 = np.array([1.4142135, 2.236068, 5.])
    assert np.allclose(output17.asnumpy(), expect_output17)

    output18 = net(c, dim=1)
    expect_output18 = np.array([3.7416575, 4.2426405])
    assert np.allclose(output18.asnumpy(), expect_output18)

    output19 = net(c, norm_ord=1, dim=1)
    expect_output19 = np.array([6., 6.])
    assert np.allclose(output19.asnumpy(), expect_output19)

    d = ops.arange(8, dtype=ms.float32).reshape(2, 2, 2)

    output20 = net(d, dim=(1, 2))
    expect_output20 = np.array([3.7416575, 11.224972])
    assert np.allclose(output20.asnumpy(), expect_output20)

    output21 = net(d[0, :, :]).asnumpy(), net(d[1, :, :]).asnumpy()
    expect_output21 = np.array([3.7416575, 11.224972])
    assert np.allclose(output21, expect_output21)
