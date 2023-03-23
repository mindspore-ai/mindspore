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
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
import mindspore.ops as ops


class Net(nn.Cell):
    def __init__(self, kernel_size, stride=0, padding=0):
        super(Net, self).__init__()
        self.max_unpool2d = nn.MaxUnpool2d(kernel_size, stride, padding)

    def construct(self, x, indices, output_size=None):
        return self.max_unpool2d(x, indices, output_size)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_unpool2d_normal(mode):
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[6., 8.],
                          [14., 16.]]]).astype(np.float32))
    incices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    net = Net(kernel_size=2, stride=2, padding=0)
    output = net(x, incices).asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.],
                                 [0, 6., 0., 8.],
                                 [0., 0., 0., 0.],
                                 [0., 14., 0., 16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_unpool2d_normal_output_size(mode):
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[6., 8.],
                          [14., 16.]]]).astype(np.float32))
    incices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    net = Net(kernel_size=2, stride=2, padding=0)
    output_size = (1, 4, 4)
    output = net(x, incices, output_size).asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.],
                                 [0, 6., 0., 8.],
                                 [0., 0., 0., 0.],
                                 [0., 14., 0., 16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_f_max_unpool2d_normal(mode):
    """
    Feature: max_unpool2d
    Description: Verify the result of MaxUnpool2d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[6., 8.],
                          [14., 16.]]]).astype(np.float32))
    indices = Tensor(np.array([[[5, 7], [13, 15]]]).astype(np.int64))
    output = ops.max_unpool2d(x, indices, 2, stride=2, padding=0)
    output = output.asnumpy()
    expected_output = np.array([[[0., 0., 0., 0.],
                                 [0, 6., 0., 8.],
                                 [0., 0., 0., 0.],
                                 [0., 14., 0., 16.]]]).astype(np.float32)
    assert np.allclose(output, expected_output, rtol=0.0001)
