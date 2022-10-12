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
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self, kernel_size, stride=0, padding=0, output_size=()):
        super(Net, self).__init__()
        self.max_unpool3d = nn.MaxUnpool3d(kernel_size, stride, padding, output_size)

    def construct(self, x, indices):
        return self.max_unpool3d(x, indices)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_max_unpool3d_normal(mode):
    """
    Feature: max_unpool3d
    Description: Verify the result of MaxUnpool3d
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[[7.]]]], [[[[15.]]]]]), mindspore.float32)
    incices = Tensor(np.array([[[[[7]]]], [[[[7]]]]]), mindspore.int64)
    net = Net(kernel_size=2, stride=1, padding=0)
    output = net(x, incices).asnumpy()
    expect = np.array([[[[[0., 0.],
                          [0., 0.]],
                         [[0., 0.],
                          [0., 7.]]]],
                       [[[[0., 0.],
                          [0., 0.]],
                         [[0., 0.],
                          [0., 15.]]]]]).astype(np.float32)
    assert np.allclose(output, expect, rtol=0.0001)
