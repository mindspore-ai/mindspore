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
import mindspore.context as context
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, padding):
        super(Net, self).__init__()
        self.pad = nn.ReflectionPad3d(padding)

    def construct(self, x):
        return self.pad(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_reflection_pad_3d(mode):
    """
    Feature: ReflectionPad3d
    Description: Infer process of ReflectionPad3d with three type parameters.
    Expectation: success
    """
    context.set_context(mode=mode)
    arr = np.arange(8).astype(np.float32).reshape((1, 2, 2, 2))
    x = Tensor(arr)
    padding = (1, 1, 1, 0, 0, 1)
    net3d = Net(padding)
    output = net3d(x)
    expected_output = Tensor(np.array([[[[3, 2, 3, 2], [1, 0, 1, 0], [3, 2, 3, 2]],
                                        [[7, 6, 7, 6], [5, 4, 5, 4], [7, 6, 7, 6]],
                                        [[3, 2, 3, 2], [1, 0, 1, 0], [3, 2, 3, 2]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)

    padding = 1
    output = Net(padding)(x)
    expected_output = Tensor(np.array([[[[7., 6., 7., 6.], [5., 4., 5., 4.],
                                         [7., 6., 7., 6.], [5., 4., 5., 4.]],
                                        [[3., 2., 3., 2.], [1., 0., 1., 0.],
                                         [3., 2., 3., 2.], [1., 0., 1., 0.]],
                                        [[7., 6., 7., 6.], [5., 4., 5., 4.],
                                         [7., 6., 7., 6.], [5., 4., 5., 4.]],
                                        [[3., 2., 3., 2.], [1., 0., 1., 0.],
                                         [3., 2., 3., 2.], [1., 0., 1., 0.]]]]).astype(np.float32))
    assert np.array_equal(output.asnumpy(), expected_output)
