# Copyright 2021 Huawei Technologies Co., Ltd
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

import platform
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class RCWM_count_in(nn.Cell):
    def __init__(self):
        super(RCWM_count_in, self).__init__()
        self.RCWM_count_in = P.RandomChoiceWithMask(count=4, seed=1)

    def construct(self, x):
        return self.RCWM_count_in(x)


class RCWM_count_out(nn.Cell):
    def __init__(self):
        super(RCWM_count_out, self).__init__()
        self.RCWM_count_out = P.RandomChoiceWithMask(count=10, seed=1)

    def construct(self, x):
        return self.RCWM_count_out(x)


class RCWM_3D(nn.Cell):
    def __init__(self):
        super(RCWM_3D, self).__init__()
        self.RCWM_3D = P.RandomChoiceWithMask(count=10, seed=1)

    def construct(self, x):
        return self.RCWM_3D(x)


class RCWM_1D(nn.Cell):
    def __init__(self):
        super(RCWM_1D, self).__init__()
        self.RCWM_1D = P.RandomChoiceWithMask(count=10, seed=9)

    def construct(self, x):
        return self.RCWM_1D(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_RCWM_3D():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_tensor = Tensor(np.ones([3, 4, 5]).astype(np.bool))
    expect1 = (10, 3)
    expect2 = (10,)
    rcwm = RCWM_3D()
    output1, output2 = rcwm(input_tensor)
    assert output1.shape == expect1
    assert output2.shape == expect2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_RCWM_count_out():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_tensor = Tensor(np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
                                    [0, 0, 0, 1]]).astype(np.bool))
    expect1 = (10, 2)
    expect2 = (10,)
    rcwm = RCWM_count_out()
    output1, output2 = rcwm(input_tensor)
    assert output1.shape == expect1
    assert output2.shape == expect2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_RCWM_count_in():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_tensor = Tensor(np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1],
                                    [0, 0, 0, 1]]).astype(np.bool))
    expect1 = (4, 2)
    expect2 = (4,)
    rcwm = RCWM_count_in()
    output1, output2 = rcwm(input_tensor)
    assert output1.shape == expect1
    assert output2.shape == expect2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_RCWM_1D():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_tensor = Tensor(
        np.array([1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]).astype(np.bool))
    expect_index = np.array([[11], [0], [8], [2], [9], [7],
                             [10], [15], [0], [0]]).astype(np.int32)
    expect_index_mac = np.array([[11], [7], [9], [15], [2], [10],
                                 [8], [0], [0], [0]]).astype(np.int32)
    expect_index_windows = np.array([[2], [7], [10], [15], [8], [9],
                                     [0], [11], [0], [0]]).astype(np.int32)
    expect_mask = np.array(
        [True, True, True, True, True, True, True, True, False, False])
    rcwm = RCWM_1D()
    output1, output2 = rcwm(input_tensor)
    print(output1.asnumpy())
    print(output2)
    if platform.system().lower() == "darwin":
        assert np.array_equal(output1.asnumpy(), expect_index_mac)
    elif platform.system().lower() == "windows":
        assert np.array_equal(output1.asnumpy(), expect_index_windows)
    else:
        assert np.array_equal(output1.asnumpy(), expect_index)
    assert np.array_equal(output2.asnumpy(), expect_mask)
