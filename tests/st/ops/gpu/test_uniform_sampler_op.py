# Copyright 2020 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

class UniformSamplerNet(nn.Cell):
    def __init__(self, num_true, num_sampled, unique, range_max):
        super(UniformSamplerNet, self).__init__()
        self.sampler = P.UniformSampler(num_true, num_sampled, unique, range_max)

    def construct(self, x):
        return self.sampler(x)


def uniform_sampler(x, num_true, num_sampled, unique, range_max):
    uniform_sampler_net = UniformSamplerNet(num_true, num_sampled, unique, range_max)
    out1, out2, out3 = uniform_sampler_net(Tensor(x.astype(np.int32)))
    return out1.shape, out2.shape, out3.shape

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_unique_1_true():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.array([[1], [3], [4], [6], [3]]), 1, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_not_unique_1_true():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.array([[1], [3], [4], [6], [3]]), 1, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_unique_2_true():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.array([[1, 2], [3, 2], [4, 2], [6, 2], [3, 2]]), 2, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_not_unique_2_true():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.array([[1, 2], [3, 2], [4, 2], [6, 2], [3, 2]]), 2, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_large():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.array([[12221, 41414], [3312, 5125152], [3312454, 51252],
                                              [65125, 225125], [35125, 5125122]]), 2, 5, False, 100)
    expected_1 = (5,)
    expected_2 = (5, 2)
    expected_3 = (5,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_sampler_large_random():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms1, ms2, ms3 = uniform_sampler(np.arange(2142).reshape(34, 63), 63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)
