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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

class UniformCandidateSamplerNet(nn.Cell):
    def __init__(self, num_true, num_sampled, unique, range_max):
        super(UniformCandidateSamplerNet, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true, num_sampled,
                                                 unique, range_max)

    def construct(self, x):
        return self.sampler(x)


def uniform_candidate_sampler(x, num_true, num_sampled, unique, range_max):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true,
                                                               num_sampled,
                                                               unique,
                                                               range_max)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int32)))
    return out1.shape, out2.shape, out3.shape

def uniform_candidate_sampler_int64(x, num_true, num_sampled, unique, range_max):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true,
                                                               num_sampled,
                                                               unique,
                                                               range_max)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int64)))
    return out1.shape, out2.shape, out3.shape


class UniformCandidateSamplerHitNet(nn.Cell):
    def __init__(self, num_true, num_sampled, unique, range_max, seed, remove_accidental_hits):
        super(UniformCandidateSamplerHitNet, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true, num_sampled, unique,
                                                 range_max, seed=seed,
                                                 remove_accidental_hits=remove_accidental_hits)

    def construct(self, x):
        return self.sampler(x)


def uniform_candidate_sampler_hit(x, num_true, num_sampled, unique, range_max, seed,
                                  remove_accidental_hits):
    uniform_candidate_sampler_net = UniformCandidateSamplerHitNet(num_true,
                                                                  num_sampled,
                                                                  unique,
                                                                  range_max,
                                                                  seed,
                                                                  remove_accidental_hits)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int32)))
    return out1, out2, out3


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_unique_1_true():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1], [3], [4], [6], [3]]),
                                              1, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_not_unique_1_true():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1], [3], [4], [6], [3]]),
                                              1, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_unique_2_true():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1, 2], [3, 2], [4, 2],
                                                        [6, 2], [3, 2]]),
                                              2, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_not_unique_2_true():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1, 2], [3, 2],
                                                        [4, 2], [6, 2],
                                                        [3, 2]]),
                                              2, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_large():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[12221, 41414],
                                                        [3312, 5125152],
                                                        [3312454, 51252],
                                                        [65125, 225125],
                                                        [35125, 5125122]]),
                                              2, 5, False, 100)
    expected_1 = (5,)
    expected_2 = (5, 2)
    expected_3 = (5,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_large_random():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.arange(2142).reshape(34, 63),
                                              63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_large_random_int64_input():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler_int64(np.arange(2142).reshape(34, 63),
                                                    63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_unique_1_true_hit():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, False)
    expected_1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, False)
    np.all(ms1.shape == expected_1.shape)
    np.testing.assert_array_equal(ms1.asnumpy(), expected_1.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_uniform_candidate_sampler_unique_1_true_no_hit():
    """
    Feature: UniformCandidateSampler cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, True)
    expected_1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, True)
    np.all(ms1.shape == expected_1.shape)
    np.testing.assert_array_equal(ms1.asnumpy(), expected_1.asnumpy())
