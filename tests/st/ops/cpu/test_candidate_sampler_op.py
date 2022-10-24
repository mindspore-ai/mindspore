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

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops.functional import vmap


class UniformCandidateSamplerNet(nn.Cell):
    def __init__(self, num_true, num_sampled, unique, range_max):
        super(UniformCandidateSamplerNet, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true, num_sampled, unique, range_max)

    def construct(self, x):
        return self.sampler(x)


def uniform_candidate_sampler(x, num_true, num_sampled, unique, range_max):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true, num_sampled, unique, range_max)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int32)))
    return out1.shape, out2.shape, out3.shape


def uniform_candidate_sampler_functional(x, num_true, num_sample, unique, range_max):
    out1, out2, out3 = F.uniform_candidate_sampler(Tensor(x.astype(np.int32)), num_true, num_sample, unique, range_max)
    return out1.shape, out2.shape, out3.shape


def uniform_candidate_sampler_int64(x, num_true, num_sampled, unique, range_max):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true, num_sampled, unique, range_max)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int64)))
    return out1.shape, out2.shape, out3.shape


class UniformCandidateSamplerHitNet(nn.Cell):
    def __init__(self, num_true, num_sampled, unique, range_max, seed, remove_accidental_hits):
        super(UniformCandidateSamplerHitNet, self).__init__()
        self.sampler = P.UniformCandidateSampler(num_true,
                                                 num_sampled,
                                                 unique,
                                                 range_max,
                                                 seed=seed,
                                                 remove_accidental_hits=remove_accidental_hits)

    def construct(self, x):
        return self.sampler(x)


def uniform_candidate_sampler_hit(x, num_true, num_sampled, unique, range_max, seed, remove_accidental_hits):
    uniform_candidate_sampler_net = UniformCandidateSamplerHitNet(num_true, num_sampled, unique, range_max, seed,
                                                                  remove_accidental_hits)
    out1, out2, out3 = uniform_candidate_sampler_net(Tensor(x.astype(np.int32)))
    return out1.shape, out2.shape, out3.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_unique_1_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The unique is true for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1], [3], [4], [6], [3]]), 1, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_not_unique_1_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The unique is false for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1], [3], [4], [6], [3]]), 1, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_unique_2_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The unique is true and num_true is 2 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1, 2], [3, 2], [4, 2], [6, 2], [3, 2]]), 2, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_not_unique_2_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The unique is false and num_true is 2 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.array([[1, 2], [3, 2], [4, 2], [6, 2], [3, 2]]), 2, 3, False, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_large():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The input data is large for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(
        np.array([[12221, 41414], [3312, 5125152], [3312454, 51252], [65125, 225125], [35125, 5125122]]), 2, 5, False,
        100)
    expected_1 = (5,)
    expected_2 = (5, 2)
    expected_3 = (5,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_large_random():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The input data is random large with type int32 for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler(np.arange(2142).reshape(34, 63), 63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_large_random_int64_input():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The input data is random large with type int64 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler_int64(np.arange(2142).reshape(34, 63), 63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_unique_not_hit():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The remove flag is false with seed is 1 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, False)
    expected_1 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_unique_hit():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The remove flag is true with seed is 1 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, True, 4, 1, True)
    expected_1 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_not_unique_not_hit1():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: The remove flag is true and unique is false with seed is 1 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, False, 4, 1, True)
    expected_1 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_not_unique_not_hit2():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: sample without skip with seed is 1 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 5, False, 4, 1, True)
    expected_1 = (5,)
    np.testing.assert_array_equal(ms1, expected_1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_not_unique_not_hit3():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: sample without skip with remove flag is 1 for UniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, _, _ = uniform_candidate_sampler_hit(np.array([[1]]), 1, 3, False, 4, 1, False)
    expected_1 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)


class UniformCandidateSamplerNetVmap(nn.Cell):
    def __init__(self, net, in_axes=None, out_axes=None):
        super(UniformCandidateSamplerNetVmap, self).__init__()
        self.vmap_net = vmap(net, in_axes=in_axes, out_axes=out_axes)

    def construct(self, x):
        return self.vmap_net(x)


def uniform_candidate_sampler_vmap(x, num_true, num_sampled, unique, range_max, in_axes, out_axes=0):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true, num_sampled, unique, range_max)
    uniform_candidate_sampler_vmap_net = UniformCandidateSamplerNetVmap(uniform_candidate_sampler_net,
                                                                        in_axes=in_axes,
                                                                        out_axes=out_axes)
    out1, out2, out3 = uniform_candidate_sampler_vmap_net(Tensor(x.astype(np.int32)))
    return out1.shape, out2.shape, out3.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_vmap_unique_1_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: Vmap case for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    in_axes = (0)
    ms1, ms2, ms3 = uniform_candidate_sampler_vmap(np.array([[[1], [3], [4], [6], [3]], [[1], [3], [4], [6], [3]]]), 1,
                                                   3, True, 4, in_axes)

    expected_1 = (2, 3)
    expected_2 = (2, 5, 1)
    expected_3 = (2, 3)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


class UniformCandidateSamplerNetVmap2(nn.Cell):
    def __init__(self, net, in_axes=None, out_axes=None):
        super(UniformCandidateSamplerNetVmap2, self).__init__()
        self.vmap2_net = vmap(vmap(net, in_axes=in_axes, out_axes=out_axes), in_axes=in_axes, out_axes=out_axes)

    def construct(self, x):
        return self.vmap2_net(x)


def uniform_candidate_sampler_vmap2_int64(x, num_true, num_sampled, unique, range_max, in_axes, out_axes=0):
    uniform_candidate_sampler_net = UniformCandidateSamplerNet(num_true, num_sampled, unique, range_max)
    uniform_candidate_sampler_vmap_net = UniformCandidateSamplerNetVmap2(uniform_candidate_sampler_net,
                                                                         in_axes=in_axes,
                                                                         out_axes=out_axes)
    out1, out2, out3 = uniform_candidate_sampler_vmap_net(Tensor(x.astype(np.int64)))
    return out1.shape, out2.shape, out3.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_vmap2_unique_1_true():
    """
    Feature: UniformCandidateSampler CPU TEST.
    Description: Vmap case for UniformCandidateSampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    in_axes = (1)
    ms1, ms2, ms3 = uniform_candidate_sampler_vmap2_int64(np.arange(100).reshape(5, 10, 2, 1), 1, 3, True, 4, in_axes)

    expected_1 = (10, 2, 3)
    expected_2 = (10, 2, 5, 1)
    expected_3 = (10, 2, 3)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_functional_unique_1_true():
    """
    Feature: Functional interface of UniformCandidateSampler CPU TEST.
    Description: The unique is true for uniform_candidate_sampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler_functional(np.array([[1], [3], [4], [6], [3]]), 1, 3, True, 4)
    expected_1 = (3,)
    expected_2 = (5, 1)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_functional_not_unique_2_true():
    """
    Feature: Functional interface of UniformCandidateSampler CPU TEST.
    Description: The unique is false and num_true is 2 for uniform_candidate_sampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler_functional(np.array([[1, 2], [3, 2], [4, 2], [6, 2], [3, 2]]), 2, 3,
                                                         False, 4)
    expected_1 = (3,)
    expected_2 = (5, 2)
    expected_3 = (3,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_uniform_candidate_sampler_functional_large_random():
    """
    Feature: Functional interface of UniformCandidateSampler CPU TEST.
    Description: The input data is random large with type int32 for uniform_candidate_sampler
    Expectation: The shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    ms1, ms2, ms3 = uniform_candidate_sampler_functional(np.arange(2142).reshape(34, 63), 63, 10, False, 12)
    expected_1 = (10,)
    expected_2 = (34, 63)
    expected_3 = (10,)
    np.testing.assert_array_equal(ms1, expected_1)
    np.testing.assert_array_equal(ms2, expected_2)
    np.testing.assert_array_equal(ms3, expected_3)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_log_uniform_candidate_sampler_unique():
    """
    Feature: LogUniformCandidateSampler CPU TEST.
    Description: The unique is true and num_true is 2 for LogUniformCandidateSampler
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    sampled_candidates, true_expected_count, sampled_expected_count = F.log_uniform_candidate_sampler(
        Tensor(np.array([[1, 7], [0, 4], [3, 3]]), ms.int64), 2, 5, True, 5, 1)

    expected_1 = np.array([4, 1, 2, 0, 3])
    expected_2 = np.array([[0.99236274, 0.7252593], [0.99990803, 0.8698345], [0.9201084, 0.9201084]])
    expected_3 = np.array([0.8698345, 0.99236274, 0.96404004, 0.99990803, 0.9201084])
    assert np.array_equal(sampled_candidates.asnumpy(), expected_1)
    assert np.allclose(true_expected_count.asnumpy(), expected_2)
    assert np.allclose(sampled_expected_count.asnumpy(), expected_3)
