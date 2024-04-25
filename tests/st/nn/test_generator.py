# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

import mindspore as ms
from mindspore.nn.generator import Generator, default_generator, get_rng_state, set_rng_state


def run(generator):
    seed, offset = generator(1)
    return seed, offset


def run_twice(generator):
    seed1, offset1 = run(generator)
    seed2, offset2 = run(generator)
    return seed1, offset1, seed2, offset2


def assert_state(seed0, offset0, expected_seed, expected_offset):
    assert int(seed0) == expected_seed, f"seed problem, {seed0}, {expected_seed}"
    assert int(offset0) == expected_offset, f"offset problem, {offset0}, {expected_offset}"


def assert_state_same(seed1, offset1, seed2, offset2):
    assert int(seed1) == int(seed2)
    assert int(offset1) == int(offset2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_offset_inc(mode):
    """
    Feature: nn.Generator
    Description: Verify the offset inc
    Expectation: success
    """
    ms.set_context(mode=mode)
    generator = Generator()
    generator.manual_seed(1)
    seed, offset, seed2, offset2 = run_twice(generator)
    assert_state(seed, offset, 1, 0)
    assert_state(seed2, offset2, 1, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_restore_state(mode):
    """
    Feature: nn.Generator
    Description: Verify restore state
    Expectation: success
    """
    ms.set_context(mode=mode)
    generator1 = Generator()
    generator1.manual_seed(5)
    run_twice(generator1)
    state = generator1.get_state()

    seed1, offset1 = run(generator1)

    generator2 = Generator()
    generator2.set_state(*state)

    seed2, offset2 = run(generator2)
    assert_state_same(seed1, offset1, seed2, offset2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_default_generator(mode):
    """
    Feature: default_generator
    Description: Verify the function of default_generator
    Expectation: success
    """
    default_gen = default_generator()

    origin_seed, origin_offset = default_gen.get_state()
    seed1, offset1 = get_rng_state()

    assert_state_same(origin_seed, origin_offset, seed1, offset1)

    set_rng_state(12, 12)
    seed2, offset2 = get_rng_state()
    assert np.allclose(seed2.asnumpy(), 12)
    assert np.allclose(offset2.asnumpy(), 12)

    set_rng_state(origin_seed, origin_offset)
