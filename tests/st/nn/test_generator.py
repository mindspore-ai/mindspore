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

import mindspore as ms
from mindspore.common.generator import Generator, default_generator, get_rng_state, set_rng_state
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

# pylint: disable=protected-access

@test_utils.run_with_cell
def run(generator):
    seed, offset = generator._step(ms.Tensor(1, ms.int64))
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_offset_inc(mode):
    """
    Feature: common.Generator
    Description: Verify the offset inc
    Expectation: success
    """
    ms.set_context(mode=mode)
    generator = Generator()
    generator.manual_seed(1)
    seed, offset, seed2, offset2 = run_twice(generator)
    if mode == ms.GRAPH_MODE:
        assert_state(seed, offset, 1, 0)
        assert_state(seed2, offset2, 1, 1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_restore_state(mode):
    """
    Feature: common.Generator
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
    generator2.set_state(state)

    seed2, offset2 = run(generator2)
    assert seed1 == 5
    assert_state_same(seed1, offset1, seed2, offset2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_default_generator():
    """
    Feature: default_generator
    Description: Verify the function of default_generator
    Expectation: success
    """
    state = get_rng_state()
    seed1, offset1 = default_generator._step(ms.Tensor(5, ms.int64))
    set_rng_state(state)
    seed2, offset2 = default_generator._step(ms.Tensor(5, ms.int64))
    assert_state_same(seed1, offset1, seed2, offset2)
