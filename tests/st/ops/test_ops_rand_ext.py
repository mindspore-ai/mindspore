# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test mindspore.mint random number apis."""

import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function.random_func import rand_ext, rand_like_ext
from mindspore.ops.function.random_func import randn_ext, randn_like_ext
from mindspore.ops.function.random_func import randint_ext, randint_like_ext

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_rand(*size, dtype=None, generator=None, device=None):
    return rand_ext(*size, dtype=dtype, generator=generator, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randlike(tensor, dtype=None, device=None):
    return rand_like_ext(tensor, dtype=dtype, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randn(*size, dtype=None, generator=None, device=None):
    return randn_ext(*size, dtype=dtype, generator=generator, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randnlike(tensor, dtype=None, device=None):
    return randn_like_ext(tensor, dtype=dtype, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randint(low, high, size, dtype=None, generator=None, device=None):
    return randint_ext(low, high, size, dtype=dtype, generator=generator, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randintlike(tensor, low, high, dtype=None, device=None):
    return randint_like_ext(tensor, low, high, dtype=dtype, device=device)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand/randn/randint.
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    shape = (5, 5)
    x1 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    x2 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y1 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    y2 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z1 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()
    z2 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()

    assert np.any(x1 != x2)
    assert np.any(y1 != y2)
    assert np.any(z1 != z2)

    state = generator.get_state()
    x1 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y1 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z1 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()
    generator.set_state(state)
    x2 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y2 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z2 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()

    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randlike_randomness(mode):
    """
    Feature: randlike function.
    Description: test randomness of rand_like/randn_like/randint_like.
    Expectation: expect correct result.
    """
    tensor = ms.Tensor(np.random.randn(5, 5)).to('Ascend')
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    x2 = run_randlike(tensor, mode=mode).asnumpy()
    y1 = run_randnlike(tensor, mode=mode).asnumpy()
    y2 = run_randnlike(tensor, mode=mode).asnumpy()
    z1 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()
    z2 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()

    assert np.any(x1 != x2)
    assert np.any(y1 != y2)
    assert np.any(z1 != z2)

    state = ms.get_rng_state()
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    y1 = run_randnlike(tensor, mode=mode).asnumpy()
    z1 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()
    ms.set_rng_state(state)
    x2 = run_randlike(tensor, mode=mode).asnumpy()
    y2 = run_randnlike(tensor, mode=mode).asnumpy()
    z2 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()

    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=2048.0)
def test_rand_generator_consistency():
    """
    Feature: rand function.
    Description: test generator consistency of random op in graph and pynative mode.
    Expectation: expect correct result.
    """

    @test_utils.run_with_cell
    def randn_cell(shape, dtype=None, generator=None, device=None):
        return randn_ext(shape, dtype=dtype, generator=generator, device=device)

    ms.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    generator = ms.Generator()
    generator.manual_seed(5)

    shape = (2, 2)

    out1 = randn_ext(shape, generator=generator).asnumpy()
    state = generator.get_state()
    out2 = randn_ext(shape, generator=generator).asnumpy()
    assert not np.all(out1 == out2)
    generator.set_state(state)
    out_after_reset = randn_cell(shape, generator=generator).asnumpy()
    assert np.all(out2 == out_after_reset)
    out3 = randn_cell(shape, generator=generator).asnumpy()
    generator.set_state(state)
    randn_ext(shape, generator=generator).asnumpy()
    out3_pynative = randn_ext(shape, generator=generator).asnumpy()
    assert np.all(out3 == out3_pynative)
