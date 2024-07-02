import numpy as np
import pytest

import mindspore as ms
from mindspore.ops import rand_ext, rand_like_ext

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_rand(*size, dtype=None, generator=None):
    return rand_ext(*size, dtype=dtype, generator=generator)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randlike(tensor, dtype=None):
    return rand_like_ext(tensor, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_normal(mode):
    """
    Feature: rand, rand_like function.
    Description: test function rand and rand_like
    Expectation: expect correct result.
    """
    x = run_rand(5, 5, dtype=ms.float64, mode=mode).asnumpy()
    y = run_randlike(ms.Tensor(np.random.randn(5, 5)),
                     dtype=ms.float64, mode=mode).asnumpy()
    assert np.all((x < 1) & (x >= 0))
    assert np.all((y < 1) & (y >= 0))
    assert x.dtype == np.float64
    assert y.dtype == np.float64


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    x1 = run_rand(5, 5, generator=generator, mode=mode).asnumpy()
    x2 = run_rand(5, 5, generator=generator, mode=mode).asnumpy()

    assert np.all(x1 != x2)

    state = generator.get_state()
    x1 = run_rand(5, 5, generator=generator, mode=mode).asnumpy()
    generator.set_state(state)
    x2 = run_rand(5, 5, generator=generator, mode=mode).asnumpy()

    assert np.all(x1 == x2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randlike_randomness(mode):
    """
    Feature: randlike function.
    Description: test randomness of rand_like
    Expectation: expect correct result.
    """
    tensor = ms.Tensor(np.random.randn(5, 5))
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    x2 = run_randlike(tensor, mode=mode).asnumpy()

    assert np.all(x1 != x2)

    state = ms.get_rng_state()
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    ms.set_rng_state(state)
    x2 = run_randlike(tensor, mode=mode).asnumpy()

    assert np.all(x1 == x2)
