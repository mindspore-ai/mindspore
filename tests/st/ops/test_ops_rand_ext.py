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
"""Test mindspore.mint random number apis"""

import os
import subprocess
import sys
import numpy as np
import pytest

import mindspore as ms
from mindspore.graph.api import _pynative_executor
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
def run_randint_default_low_overload(high, size, dtype=None, generator=None, device=None):
    return randint_ext(high, size, dtype=dtype, generator=generator, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randintlike(tensor, low, high, dtype=None, device=None):
    return randint_like_ext(tensor, low, high, dtype=dtype, device=device)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randint_like_default_low_overload(tensor, high, dtype=None, device=None):
    return randint_like_ext(tensor, high, dtype=dtype, device=device)


@arg_mark(plat_marks=['platform_ascend'],
        level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_call(mode):
    """
    Feature: rand, rand_like, randn, randn_like, randint, randint_like function.
    Description: test function call
    Expectation: expect correct result.
    """
    # rand, randlike
    shape = (5, 5)
    x = run_rand(*shape, dtype=ms.float64, mode=mode).asnumpy()
    x2 = run_rand(shape, dtype=ms.float64, mode=mode).asnumpy()
    y = run_randlike(ms.Tensor(np.random.randn(*shape)).to('Ascend'),
                     dtype=ms.float64, mode=mode).asnumpy()
    assert np.all((x < 1) & (x >= 0))
    assert np.all((x2 < 1) & (x2 >= 0))
    assert np.all((y < 1) & (y >= 0))
    assert x.dtype == np.float64
    assert x2.dtype == np.float64
    assert y.dtype == np.float64
    assert x.shape == shape
    assert x2.shape == shape
    assert y.shape == shape

    # randn, randn_like
    shape = (7, 8, 9)
    x = run_randn(*shape, dtype=ms.float64, mode=mode).asnumpy()
    x2 = run_randn(shape, dtype=ms.float64, mode=mode).asnumpy()
    y = run_randnlike(ms.Tensor(np.random.randn(*shape)).to('Ascend'),
                      dtype=ms.float64, mode=mode).asnumpy()
    assert x.dtype == np.float64
    assert x2.dtype == np.float64
    assert y.dtype == np.float64
    assert x.shape == shape
    assert x2.shape == shape
    assert y.shape == shape

    # randint, randint_like
    low = -10
    high = 10
    shape = (2, 3, 4, 5)
    dtype = ms.int32
    x = run_randint(low, high, shape, dtype=dtype, mode=mode).asnumpy()
    y = run_randintlike(ms.Tensor(np.random.randn(*shape)).to('Ascend'), low, high,
                        dtype=dtype, mode=mode).asnumpy()
    assert np.all((x >= low) & (x < high))
    assert np.all((y >= low) & (y < high))
    assert x.dtype == np.int32
    assert y.dtype == np.int32
    assert x.shape == shape
    assert y.shape == shape

    # test randint/randint_like overload with default low=0
    x = run_randint_default_low_overload(high, shape, dtype=dtype, mode=mode).asnumpy()
    assert np.all(x >= 0)
    y = run_randint_like_default_low_overload(
        ms.Tensor(x).to('Ascend'), 100, dtype=dtype, mode=mode).asnumpy()
    assert np.all(y >= 0)


@arg_mark(plat_marks=['platform_ascend'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randlike_randomness(mode):
    """
    Feature: randlike function.
    Description: test randomness of rand_like
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

@arg_mark(plat_marks=['platform_ascend'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_rand_generator_consistency():
    """
    Feature: rand function.
    Description: test generator consistency of random op in graph and pynative mode
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
    generator.set_state(state) # reset state to after out1
    out_after_reset = randn_cell(shape, generator=generator).asnumpy() # should output the same as out2
    assert np.all(out2 == out_after_reset)
    out3 = randn_cell(shape, generator=generator).asnumpy() # 3rd output from graph mode
    generator.set_state(state) # reset state to after out1
    randn_ext(shape, generator=generator).asnumpy()
    out3_pynative = randn_ext(shape, generator=generator).asnumpy() # 3rd output from pynative mode
    assert np.all(out3 == out3_pynative)



def _run_test_in_subprocess(test_name):
    """
    Run a test function in subprocess with MS_DEV_DISABLE_AUTO_H2D=1.

    Args:
        test_name: The name of the test function to run (e.g., "test_randn_with_device_impl[pynative]")
    """
    test_file = __file__

    env = os.environ.copy()
    env['MS_DEV_DISABLE_AUTO_H2D'] = '2'

    cmd = [sys.executable, '-m', 'pytest', '-s', f'{test_file}::{test_name}']

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, (
        f"Test failed with return code {result.returncode}\n" f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randn_with_device_impl(mode):
    """
    Feature: mint.randn function with device.
    Description: Implementation of test_randn_with_device.
    Expectation: expect correct result.
    """

    shape = (2, 2)

    # 1.device=Ascend
    x = run_randn(*shape, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_randn(*shape, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as error_info:
            run_randn(*shape, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceNormal op currently doesn't support CPU.
        assert 'The kernel InplaceNormal unregistered' in str(error_info.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as error_info:
            run_randn(*shape, mode=mode, device='GPU')
            _pynative_executor.sync()

    # 5.use global device target=Ascend
    ms.set_context(device_target='Ascend')
    x = run_randn(*shape, mode=mode)
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 6.use global device target=CPU
        ms.set_context(device_target='CPU')
        with pytest.raises(RuntimeError) as error_info:
            run_randn(*shape, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceNormal unregistered' in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randn_with_device(mode):
    """
    Feature: mint.randn function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_randn_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randn_like_with_device_impl(mode):
    """
    Feature: mint.randn_like function with device.
    Description: Implementation of test_randn_like_with_device.
    Expectation: expect correct result.
    """

    tensor = ms.Tensor(np.random.randn(2, 2))

    # 1.device=Ascend
    x = run_randnlike(tensor, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_randnlike(tensor, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as error_info:
            run_randnlike(tensor, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceNormal op currently doesn't support CPU.
        assert 'The kernel InplaceNormal unregistered' in str(error_info.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as error_info:
            run_randnlike(tensor, mode=mode, device='GPU')
            _pynative_executor.sync()

    # 5.use device of input: Ascend
    tensor = tensor.to('Ascend')
    x = run_randnlike(tensor, mode=mode)
    assert x.device == tensor.device

    if mode == 'pynative':
        # 6.use device of input: CPU
        tensor = tensor.to('CPU')
        with pytest.raises(RuntimeError) as error_info:
            run_randnlike(tensor, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceNormal unregistered' in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randn_like_with_device(mode):
    """
    Feature: mint.randn_like function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_randn_like_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randint_with_device_impl(mode):
    """
    Feature: mint.randint function with device.
    Description: test function call
    Expectation: expect correct result.
    """

    shape = (5, 5)
    low = -10
    high = 10
    # 1.device=Ascend
    x = run_randint(low, high, shape, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_randint(low, high, shape, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as err:
            run_randint(low, high, shape, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceRandom op currently doesn't support CPU.
        assert 'The kernel InplaceRandom unregistered' in str(err.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as err:
            run_randint(low, high, shape, mode=mode, device='GPU')
            _pynative_executor.sync()

    # 5.device=None use global device target
    ms.set_context(device_target='Ascend')
    x = run_randint(low, high, shape, mode=mode)
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 6.use global device target=CPU
        ms.set_context(device_target='CPU')
        with pytest.raises(RuntimeError) as err:
            run_randint(low, high, shape, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceRandom unregistered' in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_randint_with_device(mode):
    """
    Feature: mint.rand function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_randint_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randint_like_with_device_impl(mode):
    """
    Feature: mint.randint_like function with device.
    Description: test function call
    Expectation: expect correct result.
    """

    tensor = ms.Tensor(np.random.randn(5, 5))
    low = -10
    high = 10
    # 1.device=Ascend
    x = run_randintlike(tensor, low, high, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_randintlike(tensor, low, high, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as err:
            run_randintlike(tensor, low, high, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceRandom op currently doesn't support CPU.
        assert 'The kernel InplaceRandom unregistered' in str(err.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as err:
            run_randintlike(tensor, low, high, mode=mode, device='GPU')
        _pynative_executor.sync()

    # 5.device=None use device of input
    tensor = tensor.to('Ascend')
    x = run_randintlike(tensor, low, high, mode=mode)
    assert x.device == tensor.device

    if mode == 'pynative':
        # 6.use device of input: CPU
        tensor = tensor.to('CPU')
        with pytest.raises(RuntimeError) as err:
            run_randintlike(tensor, low, high, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceRandom unregistered' in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_randint_like_with_device(mode):
    """
    Feature: mint.randint_like function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_randint_like_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_with_device_impl(mode):
    """
    Feature: mint.rand function with device.
    Description: test function call
    Expectation: expect correct result.
    """

    shape = (5, 5)

    # 1.device=Ascend
    x = run_rand(*shape, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_rand(*shape, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as err:
            run_rand(*shape, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceUniform op currently doesn't support CPU.
        assert 'The kernel InplaceUniform unregistered' in str(err.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as err:
            run_rand(*shape, mode=mode, device='GPU')
            _pynative_executor.sync()

    # 5.device=None use global device target
    ms.set_context(device_target='Ascend')
    x = run_rand(*shape, mode=mode)
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 6.use global device target=CPU
        ms.set_context(device_target='CPU')
        with pytest.raises(RuntimeError) as err:
            run_rand(*shape, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceUniform unregistered' in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_rand_with_device(mode):
    """
    Feature: mint.rand function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_rand_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_like_with_device_impl(mode):
    """
    Feature: mint.rand_like function with device.
    Description: test function call
    Expectation: expect correct result.
    """

    tensor = ms.Tensor(np.random.randn(5, 5))

    # 1.device=Ascend
    x = run_randlike(tensor, mode=mode, device='Ascend')
    assert x.device.startswith('Ascend')

    # 2.device=npu
    x = run_randlike(tensor, mode=mode, device='npu')
    assert x.device.startswith('Ascend')

    if mode == 'pynative':
        # 3.device=CPU
        with pytest.raises(RuntimeError) as err:
            run_randlike(tensor, mode=mode, device='CPU')
            _pynative_executor.sync()
        # InplaceUniform op currently doesn't support CPU.
        assert 'The kernel InplaceUniform unregistered' in str(err.value)

        # 4.device=GPU
        with pytest.raises(ValueError) as err:
            run_randlike(tensor, mode=mode, device='GPU')
        _pynative_executor.sync()

    # 5.device=None use device of input
    tensor = tensor.to('Ascend')
    x = run_randlike(tensor, mode=mode)
    assert x.device == tensor.device

    if mode == 'pynative':
        # 6.use device of input: CPU
        tensor = tensor.to('CPU')
        with pytest.raises(RuntimeError) as err:
            run_randlike(tensor, mode=mode)
            _pynative_executor.sync()
        assert 'The kernel InplaceUniform unregistered' in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_rand_like_with_device(mode):
    """
    Feature: mint.rand_like function with device.
    Description: test function call in subprocess
    Expectation: expect correct result.
    """
    test_name = f"test_rand_like_with_device_impl[{mode}]"
    _run_test_in_subprocess(test_name)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_random_op_with_negative_dimension(mode):
    """
    Feature: mint.randn function with negative dimension.
    Description: test function call
    Expectation: expect ValueError.
    """
    with pytest.raises(ValueError) as err:
        run_randn(2, -1, mode=mode)
        if mode == 'pynative':
            _pynative_executor.sync()
    assert 'input shape must be greater than or equal to 0, but got -1' in str(err.value)

    with pytest.raises(ValueError) as err:
        run_randn(-1, mode=mode)
        if mode == 'pynative':
            _pynative_executor.sync()
    assert 'input shape must be greater than or equal to 0, but got -1' in str(err.value)

    with pytest.raises(ValueError) as err:
        run_rand(2, -1, mode=mode)
        if mode == 'pynative':
            _pynative_executor.sync()
    assert 'input shape must be greater than or equal to 0, but got -1' in str(err.value)

    with pytest.raises(ValueError) as err:
        run_randint(0, 5, (2, -1), mode=mode)
        if mode == 'pynative':
            _pynative_executor.sync()
    assert 'input shape must be greater than or equal to 0, but got -1' in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative'])
def test_randint_default_dtype(mode):
    """
    Feature: mint.randn function with default dtype.
    Description: test default mint.randn dtype is int64
    Expectation: expect correct result.
    """
    low = -3
    high = 0
    size = (0, 12)
    out = run_randint(low, high, size, mode=mode)
    assert out.dtype == ms.int64
