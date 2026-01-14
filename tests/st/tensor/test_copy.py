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
"""
Tests for Tensor.copy.
"""
import pytest
import hashlib
import numpy as np
import mindspore as ms
import mindspore.common.dtype as msdtype
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

@test_utils.run_with_cell
def copy_forward_func(x):
    return x.copy()

@test_utils.run_with_cell
def copy_backward_func(x):
    return ms.grad(copy_forward_func, (0))(x)

def set_mode(mode):
    if mode == 'kbk':
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    elif mode == 'ge':
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        raise ValueError(f"Unsupported mode {mode}")

def tensor_copy_limits_testcase(mode, shape, dtypes=None):
    dtype_limits_map = {
        ms.int8: (np.iinfo(np.int8).min, np.iinfo(np.int8).max),
        ms.int16: (np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        ms.int32: (np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        ms.int64: (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        ms.uint8: (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
        ms.uint16: (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
        ms.uint32: (np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
        ms.uint64: (np.iinfo(np.uint64).min, np.iinfo(np.uint64).max),
        ms.float16: (np.finfo(np.float16).min, np.finfo(np.float16).max),
        ms.float32: (np.finfo(np.float32).min, np.finfo(np.float32).max),
        ms.float64: (np.finfo(np.float64).min, np.finfo(np.float64).max),
        ms.complex64: (np.finfo(np.float32).min, np.finfo(np.float32).max),
        ms.complex128: (np.finfo(np.float64).min, np.finfo(np.float64).max),
    }

    def get_array(tensor):
        return tensor.float().asnumpy() if tensor.dtype == ms.bfloat16 else tensor.asnumpy()

    def get_md5(tensor):
        return hashlib.md5(np.ascontiguousarray(get_array(tensor)).tobytes()).hexdigest()

    def generate_tensor(shape, dtype):
        def _uniform_float64(_min, _max, shape):
            samples = np.random.random(shape).astype(np.float64)
            samples = samples * np.float64(_max) +  (1 - samples) * np.float64(_min)
            return samples

        min_, max_ = dtype_limits_map[dtype]
        if dtype == ms.float64:
            samples = _uniform_float64(min_, max_, shape)
            return ms.Tensor(samples.astype(np.float64), dtype=dtype)
        if dtype == ms.complex64:
            real = np.random.uniform(min_, max_, shape)
            imag = np.random.uniform(min_, max_, shape)
            return ms.Tensor(real + 1j * imag, dtype=dtype)
        if dtype == ms.complex128:
            real = _uniform_float64(min_, max_, shape)
            imag = _uniform_float64(min_, max_, shape)
            return ms.Tensor(real + 1j * imag, dtype=dtype)
        return ms.Tensor(np.random.uniform(min_, max_, shape).astype(msdtype.dtype_to_nptype(dtype)),
                         dtype=dtype)

    if dtypes is None:
        dtypes = msdtype.all_types

    set_mode(mode)
    all_result = True
    for dtype in dtypes:
        try:
            if dtype not in dtype_limits_map:
                print(f"skip {dtype}...")
                continue
            print(f"Tensor.copy {shape} {dtype} ...")
            src = generate_tensor(shape, dtype)
            dst = copy_forward_func(src)
            assert get_md5(dst) == get_md5(src)

            if dtype not in (ms.uint16, ms.uint32, ms.uint64):
                grad = copy_backward_func(src)
                expect_grad = ms.Tensor(np.ones(shape), dtype=dtype)
                assert get_md5(grad) == get_md5(expect_grad)
        except Exception as e:  # pylint: disable=W0703
            print(f"==================Tensor.copy {shape} {dtype} limits copy failed==================\n")
            print(e)
            all_result = False
    assert all_result


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=6.00)
@pytest.mark.parametrize("mode", ['pynative', 'kbk', 'ge'])
def test_tensor_copy_limits(mode):
    """
    Feature: Tensor interface copy limits.
    Description: Test copy forward and backward.
    Expectation: Expect correct result.
    """
    tensor_copy_limits_testcase(mode, (100, 100))


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize("mode", ['pynative', 'kbk', 'ge'])
@pytest.mark.parametrize("shape", [(0,), (2, 0)])
@pytest.mark.parametrize("dtype", [ms.int32, ms.int64, ms.uint32, ms.uint64])
def test_tensor_copy_empty_tensor(mode, shape, dtype):
    """
    Feature: Tensor interface copy empty tensor.
    Description: Test copy forward and backward.
    Expectation: Expect correct result.
    """
    set_mode(mode)

    src = ms.Tensor(np.random.randint(-10, 10, shape), dtype=dtype)
    dst = copy_forward_func(src)
    assert dst.shape == src.shape
    assert dst.dtype == src.dtype
    np.testing.assert_equal(dst.asnumpy(), src.asnumpy())

    if dtype not in (ms.uint32, ms.uint64):
        grad = copy_backward_func(src)
        expect_grad = ms.Tensor(np.ones(shape), dtype=dtype)
        assert grad.shape == expect_grad.shape
        assert grad.dtype == expect_grad.dtype
        np.testing.assert_equal(grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=24576.00)
@pytest.mark.parametrize("mode", ['pynative', 'kbk', 'ge'])
def test_tensor_copy_limits_large_size(mode):
    """
    Feature: Tensor interface copy limits with large size.
    Description: Test copy forward and backward.
    Expectation: Expect correct result.
    """
    dtypes = (ms.int64, ms.uint32)
    tensor_copy_limits_testcase(mode, (402653184,), dtypes)
