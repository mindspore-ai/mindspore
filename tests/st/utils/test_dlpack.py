"""Module test for dlpack"""
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

from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, mint, ops
from mindspore.utils.dlpack import from_dlpack, to_dlpack

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_npu_tensor_conversion():
    """
    Feature: test dlpack for npu tensor
    Description: test from_dlpack and to_dlpack for npu tensor
    Expectation: success
    """
    x = Tensor(np.array([1, 2, 3]), ms.float32)
    x = x.add_(1)
    x_ptr = x.data_ptr()
    dlpack_x = to_dlpack(x)
    y = from_dlpack(dlpack_x)
    y_ptr = y.data_ptr()
    assert x_ptr == y_ptr
    assert np.allclose(x.asnumpy(), y.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_empty_tensor():
    """
    Feature: test dlpack for empty tensor
    Description: test from_dlpack and to_dlpack for empty tensor
    Expectation: success
    """
    ms_tensor = Tensor(np.array([])) * 1
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert ms_tensor.asnumpy().shape == ms_tensor_from_ms_pack.asnumpy().shape
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize("dtype", [ms.uint8, ms.uint16, ms.uint32, ms.uint64])
def test_dlpack_uint(dtype):
    """
    Feature: test dlpack for different data types
    Description: test from_dlpack and to_dlpack for various data types
    Expectation: success
    """
    ms_tensor = ms.Tensor([1, 2, 3], dtype=ms.uint8)
    ms_tensor = ms_tensor + ms_tensor
    ms_tensor = ops.cast(ms_tensor, dtype)
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert np.allclose(ms_tensor.asnumpy(), ms_tensor_from_ms_pack.asnumpy())
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize("dtype", [ms.int8, ms.int16, ms.int32, ms.int64, ms.float16, ms.float32, ms.float64])
def test_dlpack_different_types(dtype):
    """
    Feature: test dlpack for different data types
    Description: test from_dlpack and to_dlpack for various data types
    Expectation: success
    """
    ms_tensor = Tensor(np.array([1, 1, 0]), dtype) * 1
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert np.allclose(ms_tensor.asnumpy(), ms_tensor_from_ms_pack.asnumpy())
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_bf16():
    """
    Feature: test dlpack for different data types
    Description: test from_dlpack and to_dlpack for various data types
    Expectation: success
    """
    ms_tensor = Tensor(np.array([1, 1, 0]), ms.bfloat16) * 1
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_scalar_tensor():
    """
    Feature: test dlpack for scalar tensor
    Description: test from_dlpack and to_dlpack for scalar tensor
    Expectation: success
    """
    ms_tensor = Tensor(1.0) * 1
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert np.allclose(ms_tensor.asnumpy(), ms_tensor_from_ms_pack.asnumpy())
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_non_contiguous_tensor():
    """
    Feature: test dlpack for non-contiguous tensor
    Description: test from_dlpack and to_dlpack for non-contiguous tensor
    Expectation: success
    """
    a = mint.ones((3, 4), dtype=ms.float16) * 1
    ms_tensor = a[1, :]
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert np.allclose(ms_tensor.asnumpy(), ms_tensor_from_ms_pack.asnumpy())
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dlpack_shared_memory():
    """
    Feature: test dlpack for shared memory
    Description: test if from_dlpack shares memory with the original tensor
    Expectation: success
    """
    a = Tensor(np.array([1, 2, 3]), ms.float32) * 1
    ms_dlpack = to_dlpack(a)
    b = from_dlpack(ms_dlpack)
    a[0] = 100
    assert np.allclose(a.asnumpy(), b.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dlpack_cpu_tensor_conversion():
    """
    Feature: test dlpack for cpu tensor
    Description: test from_dlpack and to_dlpack for cpu tensor conversion
    Expectation: success
    """
    x = Tensor(np.array([1, 2, 3]), ms.float32)
    x_ptr = x.data_ptr()
    dlpack_x = to_dlpack(x)
    y = from_dlpack(dlpack_x)
    y_ptr = y.data_ptr()
    assert x_ptr == y_ptr
    assert np.allclose(x.asnumpy(), y.asnumpy())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("dtype", [ms.int8, ms.int16, ms.int32, ms.int64, ms.float16, ms.float32, ms.float64])
def test_dlpack_cpu_different_types(dtype):
    """
    Feature: test dlpack for cpu tensor with different data types
    Description: test from_dlpack and to_dlpack for cpu tensor with various data types
    Expectation: success
    """
    ms_tensor = Tensor(np.array([1, 1, 0]), dtype)
    ms_dlpack = to_dlpack(ms_tensor)
    ms_tensor_from_ms_pack = from_dlpack(ms_dlpack)
    assert np.allclose(ms_tensor.asnumpy(), ms_tensor_from_ms_pack.asnumpy())
    assert ms_tensor.data_ptr() == ms_tensor_from_ms_pack.data_ptr()
