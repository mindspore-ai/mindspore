# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, CSRTensor, COOTensor, ops
from mindspore import dtype as mstype

from .sparse_utils import get_platform
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_cos():
    '''
    Feature: Test sparse unary function api csr_cos.
    Description: Test ops.csr_cos.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_cos(x).values.asnumpy()
    expect = ops.cos(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_cos():
    '''
    Feature: Test sparse unary function api coo_cos.
    Description: Test ops.coo_cos.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_cos(x).values.asnumpy()
    expect = ops.cos(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_tan():
    '''
    Feature: Test sparse unary function api csr_tan.
    Description: Test ops.csr_tan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_tan(x).values.asnumpy()
    expect = ops.tan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_coo_tan():
    '''
    Feature: Test sparse unary function api coo_tan.
    Description: Test ops.coo_tan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_tan(x).values.asnumpy()
    expect = ops.tan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_exp():
    '''
    Feature: Test sparse unary function api csr_exp.
    Description: Test ops.csr_exp.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_exp(x).values.asnumpy()
    expect = ops.exp(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_exp():
    '''
    Feature: Test sparse unary function api coo_exp.
    Description: Test ops.coo_exp.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_exp(x).values.asnumpy()
    expect = ops.exp(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_inv():
    '''
    Feature: Test sparse unary function api csr_inv.
    Description: Test ops.csr_inv.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_inv(x).values.asnumpy()
    expect = ops.inv(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_inv():
    '''
    Feature: Test sparse unary function api coo_inv.
    Description: Test ops.coo_inv.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_inv(x).values.asnumpy()
    expect = ops.inv(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_relu():
    '''
    Feature: Test sparse unary function api csr_relu.
    Description: Test ops.csr_relu.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_relu(x).values.asnumpy()
    expect = ops.relu(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_relu():
    '''
    Feature: Test sparse unary function api coo_relu.
    Description: Test ops.coo_relu.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_relu(x).values.asnumpy()
    expect = ops.relu(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_expm1():
    '''
    Feature: Test sparse unary function api csr_expm1.
    Description: Test ops.csr_expm1.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_expm1(x).values.asnumpy()
    expect = ops.expm1(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_expm1():
    '''
    Feature: Test sparse unary function api coo_expm1.
    Description: Test ops.coo_expm1.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_expm1(x).values.asnumpy()
    expect = ops.expm1(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_isfinite():
    '''
    Feature: Test sparse unary function api csr_isfinite.
    Description: Test ops.csr_isfinite.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_isfinite(x).values.asnumpy()
    expect = ops.isfinite(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_isfinite():
    '''
    Feature: Test sparse unary function api coo_isfinite.
    Description: Test ops.coo_isfinite.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_isfinite(x).values.asnumpy()
    expect = ops.isfinite(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_asin():
    '''
    Feature: Test sparse unary function api csr_asin.
    Description: Test ops.csr_asin.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_asin(x).values.asnumpy()
    expect = ops.asin(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_asin():
    '''
    Feature: Test sparse unary function api coo_asin.
    Description: Test ops.coo_asin.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_asin(x).values.asnumpy()
    expect = ops.asin(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_sqrt():
    '''
    Feature: Test sparse unary function api csr_sqrt.
    Description: Test ops.csr_sqrt.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_sqrt(x).values.asnumpy()
    expect = ops.sqrt(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_sqrt():
    '''
    Feature: Test sparse unary function api coo_sqrt.
    Description: Test ops.coo_sqrt.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_sqrt(x).values.asnumpy()
    expect = ops.sqrt(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_log():
    '''
    Feature: Test sparse unary function api csr_log.
    Description: Test ops.csr_log.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_log(x).values.asnumpy()
    expect = ops.log(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_log():
    '''
    Feature: Test sparse unary function api coo_log.
    Description: Test ops.coo_log.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_log(x).values.asnumpy()
    expect = ops.log(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_isnan():
    '''
    Feature: Test sparse unary function api csr_isnan.
    Description: Test ops.csr_isnan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_isnan(x).values.asnumpy()
    expect = ops.isnan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_isnan():
    '''
    Feature: Test sparse unary function api coo_isnan.
    Description: Test ops.coo_isnan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_isnan(x).values.asnumpy()
    expect = ops.isnan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_acos():
    '''
    Feature: Test sparse unary function api csr_acos.
    Description: Test ops.csr_acos.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_acos(x).values.asnumpy()
    expect = ops.acos(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_acos():
    '''
    Feature: Test sparse unary function api coo_acos.
    Description: Test ops.coo_acos.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_acos(x).values.asnumpy()
    expect = ops.acos(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_floor():
    '''
    Feature: Test sparse unary function api csr_floor.
    Description: Test ops.csr_floor.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_floor(x).values.asnumpy()
    expect = ops.floor(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_floor():
    '''
    Feature: Test sparse unary function api coo_floor.
    Description: Test ops.coo_floor.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_floor(x).values.asnumpy()
    expect = ops.floor(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_atan():
    '''
    Feature: Test sparse unary function api csr_atan.
    Description: Test ops.csr_atan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_atan(x).values.asnumpy()
    expect = ops.atan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_atan():
    '''
    Feature: Test sparse unary function api coo_atan.
    Description: Test ops.coo_atan.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_atan(x).values.asnumpy()
    expect = ops.atan(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_square():
    '''
    Feature: Test sparse unary function api csr_square.
    Description: Test ops.csr_square.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_square(x).values.asnumpy()
    expect = ops.square(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_square():
    '''
    Feature: Test sparse unary function api coo_square.
    Description: Test ops.coo_square.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_square(x).values.asnumpy()
    expect = ops.square(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_relu6():
    '''
    Feature: Test sparse unary function api csr_relu6.
    Description: Test ops.csr_relu6.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_relu6(x).values.asnumpy()
    expect = ops.relu6(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_relu6():
    '''
    Feature: Test sparse unary function api coo_relu6.
    Description: Test ops.coo_relu6.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_relu6(x).values.asnumpy()
    expect = ops.relu6(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_sinh():
    '''
    Feature: Test sparse unary function api csr_sinh.
    Description: Test ops.csr_sinh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_sinh(x).values.asnumpy()
    expect = ops.sinh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_sinh():
    '''
    Feature: Test sparse unary function api coo_sinh.
    Description: Test ops.coo_sinh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_sinh(x).values.asnumpy()
    expect = ops.sinh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_csr_ceil():
    '''
    Feature: Test sparse unary function api csr_ceil.
    Description: Test ops.csr_ceil.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_ceil(x).values.asnumpy()
    expect = ops.ceil(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_ceil():
    '''
    Feature: Test sparse unary function api coo_ceil.
    Description: Test ops.coo_ceil.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_ceil(x).values.asnumpy()
    expect = ops.ceil(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_cosh():
    '''
    Feature: Test sparse unary function api csr_cosh.
    Description: Test ops.csr_cosh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_cosh(x).values.asnumpy()
    expect = ops.cosh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_cosh():
    '''
    Feature: Test sparse unary function api coo_cosh.
    Description: Test ops.coo_cosh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_cosh(x).values.asnumpy()
    expect = ops.cosh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_softsign():
    '''
    Feature: Test sparse unary function api csr_softsign.
    Description: Test ops.csr_softsign.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_softsign(x).values.asnumpy()
    expect = ops.softsign(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_softsign():
    '''
    Feature: Test sparse unary function api coo_softsign.
    Description: Test ops.coo_softsign.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_softsign(x).values.asnumpy()
    expect = ops.softsign(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_log1p():
    '''
    Feature: Test sparse unary function api csr_log1p.
    Description: Test ops.csr_log1p.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_log1p(x).values.asnumpy()
    expect = ops.log1p(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_log1p():
    '''
    Feature: Test sparse unary function api coo_log1p.
    Description: Test ops.coo_log1p.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_log1p(x).values.asnumpy()
    expect = ops.log1p(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_round():
    '''
    Feature: Test sparse unary function api csr_round.
    Description: Test ops.csr_round.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_round(x).values.asnumpy()
    expect = ops.round(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_round():
    '''
    Feature: Test sparse unary function api coo_round.
    Description: Test ops.coo_round.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_round(x).values.asnumpy()
    expect = ops.round(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_tanh():
    '''
    Feature: Test sparse unary function api csr_tanh.
    Description: Test ops.csr_tanh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_tanh(x).values.asnumpy()
    expect = ops.tanh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_tanh():
    '''
    Feature: Test sparse unary function api coo_tanh.
    Description: Test ops.coo_tanh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_tanh(x).values.asnumpy()
    expect = ops.tanh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_asinh():
    '''
    Feature: Test sparse unary function api csr_asinh.
    Description: Test ops.csr_asinh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_asinh(x).values.asnumpy()
    expect = ops.asinh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_asinh():
    '''
    Feature: Test sparse unary function api coo_asinh.
    Description: Test ops.coo_asinh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_asinh(x).values.asnumpy()
    expect = ops.asinh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_neg():
    '''
    Feature: Test sparse unary function api csr_neg.
    Description: Test ops.csr_neg.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_neg(x).values.asnumpy()
    expect = ops.neg(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_neg():
    '''
    Feature: Test sparse unary function api coo_neg.
    Description: Test ops.coo_neg.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_neg(x).values.asnumpy()
    expect = ops.neg(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_acosh():
    '''
    Feature: Test sparse unary function api csr_acosh.
    Description: Test ops.csr_acosh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_acosh(x).values.asnumpy()
    expect = ops.acosh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_acosh():
    '''
    Feature: Test sparse unary function api coo_acosh.
    Description: Test ops.coo_acosh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_acosh(x).values.asnumpy()
    expect = ops.acosh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_isinf():
    '''
    Feature: Test sparse unary function api csr_isinf.
    Description: Test ops.csr_isinf.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_isinf(x).values.asnumpy()
    expect = ops.isinf(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_isinf():
    '''
    Feature: Test sparse unary function api coo_isinf.
    Description: Test ops.coo_isinf.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_isinf(x).values.asnumpy()
    expect = ops.isinf(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_atanh():
    '''
    Feature: Test sparse unary function api csr_atanh.
    Description: Test ops.csr_atanh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_atanh(x).values.asnumpy()
    expect = ops.atanh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_atanh():
    '''
    Feature: Test sparse unary function api coo_atanh.
    Description: Test ops.coo_atanh.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_atanh(x).values.asnumpy()
    expect = ops.atanh(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_sigmoid():
    '''
    Feature: Test sparse unary function api csr_sigmoid.
    Description: Test ops.csr_sigmoid.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_sigmoid(x).values.asnumpy()
    expect = ops.sigmoid(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_coo_sigmoid():
    '''
    Feature: Test sparse unary function api coo_sigmoid.
    Description: Test ops.coo_sigmoid.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_sigmoid(x).values.asnumpy()
    expect = ops.sigmoid(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_abs():
    '''
    Feature: Test sparse unary function api csr_abs.
    Description: Test ops.csr_abs.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_abs(x).values.asnumpy()
    expect = ops.abs(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_abs():
    '''
    Feature: Test sparse unary function api coo_abs.
    Description: Test ops.coo_abs.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_abs(x).values.asnumpy()
    expect = ops.abs(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_csr_sin():
    '''
    Feature: Test sparse unary function api csr_sin.
    Description: Test ops.csr_sin.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indptr = Tensor([0, 1, 4, 6], dtype=mstype.int32)
    indices = Tensor([3, 0, 1, 2, 1, 3], dtype=mstype.int32)
    values = Tensor(np.arange(6) - 3.5, dtype=mstype.float32)
    shape = (3, 4)
    x = CSRTensor(indptr, indices, values, shape)

    output = ops.csr_sin(x).values.asnumpy()
    expect = ops.sin(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
def test_coo_sin():
    '''
    Feature: Test sparse unary function api coo_sin.
    Description: Test ops.coo_sin.
    Expectation: Success.
    '''
    if get_platform() != 'linux':
        return
    indices = Tensor([[0, 1], [1, 2]], dtype=mstype.int64)
    values = Tensor([-1, 2], dtype=mstype.float32)
    shape = (3, 4)
    x = COOTensor(indices, values, shape)

    output = ops.coo_sin(x).values.asnumpy()
    expect = ops.sin(x.values).asnumpy()
    assert np.allclose(output, expect, equal_nan=True)
