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

import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.ops import operations as P


def inplace_op_np(op, x: np.ndarray, v: np.ndarray, indices):
    result = x.copy()
    if v.shape[0] == 1:
        v = np.squeeze(v, axis=0)
    if op == 'add':
        result[indices, :] += v
    elif op == 'sub':
        result[indices, :] -= v
    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape, indice_len', [((10, 4, 3, 2), 4), ((5, 2, 4, 6), 3)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int32])
def test_inplace_add(shape, indice_len, dtype):
    """
    Feature: test InplaceAdd
    Description: test InplaceAdd
    Expectation: result is the same as expected
    """
    context.set_context(device_target='CPU')
    x = np.random.random(shape).astype(dtype)
    v = np.random.random((indice_len,) + shape[1:]).astype(dtype)
    indices = np.random.choice(list(range(shape[0])), indice_len, replace=False)
    indices = tuple((int(i) for i in indices))

    result = P.InplaceAdd(indices)(Tensor(x), Tensor(v))
    expected = inplace_op_np('add', x, v, indices)
    np.allclose(result.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape, indice', [((10, 4, 3, 2), 4), ((5, 2, 4, 6), 3)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int32])
def test_inplace_add_1d(shape, indice, dtype):
    """
    Feature: test InplaceAdd
    Description: test InplaceAdd
    Expectation: result is the same as expected
    """
    context.set_context(device_target='CPU')
    x = np.random.random(shape).astype(dtype)
    v = np.random.random((1,) + shape[1:]).astype(dtype)

    result = P.InplaceAdd(indice)(Tensor(x), Tensor(v))
    expected = inplace_op_np('add', x, v, indice)
    np.allclose(result.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape, indice_len', [((10, 4, 3, 2), 4), ((5, 2, 4, 6), 3)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int32])
def test_inplace_sub(shape, indice_len, dtype):
    """
    Feature: test InplaceSub
    Description: test InplaceSub
    Expectation: result is the same as expected
    """
    context.set_context(device_target='CPU')
    x = np.random.random(shape).astype(dtype)
    v = np.random.random((indice_len,) + shape[1:]).astype(dtype)
    indices = np.random.choice(list(range(shape[0])), indice_len, replace=False)
    indices = tuple((int(i) for i in indices))

    result = P.InplaceSub(indices)(Tensor(x), Tensor(v))
    expected = inplace_op_np('sub', x, v, indices)
    np.allclose(result.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape, indice', [((10, 4, 3, 2), 4), ((5, 2, 4, 6), 3)])
@pytest.mark.parametrize('dtype', [np.float32, np.float16, np.int32])
def test_inplace_sub_1d(shape, indice, dtype):
    """
    Feature: test InplaceAdd
    Description: test InplaceAdd
    Expectation: result is the same as expected
    """
    context.set_context(device_target='CPU')
    x = np.random.random(shape).astype(dtype)
    v = np.random.random((1,) + shape[1:]).astype(dtype)

    result = P.InplaceSub(indice)(Tensor(x), Tensor(v))
    expected = inplace_op_np('sub', x, v, indice)
    np.allclose(result.asnumpy(), expected)
