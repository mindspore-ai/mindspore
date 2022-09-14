# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap


class OpNetWrapper(nn.Cell):
    """OpNetWrapper"""

    def __init__(self, op):
        """__init__"""
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        """construct"""
        return self.op(*inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_and(shape, dtype):
    """
    Feature: BitwiseAnd cpu kernel.
    Description: test the rightness of BitwiseAnd cpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseAnd()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_and(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_and(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_or(shape, dtype):
    """
    Feature: BitwiseOr cpu kernel.
    Description: test the rightness of BitwiseOr cpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseOr()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_or(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_or(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_xor(shape, dtype):
    """
    Feature: BitwiseXor cpu kernel.
    Description: test the rightness of BitwiseXor cpu kernel.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseXor()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_xor(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_xor(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('op', [P.BitwiseAnd(), P.BitwiseOr(), P.BitwiseXor()])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_vmap(op, dtype):
    """
    Feature: Bitwise cpu kernel.
    Description: test the rightness of Bitwise vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    def test_add(x, y):
        return op(x, y)

    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(dtype))
    y = Tensor(np.array([[-3, -2, -1], [3, 2, 1]]).astype(dtype))
    outputs = vmap(test_add, in_axes=(0, 1), out_axes=0)(x, y)

    x_manual = np.array([[1, 2], [3, 4], [5, 6]]).astype(dtype)
    y_manual = np.array([[-3, 3], [-2, 2], [-1, 1]]).astype(dtype)

    def manually_batched(xs, ws):
        output = []
        for i in range(xs.shape[0]):
            output.append(test_add(Tensor(xs[i]), Tensor(ws[i])).asnumpy())
        return np.stack(output)

    expect = manually_batched(x_manual, y_manual)
    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_and_tensor_interface_operator(dtype, mode, shape):
    """
    Feature: BitwiseAnd cpu kernel.
    Description: test the rightness of BitwiseAnd tensor interface.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np) & Tensor(y_np)
    expect = np.bitwise_and(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_or_tensor_interface_operator(dtype, mode, shape):
    """
    Feature: BitwiseOr cpu kernel.
    Description: test the rightness of BitwiseOr tensor interface operator.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np) | Tensor(y_np)
    expect = np.bitwise_or(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_xor_tensor_interface_operator(dtype, mode, shape):
    """
    Feature: BitwiseXor cpu kernel.
    Description: test the rightness of BitwiseXor tensor interface operator.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np) ^ Tensor(y_np)
    expect = np.bitwise_xor(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_and_tensor_interface(dtype, mode, shape):
    """
    Feature: BitwiseAnd cpu kernel.
    Description: test the rightness of BitwiseAnd tensor interface.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np).bitwise_and(Tensor(y_np))
    expect = np.bitwise_and(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_or_tensor_interface(dtype, mode, shape):
    """
    Feature: BitwiseOr cpu kernel.
    Description: test the rightness of BitwiseOr tensor interface.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np).bitwise_or(Tensor(y_np))
    expect = np.bitwise_or(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 2)])
def test_bitwise_xor_tensor_interface(dtype, mode, shape):
    """
    Feature: BitwiseXor cpu kernel.
    Description: test the rightness of BitwiseXor tensor interface.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='CPU')
    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*shape) * prop).astype(dtype)
    y_np = (np.random.randn(*shape) * prop).astype(dtype)
    outputs = Tensor(x_np).bitwise_xor(Tensor(y_np))
    expect = np.bitwise_xor(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('xshape', [(2, 3)])
@pytest.mark.parametrize('yshape', [(1, 1), (1, 3), (2, 1)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_and_broadcast(xshape, yshape, dtype):
    """
    Feature: BitwiseAnd cpu kernel.
    Description: test the rightness of BitwiseAnd cpu kernel broadcast.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseAnd()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*xshape) * prop).astype(dtype)
    y_np = (np.random.randn(*yshape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_and(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_and(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('xshape', [(2, 3)])
@pytest.mark.parametrize('yshape', [(1, 1), (1, 3), (2, 1)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_or_broadcast(xshape, yshape, dtype):
    """
    Feature: BitwiseOr cpu kernel.
    Description: test the rightness of BitwiseOr cpu kernel broadcast.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseOr()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*xshape) * prop).astype(dtype)
    y_np = (np.random.randn(*yshape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_or(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_or(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('xshape', [(2, 3)])
@pytest.mark.parametrize('yshape', [(1, 1), (1, 3), (2, 1)])
@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bitwise_xor_broadcast(xshape, yshape, dtype):
    """
    Feature: BitwiseXor cpu kernel.
    Description: test the rightness of BitwiseXor cpu kernel broadcast.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op = P.BitwiseXor()
    op_wrapper = OpNetWrapper(op)

    prop = 100 if np.random.random() > 0.5 else -100
    x_np = (np.random.randn(*xshape) * prop).astype(dtype)
    y_np = (np.random.randn(*yshape) * prop).astype(dtype)
    outputs = op_wrapper(Tensor(x_np), Tensor(y_np))
    outputs_functional = F.bitwise_xor(Tensor(x_np), Tensor(y_np))
    expect = np.bitwise_xor(x_np, y_np)

    assert np.allclose(outputs.asnumpy(), expect)
    assert np.allclose(outputs_functional.asnumpy(), expect)
