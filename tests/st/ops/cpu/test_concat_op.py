# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class ConcatConstFold(nn.Cell):
    def __init__(self):
        super(ConcatConstFold, self).__init__()
        self.cat = P.Concat(axis=0)
        self.mul = P.Mul()
        self.input_x1 = Tensor(np.array([[0, 1, 2, 1], [1, 1, 3, 5]]).astype(np.float32))
        self.input_x2 = Tensor(np.array([[4, 6, 2, 2], [0, 6, 2, 6]]).astype(np.float32))

    def construct(self, input_y):
        x = self.cat((self.input_x1, self.input_x2))
        x = self.mul(x, input_y)
        x = 2*x
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_constant_folding_float32():
    """
    Feature: concat testcase about constant fold
    Description: all inputs of concat is constant.
    Expectation: success
    """
    mul_x1 = Tensor(np.array([[0, 0, 1, 1],
                              [1, 2, 3, 1],
                              [2, 4, 5, 5],
                              [3, 6, 7, 6]]).astype(np.float32))
    cat0 = ConcatConstFold()
    output = cat0(mul_x1)
    expect = np.array([[0, 0, 4, 2],
                       [2, 4, 18, 10],
                       [16, 48, 20, 20],
                       [0, 72, 28, 72]]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


class ConcatV10(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV10, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Tensor(np.array([[[0., 0., 1.],
                                    [1., 2., 3.]],
                                   [[2., 4., 5.],
                                    [3., 6., 7.]]]).astype(nptype))

    def construct(self):
        return self.cat((self.x1,))


def axis10(nptype):
    cat = ConcatV10(nptype)
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(nptype)
    print(output)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_float32():
    """
    Feature: concat with one input
    Description: Concat with one input of float32 dtype
    Expectation: success
    """
    axis10(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_int32():
    """
    Feature: concat with one input
    Description: Concat with one input of int32 dtype
    Expectation: success
    """
    axis10(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis10_bool():
    """
    Feature: concat with one input
    Description: Concat with one input of bool dtype
    Expectation: success
    """
    axis10(np.bool)


class ConcatV32(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV32, self).__init__()

        self.cat = P.Concat(axis=2)
        self.x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis32(nptype):
    cat = ConcatV32(nptype)
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(nptype)
    print(output)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_float32():
    """
    Feature: concat in axis-2
    Description: Concat in axis 2 and float32 dtype inputs
    Expectation: success
    """
    axis32(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_int32():
    """
    Feature: concat in axis-2
    Description: Concat in axis 2 and int32 dtype inputs
    Expectation: success
    """
    axis32(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis32_bool():
    """
    Feature: concat in axis-2
    Description: Concat in axis 2 and bool dtype inputs
    Expectation: success
    """
    axis32(np.bool)


class ConcatWithList(nn.Cell):
    def __init__(self):
        super(ConcatWithList, self).__init__()
        self.concat = P.Concat(axis=2)

    def construct(self, x, y):
        input_list = [x, y]
        return self.concat(input_list)


class ConcatWithTuple(nn.Cell):
    def __init__(self):
        super(ConcatWithTuple, self).__init__()
        self.concat = P.Concat(axis=2)

    def construct(self, x, y):
        input_list = (x, y)
        return self.concat(input_list)


class GradConcat(nn.Cell):
    def __init__(self, network):
        super(GradConcat, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network

    def construct(self, x, y):
        gout = self.grad(self.network)(x, y)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_list_grad():
    """
    Feature: concat grad
    Description: ConcatGrad with list input
    Expectation: success
    """
    x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(np.float32))
    x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(np.float32))
    concat = ConcatWithList()
    output = GradConcat(concat)(x1, x2)
    expect = np.array([[[1.],
                        [1.]],
                       [[1.],
                        [1.]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_tuple_grad():
    """
    Feature: concat grad
    Description: ConcatGrad with tuple input
    Expectation: success
    """
    x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(np.float32))
    x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(np.float32))
    concat = ConcatWithTuple()
    output = GradConcat(concat)(x1, x2)
    expect = np.array([[[1.],
                        [1.]],
                       [[1.],
                        [1.]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()


class ConcatV43(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV43, self).__init__()

        self.cat = P.Concat(axis=3)
        self.x1 = Tensor(np.arange(2 * 2 * 2 * 2).reshape(2, 2, 2, 2).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 2 * 2 * 3).reshape(2, 2, 2, 3).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis43(nptype):
    cat = ConcatV43(nptype)
    output = cat()
    expect = np.array([[[[0., 1., 0., 1., 2.],
                         [2., 3., 3., 4., 5.]],
                        [[4., 5., 6., 7., 8.],
                         [6., 7., 9., 10., 11.]]],
                       [[[8., 9., 12., 13., 14.],
                         [10., 11., 15., 16., 17.]],
                        [[12., 13., 18., 19., 20.],
                         [14., 15., 21., 22., 23.]]]]).astype(nptype)
    assert (output.asnumpy() == expect).all()
    print(output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_float32():
    """
    Feature: concat in axis-3
    Description: Concat in axis 3 and float32 dtype inputs
    Expectation: success
    """
    axis43(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_int32():
    """
    Feature: concat in axis-3
    Description: Concat in axis 3 and int32 dtype inputs
    Expectation: success
    """
    axis43(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis43_bool():
    """
    Feature: concat in axis-3
    Description: Concat in axis 3 and bool dtype inputs
    Expectation: success
    """
    axis43(np.bool)


class ConcatV21(nn.Cell):
    def __init__(self, nptype):
        super(ConcatV21, self).__init__()

        self.cat = P.Concat(axis=1)
        self.x1 = Tensor(np.arange(2 * 2).reshape(2, 2).astype(nptype))
        self.x2 = Tensor(np.arange(2 * 3).reshape(2, 3).astype(nptype))

    def construct(self):
        return self.cat((self.x1, self.x2))


def axis21(nptype):
    cat = ConcatV21(nptype)
    output = cat()
    expect = np.array([[0., 1., 0., 1., 2.],
                       [2., 3., 3., 4., 5.]]).astype(nptype)
    assert (output.asnumpy() == expect).all()
    print(output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis21_float32():
    """
    Feature: concat with 2 inputs
    Description: Concat with 2 inputs of float32 dtype, asix = 1
    Expectation: success
    """
    axis21(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis21_int32():
    """
    Feature: concat with 2 inputs
    Description: Concat with 2 inputs of int32 dtype, asix = 1
    Expectation: success
    """
    axis21(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_axis21_bool():
    """
    Feature: concat with 2 inputs
    Description: Concat with 2 inputs of bool dtype, asix = 1
    Expectation: success
    """
    axis21(np.bool)


class Concat3INet(nn.Cell):
    def __init__(self):
        super(Concat3INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3):
        return self.cat((x1, x2, x3))


def concat_3i(nptype):
    cat = Concat3INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(nptype)
    x2_np = np.random.randn(32, 8, 224, 224).astype(nptype)
    x3_np = np.random.randn(32, 10, 224, 224).astype(nptype)
    output_np = np.concatenate((x1_np, x2_np, x3_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_3i_float32():
    """
    Feature: concat with 3 inputs
    Description: Concat with 3 inputs of float32 dtype
    Expectation: success
    """
    concat_3i(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_3i_int32():
    """
    Feature: concat with 3 inputs
    Description: Concat with 3 inputs of int32 dtype
    Expectation: success
    """
    concat_3i(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_3i_bool():
    """
    Feature: concat with 3 inputs
    Description: Concat with 3 inputs of bool dtype
    Expectation: success
    """
    cat = Concat3INet()

    x1_np = np.random.choice([True, False], (32, 4, 224, 224)).astype(np.bool)
    x2_np = np.random.choice([True, False], (32, 8, 224, 224)).astype(np.bool)
    x3_np = np.random.choice([True, False], (32, 10, 224, 224)).astype(np.bool)
    output_np = np.concatenate((x1_np, x2_np, x3_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms)

    assert (output_ms.asnumpy() == output_np).all()


class Concat4INet(nn.Cell):
    def __init__(self):
        super(Concat4INet, self).__init__()
        self.cat = P.Concat(axis=1)

    def construct(self, x1, x2, x3, x4):
        return self.cat((x1, x2, x3, x4))


def concat_4i(nptype):
    cat = Concat4INet()

    x1_np = np.random.randn(32, 4, 224, 224).astype(nptype)
    x2_np = np.random.randn(32, 8, 224, 224).astype(nptype)
    x3_np = np.random.randn(32, 10, 224, 224).astype(nptype)
    x4_np = np.random.randn(32, 5, 224, 224).astype(nptype)
    output_np = np.concatenate((x1_np, x2_np, x3_np, x4_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    x4_ms = Tensor(x4_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms, x4_ms)

    error = np.ones(shape=output_np.shape) * 10e-6
    diff = output_ms.asnumpy() - output_np
    assert np.all(diff < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_float32():
    """
    Feature: concat with 4 inputs
    Description: Concat with 4 inputs of float32 dtype
    Expectation: success
    """
    concat_4i(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_int32():
    """
    Feature: concat with 4 inputs
    Description: Concat with 4 inputs of int32 dtype
    Expectation: success
    """
    concat_4i(np.int32)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_int8():
    """
    Feature: concat with 4 inputs
    Description: Concat with 4 inputs of int8 dtype
    Expectation: success
    """
    concat_4i(np.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_uint64():
    """
    Feature: concat with 4 inputs
    Description: Concat with 4 inputs of uint64 dtype
    Expectation: success
    """
    concat_4i(np.uint64)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_4i_bool():
    """
    Feature: concat with 4 inputs
    Description: Concat with 4 inputs of bool dtype
    Expectation: success
    """
    cat = Concat4INet()

    x1_np = np.random.choice([True, False], (32, 4, 224, 224)).astype(np.bool)
    x2_np = np.random.choice([True, False], (32, 8, 224, 224)).astype(np.bool)
    x3_np = np.random.choice([True, False], (32, 10, 224, 224)).astype(np.bool)
    x4_np = np.random.choice([True, False], (32, 5, 224, 224)).astype(np.bool)
    output_np = np.concatenate((x1_np, x2_np, x3_np, x4_np), axis=1)

    x1_ms = Tensor(x1_np)
    x2_ms = Tensor(x2_np)
    x3_ms = Tensor(x3_np)
    x4_ms = Tensor(x4_np)
    output_ms = cat(x1_ms, x2_ms, x3_ms, x4_ms)

    assert (output_ms.asnumpy() == output_np).all()


def vmap_basic():
    def cal(a, b, axis):
        return P.Concat(axis)((a, b))

    def vmap_cal(a, b, axis):
        result = vmap(cal, in_axes=(0, 0, None))(a, b, axis)
        return result

    def naive_cal(a, b, axis):
        result = []
        for i in range(a.shape[0]):
            result.append(np.concatenate((a[i], b[i]), axis=axis))
        return np.stack(result)

    input1 = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
    input2 = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
    axis = 0
    output = vmap_cal(Tensor(input1), Tensor(input2), axis).asnumpy()
    expect = naive_cal(input1, input2, axis)
    assert np.allclose(output, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_vmap():
    """
    Feature: vmap for concat
    Description: vmap rule for Concat
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    vmap_basic()


class ConcatFunctional(nn.Cell):
    def __init__(self):
        super(ConcatFunctional, self).__init__()
        self.axis = 2
        self.x1 = Tensor(np.arange(2 * 2 * 1).reshape(2, 2, 1).astype(np.float32))
        self.x2 = Tensor(np.arange(2 * 2 * 2).reshape(2, 2, 2).astype(np.float32))

    def construct(self):
        return ops.concat((self.x1, self.x2), self.axis)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_functional():
    """
    Feature: Op Concat
    Description: Concat function interface on graph mode
    Expectation: success
    """
    cat = ConcatFunctional()
    output = cat()
    expect = np.array([[[0., 0., 1.],
                        [1., 2., 3.]],
                       [[2., 4., 5.],
                        [3., 6., 7.]]]).astype(np.float32)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_functional_pynative():
    """
    Feature: Op Concat
    Description: Concat function interface on pynative mode
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    output0 = ops.concat((x1, x2))
    expect0 = np.array([[0, 1],
                        [2, 1],
                        [0, 1],
                        [2, 1]]).astype(np.float32)
    assert (output0.asnumpy() == expect0).all()
    output1 = ops.concat((x1, x2), 1)
    expect1 = np.array([[0, 1, 0, 1],
                        [2, 1, 2, 1]]).astype(np.float32)
    assert (output1.asnumpy() == expect1).all()


class ConcatDynShapeNet(nn.Cell):
    def __init__(self, axis):
        super(ConcatDynShapeNet, self).__init__()
        self.concat = P.Concat(axis=axis)

    def construct(self, *inputs):
        return self.concat(inputs)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_dynamic_shape_no_axis():
    """
    Feature: Op Concat
    Description: Concat function interface on pynative mode
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    x3 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    x1_dyn = Tensor(shape=[None, 2], dtype=x1.dtype)
    x2_dyn = Tensor(shape=[None, 2], dtype=x2.dtype)
    x3_dyn = Tensor(shape=[None, 2], dtype=x3.dtype)
    net = ConcatDynShapeNet(1)
    net.set_inputs(*[x1_dyn, x2_dyn, x3_dyn])
    out = net(*[x1, x2, x3])
    excepted_shape = (2, 6)
    assert out.asnumpy().shape == excepted_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_dynamic_shape_axis():
    """
    Feature: Op Concat
    Description: Concat function interface on pynative mode
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x2 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x3 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x2_dyn = Tensor(shape=[None, 2], dtype=x2.dtype)
    net = ConcatDynShapeNet(0)
    net.set_inputs(*[x1, x2_dyn, x3])
    out = net(*[x1, x2, x3])
    excepted_shape = (9, 2)
    assert out.asnumpy().shape == excepted_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_concat_dynamic_shape_all():
    """
    Feature: Op Concat
    Description: Concat function interface on pynative mode
    Expectation: success
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x1 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x2 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x3 = Tensor(np.array([[0, 1], [2, 1], [3, 3]]).astype(np.float32))
    x2_dyn = Tensor(shape=[None, 2], dtype=x2.dtype)
    x3_dyn = Tensor(shape=[3, None], dtype=x3.dtype)
    net = ConcatDynShapeNet(0)
    net.set_inputs(*[x1, x2_dyn, x3_dyn])
    out = net(*[x1, x2, x3])
    excepted_shape = (9, 2)
    assert out.asnumpy().shape == excepted_shape
