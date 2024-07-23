# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SquareNet(nn.Cell):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.square = P.Square()

    def construct(self, x):
        return self.square(x)


class FloorNet(nn.Cell):
    def __init__(self):
        super(FloorNet, self).__init__()
        self.floor = P.Floor()

    def construct(self, x):
        return self.floor(x)


class RoundNet(nn.Cell):
    def __init__(self):
        super(RoundNet, self).__init__()
        self.round = P.Round()

    def construct(self, x):
        return self.round(x)


class ReciprocalNet(nn.Cell):
    def __init__(self):
        super(ReciprocalNet, self).__init__()
        self.reciprocal = P.Reciprocal()

    def construct(self, x):
        return self.reciprocal(x)


class RintNet(nn.Cell):
    def __init__(self):
        super(RintNet, self).__init__()
        self.rint = P.Rint()

    def construct(self, x):
        return self.rint(x)


class IdentityNet(nn.Cell):
    def __init__(self):
        super(IdentityNet, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        return self.identity(x)


class InvDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(InvDynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, x):
        x_unique, _ = self.unique(x)
        x_unique = self.reshape(x_unique, (3, 3))
        return F.inv(x_unique)


class InvertDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(InvertDynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, x):
        x_unique, _ = self.unique(x)
        x_unique = self.reshape(x_unique, (3, 3))
        x_unique = F.cast(x_unique, mindspore.int16)
        return F.invert(x_unique)


class SoftsignDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(SoftsignDynamicShapeNet, self).__init__()
        self.unique = P.Unique()
        self.reshape = P.Reshape()

    def construct(self, x):
        x_unique, _ = self.unique(x)
        x_unique = self.reshape(x_unique, (3, 3))
        return F.softsign(x_unique)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_square():
    x = np.array([1, 2, 3]).astype(np.int16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.int64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.int64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float16)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float16)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float32)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float32)
    print(output)
    assert np.all(output.asnumpy() == expect_output)

    x = np.array([1, 2, 3]).astype(np.float64)
    net = SquareNet()
    output = net(Tensor(x))
    expect_output = np.array([1, 4, 9]).astype(np.float64)
    print(output)
    assert np.all(output.asnumpy() == expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_floor():
    net = FloorNet()

    x = np.random.randn(3, 4).astype(np.float16)
    x = x * 100
    output = net(Tensor(x))
    expect_output = np.floor(x).astype(np.float16)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output)

    x = np.random.randn(4, 3).astype(np.float32)
    x = x * 100
    output = net(Tensor(x))
    expect_output = np.floor(x)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output)

    x = np.random.randn(4, 3).astype(np.float64)
    x = x * 100
    output = net(Tensor(x))
    expect_output = np.floor(x)
    print(output.asnumpy())
    assert np.all(output.asnumpy() == expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_rint():
    net = RintNet()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5, 6).astype(np.float16) * prop
    output = net(Tensor(x))
    expect_output = np.rint(x).astype(np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32) * prop
    output = net(Tensor(x))
    expect_output = np.rint(x).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float64) * prop
    output = net(Tensor(x))
    expect_output = np.rint(x).astype(np.float64)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_round():
    """
    Feature: template
    Description: template
    Expectation: template
    """
    net = RoundNet()

    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(np.float16)
    output = net(Tensor(x))
    expect_output = np.round(x).astype(np.float16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(np.float32)
    output = net(Tensor(x))
    expect_output = np.round(x).astype(np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    x = np.array([0.9920, -0.4077, 0.9734, -1.0362, 1.5, -2.5, 4.5]).astype(np.float64)
    output = net(Tensor(x))
    expect_output = np.round(x).astype(np.float64)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol',
                         [(np.int32, 1.0e-7), (np.float16, 1.0e-5), (np.float32, 1.0e-5), (np.float64, 1.0e-7)])
def test_reciprocal(shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for reciprocal
    Expectation: the result match to numpy
    """
    net = ReciprocalNet()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(*shape).astype(dtype) * prop
    if dtype in (np.int32, np.int64):
        # In Mac-arm platform, if x contains 0 element, 1/0 will be -1 in that platform.
        # Therefore, here we eliminate 0 uniformly.
        x[x == 0] = 1
        x = x.astype(np.float32)
    output = net(Tensor(x))
    expect_output = np.reciprocal(x)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol',
                         [(np.int32, 1.0e-4), (np.int64, 1.0e-4), (np.float16, 1.0e-3), (np.float32, 1.0e-4),
                          (np.float64, 1.0e-5), (np.complex64, 1.0e-6), (np.complex128, 1.0e-10)])
def test_inv(shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for inv
    Expectation: the result match to numpy
    """
    inv = P.Inv()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(*shape).astype(dtype) * prop
    if dtype in (np.int32, np.int64):
        # In Mac-arm platform, if x contains 0 element, 1/0 will be -1 in that platform.
        # Therefore, here we eliminate 0 uniformly.
        x[x == 0] = 1
    output = inv(Tensor(x))
    expect_output = np.reciprocal(x).astype(dtype)
    assert np.allclose(output.asnumpy(), expect_output, atol=tol, rtol=tol, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_vmap(mode):
    """
    Feature: test inv vmap feature.
    Description: test inv vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([[0.25, 0.4, 0.31, 0.52], [0.5, 0.12, 0.31, 0.58]], dtype=np.float32))
    # Case 1
    output = F.vmap(F.inv, 0, 0)(x)
    expect_output = np.array([[4., 2.5, 3.2258065, 1.923077], [2., 8.333334, 3.2258065, 1.724138]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(F.inv, 1, 0)(x)
    expect_output = np.array([[4., 2.], [2.5, 8.333334], [3.2258065, 3.2258065], [1.923077, 1.724138]],
                             dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(F.inv, 0, 1)(x)
    expect_output = np.array([[4., 2.], [2.5, 8.333334], [3.2258065, 3.2258065], [1.923077, 1.724138]],
                             dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_inv_dynamic_shape(mode):
    """
    Feature: test inv dynamic_shape feature.
    Description: test inv dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([8., -3., 0., 0., 10., 1., 21., -3., 11., 4., -2., 10., 8.]).astype(np.float32))
    output = InvDynamicShapeNet()(x)
    expect_output = np.array([[0.125, -0.33333334, np.inf],
                              [0.1, 1., 0.04761905],
                              [0.09090909, 0.25, -0.5]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
def test_invert(shape, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for invert
    Expectation: the result match to numpy
    """
    invert = P.Invert()
    prop = 100 if np.random.random() > 0.5 else -100
    input_x = (np.random.randn(*shape) * prop).astype(dtype)
    output = invert(Tensor(input_x))
    expect_output = np.invert(input_x)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_invert_vmap(mode):
    """
    Feature: test invert vmap feature.
    Description: test invert vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([[25, 4, 13, 9], [2, -1, 0, -5]], dtype=np.int16))
    # Case 1
    output = F.vmap(F.invert, 0, 0)(x)
    expect_output = np.array([[-26, -5, -14, -10], [-3, 0, -1, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(F.invert, 1, 0)(x)
    expect_output = np.array([[-26, -3], [-5, 0], [-14, -1], [-10, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(F.invert, 0, 1)(x)
    expect_output = np.array([[-26, -3], [-5, 0], [-14, -1], [-10, 4]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_invert_dynamic_shape(mode):
    """
    Feature: test invert dynamic_shape feature.
    Description: test invert dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([8, -3, 0, 0, 10, 1, 21, -3, 11, 4, -2, 10, 8]).astype(np.int16))
    output = InvertDynamicShapeNet()(x)
    expect_output = np.array([[-9, 2, -1],
                              [-11, -2, -22],
                              [-12, -5, 1]], dtype=np.int16)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_softsign(shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for Softsign
    Expectation: the result match to numpy
    """
    softsign = P.Softsign()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(*shape).astype(dtype) * prop
    output = softsign(Tensor(x))
    expect_output = x / (1.0 + np.abs(x))
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softsign_vmap(mode):
    """
    Feature: test softsign vmap feature.
    Description: test softsign vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([[0, -1, 2, 30, -30], [2, -1, 0, -5, 50]], dtype=np.float32))
    # Case 1
    output = F.vmap(F.softsign, 0, 0)(x)
    expect_output = np.array([[0., -0.5, 0.6666667, 0.9677419, -0.9677419],
                              [0.6666667, -0.5, 0., -0.8333333, 0.98039216]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    output = F.vmap(F.softsign, 1, 0)(x)
    expect_output = np.array([[0., 0.6666667],
                              [-0.5, -0.5],
                              [0.6666667, 0.],
                              [0.9677419, -0.8333333],
                              [-0.9677419, 0.98039216]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    output = F.vmap(F.softsign, 0, 1)(x)
    expect_output = np.array([[0., 0.6666667],
                              [-0.5, -0.5],
                              [0.6666667, 0.],
                              [0.9677419, -0.8333333],
                              [-0.9677419, 0.98039216]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softsign_dynamic_shape(mode):
    """
    Feature: test softsign dynamic_shape feature.
    Description: test softsign dynamic_shape feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.array([8., -3., 0., 0., 10., 1., 21., -3., 11., 4., 2., 10., 8.]).astype(np.float32))
    output = SoftsignDynamicShapeNet()(x)
    expect_output = np.array([[0.8888889, -0.75, 0.],
                              [0.90909094, 0.5, 0.95454544],
                              [0.9166667, 0.8, 0.6666667]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_identity_pynative():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    net = IdentityNet()

    x = np.random.randn(3, 4, 5, 6).astype(np.float64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.bool)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_identity_graph():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = IdentityNet()

    x = np.random.randn(3, 4, 5, 6).astype(np.float64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.float16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int64)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int32)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int16)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.uint8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.int8)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)

    x = np.random.randn(3, 4, 5, 6).astype(np.bool)
    input_tensor = Tensor(x)
    output = net(input_tensor)
    np.testing.assert_almost_equal(output.asnumpy(), input_tensor.asnumpy())
    assert id(input_tensor) != id(output)
