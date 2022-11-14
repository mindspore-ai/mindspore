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
""" test_initializer """
import math
import unittest
from functools import reduce
import numpy as np
import pytest as py
from scipy import stats

import mindspore as ms
import mindspore.common.initializer as init
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import initializer, Identity, Dirac, Sparse, VarianceScaling, Orthogonal
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.nn import Conv2d
from mindspore.ops import operations as P
from mindspore._c_expression import _random_normal, _random_uniform, _truncated_normal
from ..ut_filter import non_graph_engine


# pylint: disable= protected-access
class InitTwo(init.Initializer):
    """Initialize the array to two."""

    def _initialize(self, arr):
        init._assignment(arr, 2)


def _check_value(tensor, value_min, value_max):
    nd = tensor.asnumpy()
    for ele in nd.flatten():
        if value_min <= ele <= value_max:
            continue
        raise ValueError('value_min = %d, ele = %d, value_max = %d'
                         % (value_min, ele, value_max))


def _check_uniform(tensor, boundary_a, boundary_b):
    samples = tensor.asnumpy().reshape((-1))
    _, p = stats.kstest(samples, 'uniform', (boundary_a, (boundary_b - boundary_a)))
    print("p-value is %f" % p)
    return p > 0.0001


def test_init_initializer():
    """
    Feature: Test initializer.
    Description: Test initializer.
    Expectation: Shape and value is initialized successfully..
    """
    tensor = init.initializer(InitTwo(), [2, 2], ms.int32)
    assert tensor.shape == (2, 2)
    _check_value(tensor.init_data(), 2, 2)


def test_init_tensor():
    tensor = ms.Tensor(np.zeros([1, 2, 3]))
    tensor = init.initializer(tensor, [1, 2, 3], ms.float32)
    assert tensor.shape == (1, 2, 3)


def test_init_zero_default_dtype():
    tensor = init.initializer(init.Zero(), [2, 2])
    assert tensor.dtype == ms.float32
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero():
    tensor = init.initializer(init.Zero(), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero_alias_default_dtype():
    tensor = init.initializer('zeros', [1, 2])
    assert tensor.dtype == ms.float32
    _check_value(tensor.init_data(), 0, 0)


def test_init_zero_alias():
    tensor = init.initializer('zeros', [1, 2], ms.float32)
    _check_value(tensor.init_data(), 0, 0)


def test_init_one():
    tensor = init.initializer(init.One(), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_one_alias():
    tensor = init.initializer('ones', [1, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_constant():
    tensor = init.initializer(init.Constant(1), [2, 2], ms.float32)
    _check_value(tensor.init_data(), 1, 1)


def test_init_uniform():
    scale = 10
    tensor = init.initializer(init.Uniform(scale=scale), [5, 4], ms.float32)
    _check_value(tensor.init_data(), -scale, scale)


def test_init_uniform_alias():
    scale = 100
    tensor = init.initializer('uniform', [5, 4], ms.float32)
    _check_value(tensor.init_data(), -scale, scale)


def test_init_normal():
    tensor = init.initializer(init.Normal(), [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'Normal init failed!'


def test_init_truncated_normal():
    tensor = init.initializer(init.TruncatedNormal(), [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'TruncatedNormal init failed!'


def test_init_normal_alias():
    tensor = init.initializer('normal', [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'Normal init failed!'


def test_init_truncatednormal_alias():
    tensor = init.initializer('truncatednormal', [5, 4], ms.float32)
    assert isinstance(tensor, Tensor), 'TruncatedNormal init failed!'


def test_init_abnormal():
    with py.raises(TypeError):
        init.initializer([''], [5, 4], ms.float32)


def test_initializer_reinit():
    weights = init.initializer("XavierUniform", shape=(10, 1, 10, 10), dtype=ms.float16)
    assert isinstance(weights, Tensor), 'XavierUniform init failed!'


def test_init_xavier_uniform():
    """ test_init_xavier_uniform """
    gain = 1.2
    tensor1 = init.initializer(init.XavierUniform(gain=gain), [20, 22], ms.float32).init_data()
    tensor2 = init.initializer(init.XavierUniform(), [20, 22], ms.float32).init_data()
    tensor3 = init.initializer(init.XavierUniform(gain=gain), [20, 22, 5, 5], ms.float32).init_data()
    tensor4 = init.initializer(init.XavierUniform(), [20, 22, 5, 5], ms.float32).init_data()
    tensor5 = init.initializer('xavier_uniform', [20, 22, 5, 5], ms.float32).init_data()
    tensor6 = init.initializer('xavier_uniform', [20, 22], ms.float32).init_data()
    tensor_dict = {tensor1: gain, tensor2: None, tensor3: gain, tensor4: None, tensor5: None, tensor6: None}

    for tensor, gain_value in tensor_dict.items():
        if gain_value is None:
            gain_value = 1
        shape = tensor.asnumpy().shape
        if len(shape) > 2:
            s = reduce(lambda x, y: x * y, shape[2:])
        else:
            s = 1
        n_in = shape[1] * s
        n_out = shape[0] * s
        std = gain_value * math.sqrt(2 / (n_in + n_out))
        boundary = std * math.sqrt(3)
        assert _check_uniform(tensor, -boundary, boundary)


def test_init_xavier_uniform_error():
    with py.raises(ValueError):
        init.initializer(init.XavierUniform(), [6], ms.float32).init_data()


def test_init_he_uniform():
    """ test_init_he_uniform """
    tensor1 = init.initializer(init.HeUniform(), [20, 22], ms.float32)
    tensor2 = init.initializer(init.HeUniform(), [20, 22, 5, 5], ms.float32)
    tensor3 = init.initializer('he_uniform', [20, 22, 5, 5], ms.float32)
    tensor4 = init.initializer('he_uniform', [20, 22], ms.float32)
    tensors = [tensor1.init_data(), tensor2.init_data(), tensor3.init_data(), tensor4.init_data()]

    for tensor in tensors:
        shape = tensor.asnumpy().shape
        if len(shape) > 2:
            s = reduce(lambda x, y: x * y, shape[2:])
        else:
            s = 1
        n_in = shape[1] * s
        std = math.sqrt(2 / n_in)
        boundary = std * math.sqrt(3)
        assert _check_uniform(tensor, -boundary, boundary)


def test_init_he_uniform_error():
    with py.raises(ValueError):
        init.initializer(init.HeUniform(), [6], ms.float32).init_data()


def test_init_identity():
    """
    Feature: Test identity initializer.
    Description: Test if error is raised when the shape of the initialized tensor is not correct.
    Expectation: ValueError is raised.
    """
    with py.raises(ValueError):
        tensor = init.initializer(init.Identity(), [5, 4, 6], ms.float32)
        tensor.init_data()


def test_identity():
    """
    Feature: Test identity initializer.
    Description: Initialize an identity matrix to fill a Tensor.
    Expectation: The Tensor is initialized with identity matrix.
    """
    tensor1 = initializer(Identity(), [3, 3], ms.float32)
    tensor2 = initializer('identity', [3, 4], ms.float32)
    tensor3 = initializer('identity', [4, 3], ms.float32)
    expect1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    expect2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    expect3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
    assert (tensor1.asnumpy() == expect1).all()
    assert (tensor2.asnumpy() == expect2).all()
    assert (tensor3.asnumpy() == expect3).all()


def test_init_sparse():
    """
    Feature: Test sparse initializer.
    Description: Test if error is raised when the shape of the initialized tensor is not correct.
    Expectation: ValueError is raised.
    """
    with py.raises(ValueError):
        tensor = init.initializer(init.Sparse(sparsity=0.1), [5, 4, 6], ms.float32)
        tensor.init_data()


def test_init_dirac():
    """
    Feature: Test dirac initializer.
    Description: Test if error is raised when the shape of the initialized tensor is not correct.
    or shape[0] is not divisible by group.
    Expectation: ValueError is raised.
    """
    with py.raises(ValueError):
        tensor1 = init.initializer(init.Dirac(groups=2), [5, 4, 6], ms.float32)
        tensor1.init_data()

    with py.raises(ValueError):
        tensor2 = init.initializer(init.Dirac(groups=1), [5, 4], ms.float32)
        tensor2.init_data()

    with py.raises(ValueError):
        tensor3 = init.initializer(init.Dirac(groups=1), [5, 4, 6, 7, 8, 9], ms.float32)
        tensor3.init_data()


def test_init_orthogonal():
    """
    Feature: Test orthogonal initializer.
    Description: Test if error is raised when the shape of the initialized tensor is not correct.
    Expectation: ValueError is raised.
    """
    with py.raises(ValueError):
        tensor = init.initializer(init.Orthogonal(), [5,], ms.float32)
        tensor.init_data()


def test_orthogonal():
    """
    Feature: Test orthogonal initializer.
    Description: Initialize a (semi) orthogonal matrix to fill the input tensor.
    Expectation: The Tensor is initialized with values from orthogonal matrix.
    """
    identity = np.identity(2)
    tensor1 = initializer(Orthogonal(gain=1.), [2, 2], ms.float32)
    t1 = tensor1.init_data()
    np_t1 = t1.asnumpy()
    trans = np.transpose(np_t1)
    output = np.matmul(np_t1, trans)
    assert np.allclose(output, identity, atol=1e-6, rtol=1e-7)


def test_init_variancescaling():
    """
    Feature: Test orthogonal initializer.
    Description: Test if error is raised when scale is less than 0 or mode and distribution are not correct.
    Expectation: ValueError is raised.
    """
    with py.raises(ValueError):
        init.initializer(init.VarianceScaling(scale=-0.1), [5, 4, 6], ms.float32)

    with py.raises(ValueError):
        init.initializer(init.VarianceScaling(scale=0.1, mode='fans'), [5, 4, 6], ms.float32)

    with py.raises(ValueError):
        init.initializer(init.VarianceScaling(scale=0.1, mode='fan_in',
                                              distribution='uniformal'), [5, 4, 6], ms.float32)


def test_conv2d_abnormal_kernel_negative():
    """
    Feature: Random initializers that implemented in cpp.
    Description: Test random initializers that implemented in cpp.
    Expectation: Data is initialized successfully.
    """
    kernel = init.initializer(init.Normal(sigma=1.0), [64, 3, 7, 7], ms.float32).init_data()
    with py.raises(ValueError):
        ms.Model(
            Conv2d(in_channels=3, out_channels=64, kernel_size=-7, stride=3,
                   padding=0, weight_init=ms.Tensor(kernel)))


@non_graph_engine
def test_conv2d_abnormal_kernel_normal():
    """
    Feature: Random initializers that implemented in cpp.
    Description: Test random initializers that implemented in cpp.
    Expectation: Data is initialized successfully.
    """
    kernel = init.initializer(init.Normal(sigma=1.0), [64, 3, 7, 7], ms.float32).init_data()
    input_data = init.initializer(init.Normal(sigma=1.0), [32, 3, 224, 112], ms.float32).init_data()
    context.set_context(mode=context.GRAPH_MODE)
    model = ms.Model(
        Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3,
               padding=0, weight_init=kernel))
    model.predict(input_data)


@non_graph_engine
def test_conv2d_abnormal_kernel_truncated_normal():
    """
    Feature: Random initializers that implemented in cpp.
    Description: Test random initializers that implemented in cpp.
    Expectation: Data is initialized successfully.
    """
    input_data = init.initializer(init.TruncatedNormal(), [64, 3, 7, 7], ms.float32).init_data()
    context.set_context(mode=context.GRAPH_MODE)
    model = ms.Model(
        Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=3,
               padding=0, weight_init="truncatednormal"))
    model.predict(input_data)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.t1 = Parameter(init.initializer('uniform', [5, 4], ms.float32), name="w1")
        self.t2 = Parameter(init.initializer(init.TruncatedNormal(), [5, 4], ms.float32), name="w2")

    def construct(self, x):
        z = self.add(x, self.t1)
        z = self.add(z, self.t2)
        return z


def test_weight_shape():
    context.set_context(mode=context.GRAPH_MODE)
    a = np.arange(20).reshape(5, 4)
    t = Tensor(a, dtype=ms.float32)
    net = Net()
    out = net(t)
    print(out)


def test_init_with_same_numpy_seed():
    """
    Feature: Random initializers that depend on numpy random seed.
    Description: Test random initializers with same numpy random seed.
    Expectation: Initialized data is same with same numpy random seed.
    """
    shape = [12, 34]
    np.random.seed(1234)
    uniform1 = init.initializer('uniform', shape, ms.float32).init_data()
    normal1 = init.initializer('normal', shape, ms.float32).init_data()
    truncnorm1 = init.initializer('truncatednormal', shape, ms.float32).init_data()

    np.random.seed(1234)
    uniform2 = init.initializer('uniform', shape, ms.float32).init_data()
    normal2 = init.initializer('normal', shape, ms.float32).init_data()
    truncnorm2 = init.initializer('truncatednormal', shape, ms.float32).init_data()

    assert np.allclose(uniform1.asnumpy(), uniform2.asnumpy())
    assert np.allclose(normal1.asnumpy(), normal2.asnumpy())
    assert np.allclose(truncnorm1.asnumpy(), truncnorm2.asnumpy())

    # Reset numpy random seed after test.
    np.random.seed()


def test_cpp_random_initializer():
    """
    Feature: Random initializers that implemented in cpp.
    Description: Test random initializers that implemented in cpp.
    Expectation: Data is initialized successfully.
    """
    ut = unittest.TestCase()
    shape = (11, 512)

    # Random normal.
    data = np.ndarray(shape=shape, dtype=np.float32)
    _random_normal(0, data, 0.0, 1.0)
    ut.assertAlmostEqual(np.mean(data), 0.0, delta=0.1)
    ut.assertAlmostEqual(np.std(data), 1.0, delta=0.1)

    # Random uniform.
    data = np.ndarray(shape=shape, dtype=np.float32)
    _random_uniform(0, data, -1.0, 1.0)
    ut.assertAlmostEqual(np.mean(data), 0.0, delta=0.1)
    ut.assertGreater(np.std(data), 0.0)

    # Truncated random.
    data = np.ndarray(shape=shape, dtype=np.float32)
    _truncated_normal(0, data, -2.0, 2.0, 0.0, 1.0)
    ut.assertAlmostEqual(np.mean(data), 0.0, delta=0.1)
    ut.assertGreaterEqual(np.min(data), -2.0)
    ut.assertLessEqual(np.max(data), 2.0)

    # Same seeds, same results.
    data1 = np.ndarray(shape=shape, dtype=np.float32)
    _random_normal(12345678, data1, 0.0, 1.0)
    data2 = np.ndarray(shape=shape, dtype=np.float32)
    _random_normal(12345678, data2, 0.0, 1.0)
    assert np.allclose(data1, data2)

    # Different seeds, different results.
    data3 = np.ndarray(shape=shape, dtype=np.float32)
    _random_normal(12345679, data3, 0.0, 1.0)
    assert not np.allclose(data1, data3)

    # Check distributions by K-S test.
    np.random.seed(42)
    seed = np.random.randint(low=1, high=(1 << 63))
    count = 10000
    data = np.ndarray(shape=(count), dtype=np.float32)
    _random_uniform(seed, data, 0.0, 1.0)
    data2 = np.random.uniform(0.0, 1.0, size=count)
    _, p = stats.kstest(data, data2, N=count)
    assert p > 0.05

    _random_normal(seed, data, 0.0, 1.0)
    data2 = np.random.normal(0.0, 1.0, size=count)
    _, p = stats.kstest(data, data2, N=count)
    assert p > 0.05

    _truncated_normal(seed, data, -2, 2, 0.0, 1.0)
    data2 = stats.truncnorm.rvs(-2, 2, loc=0.0, scale=1.0, size=count, random_state=None)
    _, p = stats.kstest(data, data2, N=count)
    assert p > 0.05

    # Reset numpy random seed after test.
    np.random.seed()


def test_sparse():
    """
    Feature: Test sparse initializer.
    Description: Initialize a 2 dimension sparse matrix to fill the input tensor.
    Expectation: The Tensor is initialized with a 2 dimension sparse matrix.
    """
    tensor1 = init.initializer(Sparse(sparsity=0.2, sigma=0.01), [5, 6], ms.float32)
    output = tensor1.init_data()
    assert np.array_equal(np.count_nonzero(output.asnumpy(), axis=0), [4, 4, 4, 4, 4, 4])


def test_variancescaling():
    """
    Feature: Test varianceScaling initializer.
    Description: Randomly initialize an array with scaling to fill the input tensor.
    Expectation: The Tensor is initialized successfully.
    """
    ms.set_seed(0)
    tensor1 = initializer('varianceScaling', [2, 3], ms.float32)
    tensor2 = initializer(VarianceScaling(scale=1.0, mode='fan_out', distribution='untruncated_normal'), [2, 3],
                          ms.float32)
    tensor3 = initializer(VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'), [2, 3],
                          ms.float32)
    tensor4 = initializer(VarianceScaling(scale=3.0, mode='fan_avg', distribution='uniform'), [2, 3],
                          ms.float32)
    t1 = tensor1.init_data()
    expected_t1 = np.array([[0.49535394, -0.03666719, 0.23151064],
                            [-0.08424897, 0.39260703, -0.26104233]])
    t2 = tensor2.init_data()
    expected_t2 = np.array([[1.2710124e+00, 1.2299923e-03, -1.1589712e+00],
                            [1.1465757e+00, -2.2482322e-01, 9.2637345e-02]])
    t3 = tensor3.init_data()
    expected_t3 = np.array([[1.2023407, -0.9182362, 0.20436235],
                            [0.8581208, 1.0288558, 1.0927733]])
    t4 = tensor4.init_data()
    expected_t4 = np.array([[1.2470493, -1.0861205, -1.1339132],
                            [-0.07604776, -1.8196303, 0.5115674]])
    assert np.allclose(t1.asnumpy(), expected_t1)
    assert np.allclose(t2.asnumpy(), expected_t2)
    assert np.allclose(t3.asnumpy(), expected_t3)
    assert np.allclose(t4.asnumpy(), expected_t4)


def test_dirac():
    """
    Feature: Test dirac initializer.
    Description: Initialize input tensor with the Dirac delta function.
    Expectation: The Tensor is correctly initialized.
    """
    tensor3_1 = initializer(Dirac(groups=1), [6, 2, 3], ms.float32)
    tensor3_2 = initializer(Dirac(groups=2), [6, 2, 3], ms.float32)
    tensor3_3 = initializer(Dirac(groups=3), [6, 2, 3], ms.float32)

    tensor4_1 = initializer(Dirac(groups=1), [6, 4, 3, 3], ms.float32)
    tensor4_2 = initializer(Dirac(groups=2), [6, 4, 3, 3], ms.float32)
    tensor4_3 = initializer(Dirac(groups=3), [6, 4, 3, 3], ms.float32)

    tensor5_1 = initializer(Dirac(groups=1), [6, 2, 3, 3, 3], ms.float32)
    tensor5_2 = initializer(Dirac(groups=2), [6, 2, 3, 3, 3], ms.float32)
    tensor5_3 = initializer(Dirac(groups=3), [6, 2, 3, 3, 3], ms.float32)

    expectation3_1 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32)

    expectation3_2 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 0., 0.], [0., 0., 0.]]], dtype=np.float32)

    expectation3_3 = np.array([[[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]],
                               [[0., 1., 0.], [0., 0., 0.]],
                               [[0., 0., 0.], [0., 1., 0.]]], dtype=np.float32)

    expectation4_1 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation4_2 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation4_3 = np.array([[[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]], dtype=np.float32)

    expectation5_1 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    expectation5_2 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    expectation5_3 = np.array([[[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]],
                               [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
                                [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
                                 [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]]], dtype=np.float32)

    assert (tensor3_1.asnumpy() == expectation3_1).all()
    assert (tensor3_2.asnumpy() == expectation3_2).all()
    assert (tensor3_3.asnumpy() == expectation3_3).all()

    assert (tensor4_1.asnumpy() == expectation4_1).all()
    assert (tensor4_2.asnumpy() == expectation4_2).all()
    assert (tensor4_3.asnumpy() == expectation4_3).all()

    assert (tensor5_1.asnumpy() == expectation5_1).all()
    assert (tensor5_2.asnumpy() == expectation5_2).all()
    assert (tensor5_3.asnumpy() == expectation5_3).all()
