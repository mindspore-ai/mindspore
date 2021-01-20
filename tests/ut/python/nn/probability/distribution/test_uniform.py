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
"""
Test nn.probability.distribution.Uniform.
"""
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor


def test_uniform_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.Uniform([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)


def test_type():
    with pytest.raises(TypeError):
        msd.Uniform(0., 1., dtype=dtype.int32)


def test_name():
    with pytest.raises(TypeError):
        msd.Uniform(0., 1., name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Uniform(0., 1., seed='seed')


def test_arguments():
    """
    Args passing during initialization.
    """
    u = msd.Uniform()
    assert isinstance(u, msd.Distribution)
    u = msd.Uniform([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(u, msd.Distribution)


def test_invalid_range():
    """
    Test range of uniform distribution.
    """
    with pytest.raises(ValueError):
        msd.Uniform(0.0, 0.0, dtype=dtype.float32)
    with pytest.raises(ValueError):
        msd.Uniform(1.0, 0.0, dtype=dtype.float32)


class UniformProb(nn.Cell):
    """
    Uniform distribution: initialize with low/high.
    """

    def __init__(self):
        super(UniformProb, self).__init__()
        self.u = msd.Uniform(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.u.prob(value)
        log_prob = self.u.log_prob(value)
        cdf = self.u.cdf(value)
        log_cdf = self.u.log_cdf(value)
        sf = self.u.survival_function(value)
        log_sf = self.u.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


def test_uniform_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = UniformProb()
    value = Tensor([3.1, 3.2, 3.3, 3.4], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class UniformProb1(nn.Cell):
    """
    Uniform distribution: initialize without low/high.
    """

    def __init__(self):
        super(UniformProb1, self).__init__()
        self.u = msd.Uniform(dtype=dtype.float32)

    def construct(self, value, low, high):
        prob = self.u.prob(value, low, high)
        log_prob = self.u.log_prob(value, low, high)
        cdf = self.u.cdf(value, low, high)
        log_cdf = self.u.log_cdf(value, low, high)
        sf = self.u.survival_function(value, low, high)
        log_sf = self.u.log_survival(value, low, high)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


def test_uniform_prob1():
    """
    Test probability functions: passing low/high, value through construct.
    """
    net = UniformProb1()
    value = Tensor([0.1, 0.2, 0.3, 0.9], dtype=dtype.float32)
    low = Tensor([0.0], dtype=dtype.float32)
    high = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, low, high)
    assert isinstance(ans, Tensor)


class UniformKl(nn.Cell):
    """
    Test class: kl_loss of Uniform distribution.
    """

    def __init__(self):
        super(UniformKl, self).__init__()
        self.u1 = msd.Uniform(
            np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.u2 = msd.Uniform(dtype=dtype.float32)

    def construct(self, low_b, high_b, low_a, high_a):
        kl1 = self.u1.kl_loss('Uniform', low_b, high_b)
        kl2 = self.u2.kl_loss('Uniform', low_b, high_b, low_a, high_a)
        return kl1 + kl2


def test_kl():
    """
    Test kl_loss.
    """
    net = UniformKl()
    low_b = Tensor(np.array([0.0]).astype(np.float32), dtype=dtype.float32)
    high_b = Tensor(np.array([5.0]).astype(np.float32), dtype=dtype.float32)
    low_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    high_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(low_b, high_b, low_a, high_a)
    assert isinstance(ans, Tensor)


class UniformCrossEntropy(nn.Cell):
    """
    Test class: cross_entropy of Uniform distribution.
    """

    def __init__(self):
        super(UniformCrossEntropy, self).__init__()
        self.u1 = msd.Uniform(
            np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.u2 = msd.Uniform(dtype=dtype.float32)

    def construct(self, low_b, high_b, low_a, high_a):
        h1 = self.u1.cross_entropy('Uniform', low_b, high_b)
        h2 = self.u2.cross_entropy('Uniform', low_b, high_b, low_a, high_a)
        return h1 + h2


def test_cross_entropy():
    """
    Test cross_entropy between Uniform distributions.
    """
    net = UniformCrossEntropy()
    low_b = Tensor(np.array([0.0]).astype(np.float32), dtype=dtype.float32)
    high_b = Tensor(np.array([5.0]).astype(np.float32), dtype=dtype.float32)
    low_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    high_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(low_b, high_b, low_a, high_a)
    assert isinstance(ans, Tensor)


class UniformBasics(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(UniformBasics, self).__init__()
        self.u = msd.Uniform(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mean = self.u.mean()
        sd = self.u.sd()
        var = self.u.var()
        entropy = self.u.entropy()
        return mean + sd + var + entropy


def test_bascis():
    """
    Test mean/sd/var/mode/entropy functionality of Uniform.
    """
    net = UniformBasics()
    ans = net()
    assert isinstance(ans, Tensor)


class UniConstruct(nn.Cell):
    """
    Uniform distribution: going through construct.
    """

    def __init__(self):
        super(UniConstruct, self).__init__()
        self.u = msd.Uniform(-4.0, 4.0)
        self.u1 = msd.Uniform()

    def construct(self, value, low, high):
        prob = self.u('prob', value)
        prob1 = self.u('prob', value, low, high)
        prob2 = self.u1('prob', value, low, high)
        return prob + prob1 + prob2


def test_uniform_construct():
    """
    Test probability function going through construct.
    """
    net = UniConstruct()
    value = Tensor([-5.0, 0.0, 1.0, 5.0], dtype=dtype.float32)
    low = Tensor([-1.0], dtype=dtype.float32)
    high = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, low, high)
    assert isinstance(ans, Tensor)
