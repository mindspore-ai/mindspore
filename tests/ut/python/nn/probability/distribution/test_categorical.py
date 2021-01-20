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
Test nn.probability.distribution.Categorical.
"""
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor


def test_arguments():
    """
    Args passing during initialization.
    """
    c = msd.Categorical()
    assert isinstance(c, msd.Distribution)
    c = msd.Categorical([0.1, 0.9], dtype=dtype.int32)
    assert isinstance(c, msd.Distribution)


def test_type():
    with pytest.raises(TypeError):
        msd.Categorical([0.1], dtype=dtype.bool_)


def test_name():
    with pytest.raises(TypeError):
        msd.Categorical([0.1], name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Categorical([0.1], seed='seed')


def test_prob():
    """
    Invalid probability.
    """
    with pytest.raises(ValueError):
        msd.Categorical([-0.1], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical([1.1], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical([0.0], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical([1.0], dtype=dtype.int32)


def test_categorical_sum():
    """
    Invalid probabilities.
    """
    with pytest.raises(ValueError):
        msd.Categorical([[0.1, 0.2], [0.4, 0.6]], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical([[0.5, 0.7], [0.6, 0.6]], dtype=dtype.int32)


def rank():
    """
    Rank dimenshion less than 1.
    """
    with pytest.raises(ValueError):
        msd.Categorical(0.2, dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical(np.array(0.3).astype(np.float32), dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Categorical(
            Tensor(np.array(0.3).astype(np.float32)), dtype=dtype.int32)


class CategoricalProb(nn.Cell):
    """
    Categorical distribution: initialize with probs.
    """

    def __init__(self):
        super(CategoricalProb, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, value):
        prob = self.c.prob(value)
        log_prob = self.c.log_prob(value)
        cdf = self.c.cdf(value)
        log_cdf = self.c.log_cdf(value)
        sf = self.c.survival_function(value)
        log_sf = self.c.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


def test_categorical_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = CategoricalProb()
    value = Tensor([0, 1, 0, 1, 0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class CategoricalProb1(nn.Cell):
    """
    Categorical distribution: initialize without probs.
    """

    def __init__(self):
        super(CategoricalProb1, self).__init__()
        self.c = msd.Categorical(dtype=dtype.int32)

    def construct(self, value, probs):
        prob = self.c.prob(value, probs)
        log_prob = self.c.log_prob(value, probs)
        cdf = self.c.cdf(value, probs)
        log_cdf = self.c.log_cdf(value, probs)
        sf = self.c.survival_function(value, probs)
        log_sf = self.c.log_survival(value, probs)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


def test_categorical_prob1():
    """
    Test probability functions: passing value/probs through construct.
    """
    net = CategoricalProb1()
    value = Tensor([0, 1, 0, 1, 0], dtype=dtype.float32)
    probs = Tensor([0.3, 0.7], dtype=dtype.float32)
    ans = net(value, probs)
    assert isinstance(ans, Tensor)


class CategoricalKl(nn.Cell):
    """
    Test class: kl_loss between Categorical distributions.
    """

    def __init__(self):
        super(CategoricalKl, self).__init__()
        self.c1 = msd.Categorical([0.2, 0.2, 0.6], dtype=dtype.int32)
        self.c2 = msd.Categorical(dtype=dtype.int32)

    def construct(self, probs_b, probs_a):
        kl1 = self.c1.kl_loss('Categorical', probs_b)
        kl2 = self.c2.kl_loss('Categorical', probs_b, probs_a)
        return kl1 + kl2


def test_kl():
    """
    Test kl_loss function.
    """
    ber_net = CategoricalKl()
    probs_b = Tensor([0.3, 0.1, 0.6], dtype=dtype.float32)
    probs_a = Tensor([0.7, 0.2, 0.1], dtype=dtype.float32)
    ans = ber_net(probs_b, probs_a)
    assert isinstance(ans, Tensor)


class CategoricalCrossEntropy(nn.Cell):
    """
    Test class: cross_entropy of Categorical distribution.
    """

    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()
        self.c1 = msd.Categorical([0.1, 0.7, 0.2], dtype=dtype.int32)
        self.c2 = msd.Categorical(dtype=dtype.int32)

    def construct(self, probs_b, probs_a):
        h1 = self.c1.cross_entropy('Categorical', probs_b)
        h2 = self.c2.cross_entropy('Categorical', probs_b, probs_a)
        return h1 + h2


def test_cross_entropy():
    """
    Test cross_entropy between Categorical distributions.
    """
    net = CategoricalCrossEntropy()
    probs_b = Tensor([0.3, 0.1, 0.6], dtype=dtype.float32)
    probs_a = Tensor([0.7, 0.2, 0.1], dtype=dtype.float32)
    ans = net(probs_b, probs_a)
    assert isinstance(ans, Tensor)


class CategoricalConstruct(nn.Cell):
    """
    Categorical distribution: going through construct.
    """

    def __init__(self):
        super(CategoricalConstruct, self).__init__()
        self.c = msd.Categorical([0.1, 0.8, 0.1], dtype=dtype.int32)
        self.c1 = msd.Categorical(dtype=dtype.int32)

    def construct(self, value, probs):
        prob = self.c('prob', value)
        prob1 = self.c('prob', value, probs)
        prob2 = self.c1('prob', value, probs)
        return prob + prob1 + prob2


def test_categorical_construct():
    """
    Test probability function going through construct.
    """
    net = CategoricalConstruct()
    value = Tensor([0, 1, 2, 0, 0], dtype=dtype.float32)
    probs = Tensor([0.5, 0.4, 0.1], dtype=dtype.float32)
    ans = net(value, probs)
    assert isinstance(ans, Tensor)


class CategoricalBasics(nn.Cell):
    """
    Test class: basic mean/var/mode/entropy function.
    """

    def __init__(self):
        super(CategoricalBasics, self).__init__()
        self.c = msd.Categorical([0.2, 0.7, 0.1], dtype=dtype.int32)
        self.c1 = msd.Categorical(dtype=dtype.int32)

    def construct(self, probs):
        basics1 = self.c.mean() + self.c.var() + self.c.mode() + self.c.entropy()
        basics2 = self.c1.mean(probs) + self.c1.var(probs) +\
            self.c1.mode(probs) + self.c1.entropy(probs)
        return basics1 + basics2


def test_basics():
    """
    Test basics functionality of Categorical distribution.
    """
    net = CategoricalBasics()
    probs = Tensor([0.7, 0.2, 0.1], dtype=dtype.float32)
    ans = net(probs)
    assert isinstance(ans, Tensor)
