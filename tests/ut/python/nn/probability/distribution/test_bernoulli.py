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
Test nn.probability.distribution.Bernoulli.
"""
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") == "CPU"


def test_arguments():
    """
    Args passing during initialization.
    """
    b = msd.Bernoulli()
    assert isinstance(b, msd.Distribution)
    b = msd.Bernoulli([0.1, 0.3, 0.5, 0.9], dtype=dtype.int32)
    assert isinstance(b, msd.Distribution)


def test_type():
    with pytest.raises(TypeError):
        msd.Bernoulli([0.1], dtype=dtype.bool_)


def test_name():
    with pytest.raises(TypeError):
        msd.Bernoulli([0.1], name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Bernoulli([0.1], seed='seed')


def test_prob():
    """
    Invalid probability.
    """
    with pytest.raises(ValueError):
        msd.Bernoulli([-0.1], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Bernoulli([1.1], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Bernoulli([0.0], dtype=dtype.int32)
    with pytest.raises(ValueError):
        msd.Bernoulli([1.0], dtype=dtype.int32)


class BernoulliProb(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """

    def __init__(self):
        super(BernoulliProb, self).__init__()
        self.b = msd.Bernoulli(0.5, dtype=dtype.int32)

    def construct(self, value):
        prob = self.b.prob(value)
        log_prob = self.b.log_prob(value)
        cdf = self.b.cdf(value)
        log_cdf = self.b.log_cdf(value)
        sf = self.b.survival_function(value)
        log_sf = self.b.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bernoulli_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = BernoulliProb()
    value = Tensor([0, 0, 0, 0, 0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class BernoulliProb1(nn.Cell):
    """
    Bernoulli distribution: initialize without probs.
    """

    def __init__(self):
        super(BernoulliProb1, self).__init__()
        self.b = msd.Bernoulli(dtype=dtype.int32)

    def construct(self, value, probs):
        prob = self.b.prob(value, probs)
        log_prob = self.b.log_prob(value, probs)
        cdf = self.b.cdf(value, probs)
        log_cdf = self.b.log_cdf(value, probs)
        sf = self.b.survival_function(value, probs)
        log_sf = self.b.log_survival(value, probs)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bernoulli_prob1():
    """
    Test probability functions: passing value/probs through construct.
    """
    net = BernoulliProb1()
    value = Tensor([0, 0, 0, 0, 0], dtype=dtype.float32)
    probs = Tensor([0.5], dtype=dtype.float32)
    ans = net(value, probs)
    assert isinstance(ans, Tensor)


class BernoulliKl(nn.Cell):
    """
    Test class: kl_loss between Bernoulli distributions.
    """

    def __init__(self):
        super(BernoulliKl, self).__init__()
        self.b1 = msd.Bernoulli(0.7, dtype=dtype.int32)
        self.b2 = msd.Bernoulli(dtype=dtype.int32)

    def construct(self, probs_b, probs_a):
        kl1 = self.b1.kl_loss('Bernoulli', probs_b)
        kl2 = self.b2.kl_loss('Bernoulli', probs_b, probs_a)
        return kl1 + kl2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_kl():
    """
    Test kl_loss function.
    """
    ber_net = BernoulliKl()
    probs_b = Tensor([0.3], dtype=dtype.float32)
    probs_a = Tensor([0.7], dtype=dtype.float32)
    ans = ber_net(probs_b, probs_a)
    assert isinstance(ans, Tensor)


class BernoulliCrossEntropy(nn.Cell):
    """
    Test class: cross_entropy of Bernoulli distribution.
    """

    def __init__(self):
        super(BernoulliCrossEntropy, self).__init__()
        self.b1 = msd.Bernoulli(0.7, dtype=dtype.int32)
        self.b2 = msd.Bernoulli(dtype=dtype.int32)

    def construct(self, probs_b, probs_a):
        h1 = self.b1.cross_entropy('Bernoulli', probs_b)
        h2 = self.b2.cross_entropy('Bernoulli', probs_b, probs_a)
        return h1 + h2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_cross_entropy():
    """
    Test cross_entropy between Bernoulli distributions.
    """
    net = BernoulliCrossEntropy()
    probs_b = Tensor([0.3], dtype=dtype.float32)
    probs_a = Tensor([0.7], dtype=dtype.float32)
    ans = net(probs_b, probs_a)
    assert isinstance(ans, Tensor)


class BernoulliConstruct(nn.Cell):
    """
    Bernoulli distribution: going through construct.
    """

    def __init__(self):
        super(BernoulliConstruct, self).__init__()
        self.b = msd.Bernoulli(0.5, dtype=dtype.int32)
        self.b1 = msd.Bernoulli(dtype=dtype.int32)

    def construct(self, value, probs):
        prob = self.b('prob', value)
        prob1 = self.b('prob', value, probs)
        prob2 = self.b1('prob', value, probs)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bernoulli_construct():
    """
    Test probability function going through construct.
    """
    net = BernoulliConstruct()
    value = Tensor([0, 0, 0, 0, 0], dtype=dtype.float32)
    probs = Tensor([0.5], dtype=dtype.float32)
    ans = net(value, probs)
    assert isinstance(ans, Tensor)


class BernoulliMean(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(BernoulliMean, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5], dtype=dtype.int32)

    def construct(self):
        mean = self.b.mean()
        return mean


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_mean():
    """
    Test mean/sd/var/mode/entropy functionality of Bernoulli distribution.
    """
    net = BernoulliMean()
    ans = net()
    assert isinstance(ans, Tensor)


class BernoulliSd(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(BernoulliSd, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5], dtype=dtype.int32)

    def construct(self):
        sd = self.b.sd()
        return sd


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_sd():
    """
    Test mean/sd/var/mode/entropy functionality of Bernoulli distribution.
    """
    net = BernoulliSd()
    ans = net()
    assert isinstance(ans, Tensor)


class BernoulliVar(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(BernoulliVar, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5], dtype=dtype.int32)

    def construct(self):
        var = self.b.var()
        return var


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_var():
    """
    Test mean/sd/var/mode/entropy functionality of Bernoulli distribution.
    """
    net = BernoulliVar()
    ans = net()
    assert isinstance(ans, Tensor)


class BernoulliMode(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(BernoulliMode, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5], dtype=dtype.int32)

    def construct(self):
        mode = self.b.mode()
        return mode


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_mode():
    """
    Test mean/sd/var/mode/entropy functionality of Bernoulli distribution.
    """
    net = BernoulliMode()
    ans = net()
    assert isinstance(ans, Tensor)


class BernoulliEntropy(nn.Cell):
    """
    Test class: basic mean/sd/var/mode/entropy function.
    """

    def __init__(self):
        super(BernoulliEntropy, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5], dtype=dtype.int32)

    def construct(self):
        entropy = self.b.entropy()
        return entropy


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_entropy():
    """
    Test mean/sd/var/mode/entropy functionality of Bernoulli distribution.
    """
    net = BernoulliEntropy()
    ans = net()
    assert isinstance(ans, Tensor)
