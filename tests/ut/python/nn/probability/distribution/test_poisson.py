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
Test nn.probability.distribution.Poisson.
"""
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") != "Ascend"


def test_arguments():
    """
    Args passing during initialization.
    """
    p = msd.Poisson()
    assert isinstance(p, msd.Distribution)
    p = msd.Poisson([0.1, 0.3, 0.5, 1.0], dtype=dtype.float32)
    assert isinstance(p, msd.Distribution)


def test_type():
    with pytest.raises(TypeError):
        msd.Poisson([0.1], dtype=dtype.bool_)


def test_name():
    with pytest.raises(TypeError):
        msd.Poisson([0.1], name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Poisson([0.1], seed='seed')


def test_rate():
    """
    Invalid rate.
    """
    with pytest.raises(ValueError):
        msd.Poisson([-0.1], dtype=dtype.float32)
    with pytest.raises(ValueError):
        msd.Poisson([0.0], dtype=dtype.float32)


def test_scalar():
    with pytest.raises(TypeError):
        msd.Poisson(0.1, seed='seed')


class PoissonProb(nn.Cell):
    """
    Poisson distribution: initialize with rate.
    """

    def __init__(self):
        super(PoissonProb, self).__init__()
        self.p = msd.Poisson([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype.float32)

    def construct(self, value):
        prob = self.p.prob(value)
        log_prob = self.p.log_prob(value)
        cdf = self.p.cdf(value)
        log_cdf = self.p.log_cdf(value)
        sf = self.p.survival_function(value)
        log_sf = self.p.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_poisson_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = PoissonProb()
    value = Tensor([0.2, 0.3, 5.0, 2, 3.9], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class PoissonProb1(nn.Cell):
    """
    Poisson distribution: initialize without rate.
    """

    def __init__(self):
        super(PoissonProb1, self).__init__()
        self.p = msd.Poisson(dtype=dtype.float32)

    def construct(self, value, rate):
        prob = self.p.prob(value, rate)
        log_prob = self.p.log_prob(value, rate)
        cdf = self.p.cdf(value, rate)
        log_cdf = self.p.log_cdf(value, rate)
        sf = self.p.survival_function(value, rate)
        log_sf = self.p.log_survival(value, rate)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_poisson_prob1():
    """
    Test probability functions: passing value/rate through construct.
    """
    net = PoissonProb1()
    value = Tensor([0.2, 0.9, 1, 2, 3], dtype=dtype.float32)
    rate = Tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype.float32)
    ans = net(value, rate)
    assert isinstance(ans, Tensor)


class PoissonBasics(nn.Cell):
    """
    Test class: basic mean/sd/var/mode function.
    """

    def __init__(self):
        super(PoissonBasics, self).__init__()
        self.p = msd.Poisson([2.3, 2.5], dtype=dtype.float32)

    def construct(self):
        mean = self.p.mean()
        sd = self.p.sd()
        var = self.p.var()
        return mean + sd + var


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bascis():
    """
    Test mean/sd/var/mode functionality of Poisson distribution.
    """
    net = PoissonBasics()
    ans = net()
    assert isinstance(ans, Tensor)


class PoissonConstruct(nn.Cell):
    """
    Poisson distribution: going through construct.
    """

    def __init__(self):
        super(PoissonConstruct, self).__init__()
        self.p = msd.Poisson([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype.float32)
        self.p1 = msd.Poisson(dtype=dtype.float32)

    def construct(self, value, rate):
        prob = self.p('prob', value)
        prob1 = self.p('prob', value, rate)
        prob2 = self.p1('prob', value, rate)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_poisson_construct():
    """
    Test probability function going through construct.
    """
    net = PoissonConstruct()
    value = Tensor([0, 0, 0, 0, 0], dtype=dtype.float32)
    probs = Tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=dtype.float32)
    ans = net(value, probs)
    assert isinstance(ans, Tensor)
