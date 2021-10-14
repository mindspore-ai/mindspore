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
Test nn.probability.distribution.logistic.
"""
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") == "CPU"


def test_logistic_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.Logistic([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)


def test_type():
    with pytest.raises(TypeError):
        msd.Logistic(0., 1., dtype=dtype.int32)


def test_name():
    with pytest.raises(TypeError):
        msd.Logistic(0., 1., name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Logistic(0., 1., seed='seed')


def test_scale():
    with pytest.raises(ValueError):
        msd.Logistic(0., 0.)
    with pytest.raises(ValueError):
        msd.Logistic(0., -1.)


def test_arguments():
    """
    args passing during initialization.
    """
    l = msd.Logistic()
    assert isinstance(l, msd.Distribution)
    l = msd.Logistic([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(l, msd.Distribution)


class LogisticProb(nn.Cell):
    """
    logistic distribution: initialize with loc/scale.
    """

    def __init__(self):
        super(LogisticProb, self).__init__()
        self.logistic = msd.Logistic(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.logistic.prob(value)
        log_prob = self.logistic.log_prob(value)
        cdf = self.logistic.cdf(value)
        log_cdf = self.logistic.log_cdf(value)
        sf = self.logistic.survival_function(value)
        log_sf = self.logistic.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_logistic_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = LogisticProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class LogisticProb1(nn.Cell):
    """
    logistic distribution: initialize without loc/scale.
    """

    def __init__(self):
        super(LogisticProb1, self).__init__()
        self.logistic = msd.Logistic()

    def construct(self, value, mu, s):
        prob = self.logistic.prob(value, mu, s)
        log_prob = self.logistic.log_prob(value, mu, s)
        cdf = self.logistic.cdf(value, mu, s)
        log_cdf = self.logistic.log_cdf(value, mu, s)
        sf = self.logistic.survival_function(value, mu, s)
        log_sf = self.logistic.log_survival(value, mu, s)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_logistic_prob1():
    """
    Test probability functions: passing loc/scale, value through construct.
    """
    net = LogisticProb1()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mu = Tensor([0.0], dtype=dtype.float32)
    s = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mu, s)
    assert isinstance(ans, Tensor)


class KL(nn.Cell):
    """
    Test kl_loss. Should raise NotImplementedError.
    """

    def __init__(self):
        super(KL, self).__init__()
        self.logistic = msd.Logistic(3.0, 4.0)

    def construct(self, mu, s):
        kl = self.logistic.kl_loss('Logistic', mu, s)
        return kl


class Crossentropy(nn.Cell):
    """
    Test cross entropy. Should raise NotImplementedError.
    """

    def __init__(self):
        super(Crossentropy, self).__init__()
        self.logistic = msd.Logistic(3.0, 4.0)

    def construct(self, mu, s):
        cross_entropy = self.logistic.cross_entropy('Logistic', mu, s)
        return cross_entropy


class LogisticBasics(nn.Cell):
    """
    Test class: basic loc/scale function.
    """

    def __init__(self):
        super(LogisticBasics, self).__init__()
        self.logistic = msd.Logistic(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mean = self.logistic.mean()
        sd = self.logistic.sd()
        mode = self.logistic.mode()
        entropy = self.logistic.entropy()
        return mean + sd + mode + entropy


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bascis():
    """
    Test mean/sd/mode/entropy functionality of logistic.
    """
    net = LogisticBasics()
    ans = net()
    assert isinstance(ans, Tensor)
    mu = Tensor(1.0, dtype=dtype.float32)
    s = Tensor(1.0, dtype=dtype.float32)
    with pytest.raises(NotImplementedError):
        kl = KL()
        ans = kl(mu, s)
    with pytest.raises(NotImplementedError):
        crossentropy = Crossentropy()
        ans = crossentropy(mu, s)


class LogisticConstruct(nn.Cell):
    """
    logistic distribution: going through construct.
    """

    def __init__(self):
        super(LogisticConstruct, self).__init__()
        self.logistic = msd.Logistic(3.0, 4.0)
        self.logistic1 = msd.Logistic()

    def construct(self, value, mu, s):
        prob = self.logistic('prob', value)
        prob1 = self.logistic('prob', value, mu, s)
        prob2 = self.logistic1('prob', value, mu, s)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_logistic_construct():
    """
    Test probability function going through construct.
    """
    net = LogisticConstruct()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mu = Tensor([0.0], dtype=dtype.float32)
    s = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mu, s)
    assert isinstance(ans, Tensor)
