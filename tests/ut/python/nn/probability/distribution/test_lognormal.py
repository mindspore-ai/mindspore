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
Test nn.probability.distribution.LogNormal.
"""
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") == "CPU"


def test_lognormal_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.LogNormal([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)


def test_type():
    with pytest.raises(TypeError):
        msd.LogNormal(0., 1., dtype=dtype.int32)


def test_name():
    with pytest.raises(TypeError):
        msd.LogNormal(0., 1., name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.LogNormal(0., 1., seed='seed')


def test_sd():
    with pytest.raises(ValueError):
        msd.LogNormal(0., 0.)
    with pytest.raises(ValueError):
        msd.LogNormal(0., -1.)


def test_arguments():
    """
    args passing during initialization.
    """
    n = msd.LogNormal()
    assert isinstance(n, msd.Distribution)
    n = msd.LogNormal([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(n, msd.Distribution)


class LogNormalProb(nn.Cell):
    """
    LogNormal distribution: initialize with mean/sd.
    """

    def __init__(self):
        super(LogNormalProb, self).__init__()
        self.lognormal = msd.LogNormal(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.lognormal.prob(value)
        log_prob = self.lognormal.log_prob(value)
        cdf = self.lognormal.cdf(value)
        log_cdf = self.lognormal.log_cdf(value)
        sf = self.lognormal.survival_function(value)
        log_sf = self.lognormal.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_lognormal_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = LogNormalProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
class LogNormalProb1(nn.Cell):
    """
    LogNormal distribution: initialize without mean/sd.
    """

    def __init__(self):
        super(LogNormalProb1, self).__init__()
        self.lognormal = msd.LogNormal()

    def construct(self, value, mean, sd):
        prob = self.lognormal.prob(value, mean, sd)
        log_prob = self.lognormal.log_prob(value, mean, sd)
        cdf = self.lognormal.cdf(value, mean, sd)
        log_cdf = self.lognormal.log_cdf(value, mean, sd)
        sf = self.lognormal.survival_function(value, mean, sd)
        log_sf = self.lognormal.log_survival(value, mean, sd)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_lognormal_prob1():
    """
    Test probability functions: passing mean/sd, value through construct.
    """
    net = LogNormalProb1()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mean, sd)
    assert isinstance(ans, Tensor)


class LogNormalKl(nn.Cell):
    """
    Test class: kl_loss of LogNormal distribution.
    """

    def __init__(self):
        super(LogNormalKl, self).__init__()
        self.n1 = msd.LogNormal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.n2 = msd.LogNormal(dtype=dtype.float32)

    def construct(self, mean_b, sd_b, mean_a, sd_a):
        kl1 = self.n1.kl_loss('LogNormal', mean_b, sd_b)
        kl2 = self.n2.kl_loss('LogNormal', mean_b, sd_b, mean_a, sd_a)
        return kl1 + kl2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_kl():
    """
    Test kl_loss.
    """
    net = LogNormalKl()
    mean_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    sd_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    mean_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    sd_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(mean_b, sd_b, mean_a, sd_a)
    assert isinstance(ans, Tensor)


class LogNormalCrossEntropy(nn.Cell):
    """
    Test class: cross_entropy of LogNormal distribution.
    """

    def __init__(self):
        super(LogNormalCrossEntropy, self).__init__()
        self.n1 = msd.LogNormal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.n2 = msd.LogNormal(dtype=dtype.float32)

    def construct(self, mean_b, sd_b, mean_a, sd_a):
        h1 = self.n1.cross_entropy('LogNormal', mean_b, sd_b)
        h2 = self.n2.cross_entropy('LogNormal', mean_b, sd_b, mean_a, sd_a)
        return h1 + h2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_cross_entropy():
    """
    Test cross entropy between LogNormal distributions.
    """
    net = LogNormalCrossEntropy()
    mean_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    sd_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    mean_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    sd_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(mean_b, sd_b, mean_a, sd_a)
    assert isinstance(ans, Tensor)


class LogNormalBasics(nn.Cell):
    """
    Test class: basic mean/sd function.
    """

    def __init__(self):
        super(LogNormalBasics, self).__init__()
        self.n = msd.LogNormal(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mean = self.n.mean()
        mode = self.n.mode()
        entropy = self.n.entropy()
        return mean + mode + entropy


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bascis():
    """
    Test mean/sd/mode/entropy functionality of LogNormal.
    """
    context.set_context(device_target="Ascend")
    net = LogNormalBasics()
    ans = net()
    assert isinstance(ans, Tensor)


class LogNormalConstruct(nn.Cell):
    """
    LogNormal distribution: going through construct.
    """

    def __init__(self):
        super(LogNormalConstruct, self).__init__()
        self.lognormal = msd.LogNormal(3.0, 4.0)
        self.lognormal1 = msd.LogNormal()

    def construct(self, value, mean, sd):
        prob = self.lognormal('prob', value)
        prob1 = self.lognormal('prob', value, mean, sd)
        prob2 = self.lognormal1('prob', value, mean, sd)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_lognormal_construct():
    """
    Test probability function going through construct.
    """
    net = LogNormalConstruct()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mean, sd)
    assert isinstance(ans, Tensor)
