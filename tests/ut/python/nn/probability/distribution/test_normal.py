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
Test nn.probability.distribution.Normal.
"""
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") == "CPU"


def test_normal_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.Normal([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)


def test_type():
    with pytest.raises(TypeError):
        msd.Normal(0., 1., dtype=dtype.int32)


def test_name():
    with pytest.raises(TypeError):
        msd.Normal(0., 1., name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Normal(0., 1., seed='seed')


def test_sd():
    with pytest.raises(ValueError):
        msd.Normal(0., 0.)
    with pytest.raises(ValueError):
        msd.Normal(0., -1.)


def test_arguments():
    """
    args passing during initialization.
    """
    n = msd.Normal()
    assert isinstance(n, msd.Distribution)
    n = msd.Normal([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(n, msd.Distribution)


class NormalProb(nn.Cell):
    """
    Normal distribution: initialize with mean/sd.
    """

    def __init__(self):
        super(NormalProb, self).__init__()
        self.normal = msd.Normal(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.normal.prob(value)
        log_prob = self.normal.log_prob(value)
        cdf = self.normal.cdf(value)
        log_cdf = self.normal.log_cdf(value)
        sf = self.normal.survival_function(value)
        log_sf = self.normal.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_normal_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = NormalProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class NormalProb1(nn.Cell):
    """
    Normal distribution: initialize without mean/sd.
    """

    def __init__(self):
        super(NormalProb1, self).__init__()
        self.normal = msd.Normal()

    def construct(self, value, mean, sd):
        prob = self.normal.prob(value, mean, sd)
        log_prob = self.normal.log_prob(value, mean, sd)
        cdf = self.normal.cdf(value, mean, sd)
        log_cdf = self.normal.log_cdf(value, mean, sd)
        sf = self.normal.survival_function(value, mean, sd)
        log_sf = self.normal.log_survival(value, mean, sd)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_normal_prob1():
    """
    Test probability functions: passing mean/sd, value through construct.
    """
    net = NormalProb1()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mean, sd)
    assert isinstance(ans, Tensor)


class NormalKl(nn.Cell):
    """
    Test class: kl_loss of Normal distribution.
    """

    def __init__(self):
        super(NormalKl, self).__init__()
        self.n1 = msd.Normal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.n2 = msd.Normal(dtype=dtype.float32)

    def construct(self, mean_b, sd_b, mean_a, sd_a):
        kl1 = self.n1.kl_loss('Normal', mean_b, sd_b)
        kl2 = self.n2.kl_loss('Normal', mean_b, sd_b, mean_a, sd_a)
        return kl1 + kl2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_kl():
    """
    Test kl_loss.
    """
    net = NormalKl()
    mean_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    sd_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    mean_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    sd_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(mean_b, sd_b, mean_a, sd_a)
    assert isinstance(ans, Tensor)


class NormalCrossEntropy(nn.Cell):
    """
    Test class: cross_entropy of Normal distribution.
    """

    def __init__(self):
        super(NormalCrossEntropy, self).__init__()
        self.n1 = msd.Normal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)
        self.n2 = msd.Normal(dtype=dtype.float32)

    def construct(self, mean_b, sd_b, mean_a, sd_a):
        h1 = self.n1.cross_entropy('Normal', mean_b, sd_b)
        h2 = self.n2.cross_entropy('Normal', mean_b, sd_b, mean_a, sd_a)
        return h1 + h2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_cross_entropy():
    """
    Test cross entropy between Normal distributions.
    """
    net = NormalCrossEntropy()
    mean_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    sd_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    mean_a = Tensor(np.array([2.0]).astype(np.float32), dtype=dtype.float32)
    sd_a = Tensor(np.array([3.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(mean_b, sd_b, mean_a, sd_a)
    assert isinstance(ans, Tensor)


class NormalBasics(nn.Cell):
    """
    Test class: basic mean/sd function.
    """

    def __init__(self):
        super(NormalBasics, self).__init__()
        self.n = msd.Normal(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mean = self.n.mean()
        sd = self.n.sd()
        mode = self.n.mode()
        entropy = self.n.entropy()
        return mean + sd + mode + entropy


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_bascis():
    """
    Test mean/sd/mode/entropy functionality of Normal.
    """
    net = NormalBasics()
    ans = net()
    assert isinstance(ans, Tensor)


class NormalConstruct(nn.Cell):
    """
    Normal distribution: going through construct.
    """

    def __init__(self):
        super(NormalConstruct, self).__init__()
        self.normal = msd.Normal(3.0, 4.0)
        self.normal1 = msd.Normal()

    def construct(self, value, mean, sd):
        prob = self.normal('prob', value)
        prob1 = self.normal('prob', value, mean, sd)
        prob2 = self.normal1('prob', value, mean, sd)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test_normal_construct():
    """
    Test probability function going through construct.
    """
    net = NormalConstruct()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mean, sd)
    assert isinstance(ans, Tensor)
