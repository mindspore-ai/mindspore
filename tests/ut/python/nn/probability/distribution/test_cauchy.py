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
Test nn.probability.distribution.cauchy.
"""
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
from mindspore import context

skip_flag = context.get_context("device_target") != "Ascend"


def test_cauchy_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.Cauchy([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)


def test_type():
    with pytest.raises(TypeError):
        msd.Cauchy(0., 1., dtype=dtype.int32)


def test_name():
    with pytest.raises(TypeError):
        msd.Cauchy(0., 1., name=1.0)


def test_seed():
    with pytest.raises(TypeError):
        msd.Cauchy(0., 1., seed='seed')


def test_scale():
    with pytest.raises(ValueError):
        msd.Cauchy(0., 0.)
    with pytest.raises(ValueError):
        msd.Cauchy(0., -1.)


def test_arguments():
    """
    args passing during initialization.
    """
    l1 = msd.Cauchy()
    assert isinstance(l1, msd.Distribution)
    l2 = msd.Cauchy([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(l2, msd.Distribution)


class CauchyProb(nn.Cell):
    """
    Cauchy distribution: initialize with loc/scale.
    """
    def __init__(self):
        super(CauchyProb, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.cauchy.prob(value)
        log_prob = self.cauchy.log_prob(value)
        cdf = self.cauchy.cdf(value)
        log_cdf = self.cauchy.log_cdf(value)
        sf = self.cauchy.survival_function(value)
        log_sf = self.cauchy.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU and GPU")
def test_cauchy_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = CauchyProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)


class CauchyProb1(nn.Cell):
    """
    Cauchy distribution: initialize without loc/scale.
    """
    def __init__(self):
        super(CauchyProb1, self).__init__()
        self.cauchy = msd.Cauchy()

    def construct(self, value, mu, s):
        prob = self.cauchy.prob(value, mu, s)
        log_prob = self.cauchy.log_prob(value, mu, s)
        cdf = self.cauchy.cdf(value, mu, s)
        log_cdf = self.cauchy.log_cdf(value, mu, s)
        sf = self.cauchy.survival_function(value, mu, s)
        log_sf = self.cauchy.log_survival(value, mu, s)
        return prob + log_prob + cdf + log_cdf + sf + log_sf


@pytest.mark.skipif(skip_flag, reason="not support running in CPU and GPU")
def test_cauchy_prob1():
    """
    Test probability functions: passing loc/scale, value through construct.
    """
    net = CauchyProb1()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mu = Tensor([0.0], dtype=dtype.float32)
    s = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mu, s)
    assert isinstance(ans, Tensor)


class KL(nn.Cell):
    """
    Test kl_loss and cross entropy.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0)
        self.cauchy1 = msd.Cauchy()

    def construct(self, mu, s, mu_a, s_a):
        kl = self.cauchy.kl_loss('Cauchy', mu, s)
        kl1 = self.cauchy1.kl_loss('Cauchy', mu, s, mu_a, s_a)
        cross_entropy = self.cauchy.cross_entropy('Cauchy', mu, s)
        cross_entropy1 = self.cauchy.cross_entropy('Cauchy', mu, s, mu_a, s_a)
        return kl + kl1 + cross_entropy + cross_entropy1


@pytest.mark.skipif(skip_flag, reason="not support running in CPU and GPU")
def test_kl_cross_entropy():
    """
    Test kl_loss and cross_entropy.
    """
    net = KL()
    mu = Tensor([0.0], dtype=dtype.float32)
    s = Tensor([1.0], dtype=dtype.float32)
    mu_a = Tensor([0.0], dtype=dtype.float32)
    s_a = Tensor([1.0], dtype=dtype.float32)
    ans = net(mu, s, mu_a, s_a)
    assert isinstance(ans, Tensor)


class CauchyBasics(nn.Cell):
    """
    Test class: basic loc/scale function.
    """
    def __init__(self):
        super(CauchyBasics, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mode = self.cauchy.mode()
        entropy = self.cauchy.entropy()
        return mode + entropy


class CauchyMean(nn.Cell):
    """
    Test class: basic loc/scale function.
    """
    def __init__(self):
        super(CauchyMean, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        return self.cauchy.mean()


class CauchyVar(nn.Cell):
    """
    Test class: basic loc/scale function.
    """
    def __init__(self):
        super(CauchyVar, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        return self.cauchy.var()


class CauchySd(nn.Cell):
    """
    Test class: basic loc/scale function.
    """
    def __init__(self):
        super(CauchySd, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        return self.cauchy.sd()


@pytest.mark.skipif(skip_flag, reason="not support running in CPU and GPU")
def test_bascis():
    """
    Test mean/sd/var/mode/entropy functionality of Cauchy.
    """
    net = CauchyBasics()
    ans = net()
    assert isinstance(ans, Tensor)
    with pytest.raises(ValueError):
        net = CauchyMean()
        ans = net()
    with pytest.raises(ValueError):
        net = CauchyVar()
        ans = net()
    with pytest.raises(ValueError):
        net = CauchySd()
        ans = net()


class CauchyConstruct(nn.Cell):
    """
    Cauchy distribution: going through construct.
    """
    def __init__(self):
        super(CauchyConstruct, self).__init__()
        self.cauchy = msd.Cauchy(3.0, 4.0)
        self.cauchy1 = msd.Cauchy()

    def construct(self, value, mu, s):
        prob = self.cauchy('prob', value)
        prob1 = self.cauchy('prob', value, mu, s)
        prob2 = self.cauchy1('prob', value, mu, s)
        return prob + prob1 + prob2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU and GPU")
def test_cauchy_construct():
    """
    Test probability function going through construct.
    """
    net = CauchyConstruct()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mu = Tensor([0.0], dtype=dtype.float32)
    s = Tensor([1.0], dtype=dtype.float32)
    ans = net(value, mu, s)
    assert isinstance(ans, Tensor)
