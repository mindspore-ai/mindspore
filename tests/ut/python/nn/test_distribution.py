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
Test nn.Distribution.

Including Normal Distribution and Bernoulli Distribution.
"""
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import dtype
from mindspore import Tensor

def test_normal_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        nn.Normal([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)

def test_no_arguments():
    """
    No args passed in during initialization.
    """
    n = nn.Normal()
    assert isinstance(n, nn.Distribution)
    b = nn.Bernoulli()
    assert isinstance(b, nn.Distribution)

def test_with_arguments():
    """
    Args passed in during initialization.
    """
    n = nn.Normal([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(n, nn.Distribution)
    b = nn.Bernoulli([0.3, 0.5], dtype=dtype.int32)
    assert isinstance(b, nn.Distribution)

class NormalProb(nn.Cell):
    """
    Normal distribution: initialize with mean/sd.
    """
    def __init__(self):
        super(NormalProb, self).__init__()
        self.normal = nn.Normal(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        x = self.normal('prob', value)
        y = self.normal('log_prob', value)
        return x, y

def test_normal_prob():
    """
    Test pdf/log_pdf: passing value through construct.
    """
    net = NormalProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    pdf, log_pdf = net(value)
    assert isinstance(pdf, Tensor)
    assert isinstance(log_pdf, Tensor)

class NormalProb1(nn.Cell):
    """
    Normal distribution: initialize without mean/sd.
    """
    def __init__(self):
        super(NormalProb1, self).__init__()
        self.normal = nn.Normal()

    def construct(self, value, mean, sd):
        x = self.normal('prob', value, mean, sd)
        y = self.normal('log_prob', value, mean, sd)
        return x, y

def test_normal_prob1():
    """
    Test pdf/logpdf: passing mean/sd, value through construct.
    """
    net = NormalProb1()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    pdf, log_pdf = net(value, mean, sd)
    assert isinstance(pdf, Tensor)
    assert isinstance(log_pdf, Tensor)

class NormalProb2(nn.Cell):
    """
    Normal distribution: initialize with mean/sd.
    """
    def __init__(self):
        super(NormalProb2, self).__init__()
        self.normal = nn.Normal(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value, mean, sd):
        x = self.normal('prob', value, mean, sd)
        y = self.normal('log_prob', value, mean, sd)
        return x, y

def test_normal_prob2():
    """
    Test pdf/log_pdf: passing mean/sd through construct.
    Overwrite original mean/sd.
    """
    net = NormalProb2()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    mean = Tensor([0.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    pdf, log_pdf = net(value, mean, sd)
    assert isinstance(pdf, Tensor)
    assert isinstance(log_pdf, Tensor)

class BernoulliProb(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliProb, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5, dtype=dtype.int32)

    def construct(self, value):
        return self.bernoulli('prob', value)

class BernoulliLogProb(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliLogProb, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5, dtype=dtype.int32)

    def construct(self, value):
        return self.bernoulli('log_prob', value)


def test_bernoulli_prob():
    """
    Test pmf/log_pmf: passing value through construct.
    """
    net = BernoulliProb()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    pmf = net(value)
    assert isinstance(pmf, Tensor)

def test_bernoulli_log_prob():
    """
    Test pmf/log_pmf: passing value through construct.
    """
    net = BernoulliLogProb()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    log_pmf = net(value)
    assert isinstance(log_pmf, Tensor)

class BernoulliProb1(nn.Cell):
    """
    Bernoulli distribution: initialize without probs.
    """
    def __init__(self):
        super(BernoulliProb1, self).__init__()
        self.bernoulli = nn.Bernoulli()

    def construct(self, value, probs):
        return self.bernoulli('prob', value, probs)

class BernoulliLogProb1(nn.Cell):
    """
    Bernoulli distribution: initialize without probs.
    """
    def __init__(self):
        super(BernoulliLogProb1, self).__init__()
        self.bernoulli = nn.Bernoulli()

    def construct(self, value, probs):
        return self.bernoulli('log_prob', value, probs)


def test_bernoulli_prob1():
    """
    Test pmf/log_pmf: passing probs through construct.
    """
    net = BernoulliProb1()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    pmf = net(value, probs)
    assert isinstance(pmf, Tensor)

def test_bernoulli_log_prob1():
    """
    Test pmf/log_pmf: passing probs through construct.
    """
    net = BernoulliLogProb1()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    log_pmf = net(value, probs)
    assert isinstance(log_pmf, Tensor)

class BernoulliProb2(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliProb2, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5)

    def construct(self, value, probs):
        return self.bernoulli('prob', value, probs)

class BernoulliLogProb2(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliLogProb2, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5)

    def construct(self, value, probs):
        return self.bernoulli('log_prob', value, probs)


def test_bernoulli_prob2():
    """
    Test pmf/log_pmf: passing probs/value through construct.
    Overwrite original probs.
    """
    net = BernoulliProb2()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    pmf = net(value, probs)
    assert isinstance(pmf, Tensor)

def test_bernoulli_log_prob2():
    """
    Test pmf/log_pmf: passing probs/value through construct.
    Overwrite original probs.
    """
    net = BernoulliLogProb2()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    log_pmf = net(value, probs)
    assert isinstance(log_pmf, Tensor)


class NormalKl(nn.Cell):
    """
    Test class: kl_loss of Normal distribution.
    """
    def __init__(self):
        super(NormalKl, self).__init__()
        self.n = nn.Normal(Tensor([3.0]), Tensor([4.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        return self.n('kl_loss', 'Normal', x_, y_)

class BernoulliKl(nn.Cell):
    """
    Test class: kl_loss between Bernoulli distributions.
    """
    def __init__(self):
        super(BernoulliKl, self).__init__()
        self.b = nn.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b('kl_loss', 'Bernoulli', x_)

def test_kl():
    """
    Test kl_loss function.
    """
    nor_net = NormalKl()
    mean_b = np.array([1.0]).astype(np.float32)
    sd_b = np.array([1.0]).astype(np.float32)
    mean = Tensor(mean_b, dtype=dtype.float32)
    sd = Tensor(sd_b, dtype=dtype.float32)
    loss = nor_net(mean, sd)
    assert isinstance(loss, Tensor)

    ber_net = BernoulliKl()
    probs_b = Tensor([0.3], dtype=dtype.float32)
    loss = ber_net(probs_b)
    assert isinstance(loss, Tensor)


class NormalKlNoArgs(nn.Cell):
    """
    Test class: kl_loss of Normal distribution.
    No args during initialization.
    """
    def __init__(self):
        super(NormalKlNoArgs, self).__init__()
        self.n = nn.Normal(dtype=dtype.float32)

    def construct(self, x_, y_, w_, v_):
        return self.n('kl_loss', 'Normal', x_, y_, w_, v_)

class BernoulliKlNoArgs(nn.Cell):
    """
    Test class: kl_loss between Bernoulli distributions.
    No args during initialization.
    """
    def __init__(self):
        super(BernoulliKlNoArgs, self).__init__()
        self.b = nn.Bernoulli(dtype=dtype.int32)

    def construct(self, x_, y_):
        return self.b('kl_loss', 'Bernoulli', x_, y_)

def test_kl_no_args():
    """
    Test kl_loss function.
    """
    nor_net = NormalKlNoArgs()
    mean_b = np.array([1.0]).astype(np.float32)
    sd_b = np.array([1.0]).astype(np.float32)
    mean_a = np.array([2.0]).astype(np.float32)
    sd_a = np.array([3.0]).astype(np.float32)
    mean_b = Tensor(mean_b, dtype=dtype.float32)
    sd_b = Tensor(sd_b, dtype=dtype.float32)
    mean_a = Tensor(mean_a, dtype=dtype.float32)
    sd_a = Tensor(sd_a, dtype=dtype.float32)
    loss = nor_net(mean_b, sd_b, mean_a, sd_a)
    assert isinstance(loss, Tensor)

    ber_net = BernoulliKlNoArgs()
    probs_b = Tensor([0.3], dtype=dtype.float32)
    probs_a = Tensor([0.7], dtype=dtype.float32)
    loss = ber_net(probs_b, probs_a)
    assert isinstance(loss, Tensor)



class NormalBernoulli(nn.Cell):
    """
    Test class: basic mean/sd function.
    """
    def __init__(self):
        super(NormalBernoulli, self).__init__()
        self.n = nn.Normal(3.0, 4.0, dtype=dtype.float32)
        self.b = nn.Bernoulli(0.5, dtype=dtype.int32)

    def construct(self):
        normal_mean = self.n('mean')
        normal_sd = self.n('sd')
        bernoulli_mean = self.b('mean')
        bernoulli_sd = self.b('sd')
        return normal_mean, normal_sd, bernoulli_mean, bernoulli_sd

def test_bascis():
    """
    Test mean/sd functionality of Normal and Bernoulli.
    """
    net = NormalBernoulli()
    normal_mean, normal_sd, bernoulli_mean, bernoulli_sd = net()
    assert isinstance(normal_mean, Tensor)
    assert isinstance(normal_sd, Tensor)
    assert isinstance(bernoulli_mean, Tensor)
    assert isinstance(bernoulli_sd, Tensor)
