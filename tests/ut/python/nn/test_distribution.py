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
    b = nn.Bernoulli()
    print(n)
    print(b)

def test_with_arguments():
    """
    Args passed in during initialization.
    """
    n = nn.Normal([3.0], [4.0], dtype=dtype.float32)
    b = nn.Bernoulli([0.3, 0.5], dtype=dtype.int32)
    print(n)
    print(b)

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
    print("pdf: ", pdf)
    print("log_pdf: ", log_pdf)

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
    print("pdf: ", pdf)
    print("log_pdf: ", log_pdf)


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
    print("pdf: ", pdf)
    print("log_pdf: ", log_pdf)

class BernoulliProb(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliProb, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5, dtype=dtype.int32)

    def construct(self, value):
        x = self.bernoulli('prob', value)
        y = self.bernoulli('log_prob', value)
        return x, y

def test_bernoulli_prob():
    """
    Test pmf/log_pmf: passing value through construct.
    """
    net = BernoulliProb()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    ans = net(value)
    print("pmf: ", ans)
    print("log_pmf: ", ans)


class BernoulliProb1(nn.Cell):
    """
    Bernoulli distribution: initialize without probs.
    """
    def __init__(self):
        super(BernoulliProb1, self).__init__()
        self.bernoulli = nn.Bernoulli()

    def construct(self, value, probs):
        x = self.bernoulli('prob', value, probs)
        y = self.bernoulli('log_prob', value, probs)
        return x, y

def test_bernoulli_prob1():
    """
    Test pmf/log_pmf: passing probs through construct.
    """
    net = BernoulliProb1()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    ans = net(value, probs)
    print("pmf: ", ans)
    print("log_pmf: ", ans)


class BernoulliProb2(nn.Cell):
    """
    Bernoulli distribution: initialize with probs.
    """
    def __init__(self):
        super(BernoulliProb2, self).__init__()
        self.bernoulli = nn.Bernoulli(0.5)

    def construct(self, value, probs):
        x = self.bernoulli('prob', value, probs)
        y = self.bernoulli('log_prob', value, probs)
        return x, y

def test_bernoulli_prob2():
    """
    Test pmf/log_pmf: passing probs/value through construct.
    Overwrite original probs.
    """
    net = BernoulliProb2()
    value = Tensor([1, 0, 1, 0, 1], dtype=dtype.float32)
    probs = Tensor([0.3], dtype=dtype.float32)
    ans = net(value, probs)
    print("pmf: ", ans)
    print("log_pmf: ", ans)

class NormalKl(nn.Cell):
    """
    Test class: kl_loss of Normal distribution.
    """
    def __init__(self):
        super(NormalKl, self).__init__()
        self.n = nn.Normal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)

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
    output = nor_net(mean, sd)
    print("normal-normal kl loss: ", output)

    ber_net = BernoulliKl()
    probs_b = Tensor([0.3], dtype=dtype.float32)
    output = ber_net(probs_b)
    print("bernoulli-bernoulli kl loss: ", output)


class NormalBernoulli(nn.Cell):
    """
    Test class: basic mean/sd function.
    """
    def __init__(self):
        super(NormalBernoulli, self).__init__()
        self.n = nn.Normal(3.0, 4.0, dtype=dtype.int32)
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
    print("Mean of Normal distribution: ", normal_mean)
    print("Standard deviation of Normal distribution: ", normal_sd)
    print("Mean of Bernoulli distribution: ", bernoulli_mean)
    print("Standard deviation of Bernoulli distribution: ", bernoulli_sd)
