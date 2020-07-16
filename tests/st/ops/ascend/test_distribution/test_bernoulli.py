# Copyright 2019 Huawei Technologies Co., Ltd
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
"""test cases for bernoulli distribution"""
import numpy as np
from scipy import stats
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    """
    Test class: probability of bernoulli distribution.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.b = nn.Bernoulli(0.7, dtype=dtype.int32)

    @ms_function
    def construct(self, x_):
        return self.b('prob', x_)

class Net1(nn.Cell):
    """
    Test class: log probability of bernoulli distribution.
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.b = nn.Bernoulli(0.7, dtype=dtype.int32)

    @ms_function
    def construct(self, x_):
        return self.b('log_prob', x_)

class Net2(nn.Cell):
    """
    Test class: kl_loss between bernoulli distributions.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.b = nn.Bernoulli(0.7, dtype=dtype.int32)

    @ms_function
    def construct(self, x_):
        return self.b('kl_loss', 'Bernoulli', x_)

class Net3(nn.Cell):
    """
    Test class: mean/sd of bernoulli distribution.
    """
    def __init__(self):
        super(Net3, self).__init__()
        self.b = nn.Bernoulli([0.5, 0.5], dtype=dtype.int32)

    @ms_function
    def construct(self):
        return self.b('mean'), self.b('sd')

class Net4(nn.Cell):
    """
    Test class: log probability of bernoulli distribution.
    """
    def __init__(self, shape, seed=0):
        super(Net4, self).__init__()
        self.b = nn.Bernoulli([0.7, 0.5], seed=seed, dtype=dtype.int32)
        self.shape = shape

    @ms_function
    def construct(self, probs=None):
        return self.b('sample', self.shape, probs)

def test_pmf():
    """
    Test pmf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_pmf = bernoulli_benchmark.pmf([0, 1, 0, 1, 1]).astype(np.float32)
    pdf = Net()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(np.int32), dtype=dtype.float32)
    output = pdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pmf) < tol).all()

def test_log_likelihood():
    """
    Test log_pmf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_logpmf = bernoulli_benchmark.logpmf([0, 1, 0, 1, 1]).astype(np.float32)
    logprob = Net1()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(np.int32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpmf) < tol).all()

def test_kl_loss():
    """
    Test kl_loss.
    """
    probs1_a = 0.7
    probs1_b = 0.5
    probs0_a = 1 - probs1_a
    probs0_b = 1 - probs1_b
    expect_kl_loss = probs1_a * np.log(probs1_a / probs1_b) + probs0_a * np.log(probs0_a / probs0_b)
    kl_loss = Net2()
    output = kl_loss(Tensor([probs1_b], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

def test_basics():
    """
    Test mean/standard deviation and probs.
    """
    basics = Net3()
    mean, sd = basics()
    expect_mean = [0.5, 0.5]
    assert (mean.asnumpy() == expect_mean).all()
    assert (sd.asnumpy() == expect_mean).all()
    b = nn.Bernoulli([0.7, 0.5], dtype=dtype.int32)
    probs = b.probs()
    expect_probs = [0.7, 0.5]
    tol = 1e-6
    assert (np.abs(probs.asnumpy() - expect_probs) < tol).all()

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    sample = Net4(shape)
    output = sample()
    assert output.shape == (2, 3, 2)
