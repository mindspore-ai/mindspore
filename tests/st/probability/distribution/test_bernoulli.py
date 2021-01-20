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
"""test cases for Bernoulli distribution"""
import numpy as np
from scipy import stats
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Prob(nn.Cell):
    """
    Test class: probability of Bernoulli distribution.
    """

    def __init__(self):
        super(Prob, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.prob(x_)


def test_pmf():
    """
    Test pmf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_pmf = bernoulli_benchmark.pmf([0, 1, 0, 1, 1]).astype(np.float32)
    pmf = Prob()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(
        np.int32), dtype=dtype.float32)
    output = pmf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pmf) < tol).all()


class LogProb(nn.Cell):
    """
    Test class: log probability of Bernoulli distribution.
    """

    def __init__(self):
        super(LogProb, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.log_prob(x_)


def test_log_likelihood():
    """
    Test log_pmf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_logpmf = bernoulli_benchmark.logpmf(
        [0, 1, 0, 1, 1]).astype(np.float32)
    logprob = LogProb()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(
        np.int32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpmf) < tol).all()


class KL(nn.Cell):
    """
    Test class: kl_loss between Bernoulli distributions.
    """

    def __init__(self):
        super(KL, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.kl_loss('Bernoulli', x_)


def test_kl_loss():
    """
    Test kl_loss.
    """
    probs1_a = 0.7
    probs1_b = 0.5
    probs0_a = 1 - probs1_a
    probs0_b = 1 - probs1_b
    expect_kl_loss = probs1_a * \
        np.log(probs1_a / probs1_b) + probs0_a * np.log(probs0_a / probs0_b)
    kl_loss = KL()
    output = kl_loss(Tensor([probs1_b], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()


class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Bernoulli distribution.
    """

    def __init__(self):
        super(Basics, self).__init__()
        self.b = msd.Bernoulli([0.3, 0.5, 0.7], dtype=dtype.int32)

    def construct(self):
        return self.b.mean(), self.b.sd(), self.b.mode()


def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()
    expect_mean = [0.3, 0.5, 0.7]
    expect_sd = np.sqrt(np.multiply([0.7, 0.5, 0.3], [0.3, 0.5, 0.7]))
    expect_mode = [0.0, 0.0, 1.0]
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()


class Sampling(nn.Cell):
    """
    Test class: log probability of Bernoulli distribution.
    """

    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.b = msd.Bernoulli([0.7, 0.5], seed=seed, dtype=dtype.int32)
        self.shape = shape

    def construct(self, probs=None):
        return self.b.sample(self.shape, probs)


def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    sample = Sampling(shape)
    output = sample()
    assert output.shape == (2, 3, 2)


class CDF(nn.Cell):
    """
    Test class: cdf of bernoulli distributions.
    """

    def __init__(self):
        super(CDF, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_cdf = bernoulli_benchmark.cdf([0, 0, 1, 0, 1]).astype(np.float32)
    x_ = Tensor(np.array([0, 0, 1, 0, 1]).astype(
        np.int32), dtype=dtype.float32)
    cdf = CDF()
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()


class LogCDF(nn.Cell):
    """
    Test class: log cdf of  bernoulli distributions.
    """

    def __init__(self):
        super(LogCDF, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.log_cdf(x_)


def test_logcdf():
    """
    Test log_cdf.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_logcdf = bernoulli_benchmark.logcdf(
        [0, 0, 1, 0, 1]).astype(np.float32)
    x_ = Tensor(np.array([0, 0, 1, 0, 1]).astype(
        np.int32), dtype=dtype.float32)
    logcdf = LogCDF()
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()


class SF(nn.Cell):
    """
    Test class: survival function of Bernoulli distributions.
    """

    def __init__(self):
        super(SF, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.survival_function(x_)


def test_survival():
    """
    Test survival function.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_survival = bernoulli_benchmark.sf(
        [0, 1, 1, 0, 0]).astype(np.float32)
    x_ = Tensor(np.array([0, 1, 1, 0, 0]).astype(
        np.int32), dtype=dtype.float32)
    sf = SF()
    output = sf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()


class LogSF(nn.Cell):
    """
    Test class: log survival function of Bernoulli distributions.
    """

    def __init__(self):
        super(LogSF, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.b.log_survival(x_)


def test_log_survival():
    """
    Test log survival function.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_logsurvival = bernoulli_benchmark.logsf(
        [-1, 0.9, 0, 0, 0]).astype(np.float32)
    x_ = Tensor(np.array([-1, 0.9, 0, 0, 0]
                         ).astype(np.float32), dtype=dtype.float32)
    log_sf = LogSF()
    output = log_sf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()


class EntropyH(nn.Cell):
    """
    Test class: entropy of Bernoulli distributions.
    """

    def __init__(self):
        super(EntropyH, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self):
        return self.b.entropy()


def test_entropy():
    """
    Test entropy.
    """
    bernoulli_benchmark = stats.bernoulli(0.7)
    expect_entropy = bernoulli_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()


class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between bernoulli distributions.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.b = msd.Bernoulli(0.7, dtype=dtype.int32)

    def construct(self, x_):
        entropy = self.b.entropy()
        kl_loss = self.b.kl_loss('Bernoulli', x_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.b.cross_entropy('Bernoulli', x_)
        return h_sum_kl - cross_entropy


def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    prob = Tensor([0.3], dtype=dtype.float32)
    diff = cross_entropy(prob)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
