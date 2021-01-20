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
"""test cases for Geometric distribution"""
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
    Test class: probability of Geometric distribution.
    """

    def __init__(self):
        super(Prob, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.prob(x_)


def test_pmf():
    """
    Test pmf.
    """
    geom_benchmark = stats.geom(0.7)
    expect_pmf = geom_benchmark.pmf([0, 1, 2, 3, 4]).astype(np.float32)
    pdf = Prob()
    x_ = Tensor(np.array([-1, 0, 1, 2, 3]
                         ).astype(np.float32), dtype=dtype.float32)
    output = pdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pmf) < tol).all()


class LogProb(nn.Cell):
    """
    Test class: log probability of Geometric distribution.
    """

    def __init__(self):
        super(LogProb, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.log_prob(x_)


def test_log_likelihood():
    """
    Test log_pmf.
    """
    geom_benchmark = stats.geom(0.7)
    expect_logpmf = geom_benchmark.logpmf([1, 2, 3, 4, 5]).astype(np.float32)
    logprob = LogProb()
    x_ = Tensor(np.array([0, 1, 2, 3, 4]).astype(
        np.int32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpmf) < tol).all()


class KL(nn.Cell):
    """
    Test class: kl_loss between Geometric distributions.
    """

    def __init__(self):
        super(KL, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.kl_loss('Geometric', x_)


def test_kl_loss():
    """
    Test kl_loss.
    """
    probs1_a = 0.7
    probs1_b = 0.5
    probs0_a = 1 - probs1_a
    probs0_b = 1 - probs1_b
    expect_kl_loss = np.log(probs1_a / probs1_b) + \
        (probs0_a / probs1_a) * np.log(probs0_a / probs0_b)
    kl_loss = KL()
    output = kl_loss(Tensor([probs1_b], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()


class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Geometric distribution.
    """

    def __init__(self):
        super(Basics, self).__init__()
        self.g = msd.Geometric([0.5, 0.5], dtype=dtype.int32)

    def construct(self):
        return self.g.mean(), self.g.sd(), self.g.mode()


def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()
    expect_mean = [1.0, 1.0]
    expect_sd = np.sqrt(np.array([0.5, 0.5]) / np.square(np.array([0.5, 0.5])))
    expect_mode = [0.0, 0.0]
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()


class Sampling(nn.Cell):
    """
    Test class: log probability of bernoulli distribution.
    """

    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.g = msd.Geometric([0.7, 0.5], seed=seed, dtype=dtype.int32)
        self.shape = shape

    def construct(self, probs=None):
        return self.g.sample(self.shape, probs)


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
    Test class: cdf of Geometric distribution.
    """

    def __init__(self):
        super(CDF, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    geom_benchmark = stats.geom(0.7)
    expect_cdf = geom_benchmark.cdf([0, 1, 2, 3, 4]).astype(np.float32)
    x_ = Tensor(np.array([-1, 0, 1, 2, 3]
                         ).astype(np.int32), dtype=dtype.float32)
    cdf = CDF()
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()


class LogCDF(nn.Cell):
    """
    Test class: log cdf of Geometric distribution.
    """

    def __init__(self):
        super(LogCDF, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.log_cdf(x_)


def test_logcdf():
    """
    Test log_cdf.
    """
    geom_benchmark = stats.geom(0.7)
    expect_logcdf = geom_benchmark.logcdf([1, 2, 3, 4, 5]).astype(np.float32)
    x_ = Tensor(np.array([0, 1, 2, 3, 4]).astype(
        np.int32), dtype=dtype.float32)
    logcdf = LogCDF()
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()


class SF(nn.Cell):
    """
    Test class: survial function of Geometric distribution.
    """

    def __init__(self):
        super(SF, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.survival_function(x_)


def test_survival():
    """
    Test survival function.
    """
    geom_benchmark = stats.geom(0.7)
    expect_survival = geom_benchmark.sf([0, 1, 2, 3, 4]).astype(np.float32)
    x_ = Tensor(np.array([-1, 0, 1, 2, 3]
                         ).astype(np.int32), dtype=dtype.float32)
    sf = SF()
    output = sf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()


class LogSF(nn.Cell):
    """
    Test class: log survial function of Geometric distribution.
    """

    def __init__(self):
        super(LogSF, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        return self.g.log_survival(x_)


def test_log_survival():
    """
    Test log_survival function.
    """
    geom_benchmark = stats.geom(0.7)
    expect_logsurvival = geom_benchmark.logsf(
        [0, 1, 2, 3, 4]).astype(np.float32)
    x_ = Tensor(np.array([-1, 0, 1, 2, 3]
                         ).astype(np.float32), dtype=dtype.float32)
    log_sf = LogSF()
    output = log_sf(x_)
    tol = 5e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()


class EntropyH(nn.Cell):
    """
    Test class: entropy of Geometric distribution.
    """

    def __init__(self):
        super(EntropyH, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self):
        return self.g.entropy()


def test_entropy():
    """
    Test entropy.
    """
    geom_benchmark = stats.geom(0.7)
    expect_entropy = geom_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()


class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Geometric distributions.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.g = msd.Geometric(0.7, dtype=dtype.int32)

    def construct(self, x_):
        entropy = self.g.entropy()
        kl_loss = self.g.kl_loss('Geometric', x_)
        h_sum_kl = entropy + kl_loss
        ans = self.g.cross_entropy('Geometric', x_)
        return h_sum_kl - ans


def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    prob = Tensor([0.5], dtype=dtype.float32)
    diff = cross_entropy(prob)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
