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
"""test cases for cat distribution"""
import numpy as np
import pytest
from scipy import stats
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Prob(nn.Cell):
    """
    Test class: probability of categorical distribution.
    """

    def __init__(self):
        super(Prob, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.prob(x_)


def test_pmf():
    """
    Test pmf.
    """
    expect_pmf = [0.7, 0.3, 0.7, 0.3, 0.3]
    pmf = Prob()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(
        np.int32), dtype=dtype.float32)
    output = pmf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pmf) < tol).all()


class LogProb(nn.Cell):
    """
    Test class: log probability of categorical distribution.
    """

    def __init__(self):
        super(LogProb, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.log_prob(x_)


def test_log_likelihood():
    """
    Test log_pmf.
    """
    expect_logpmf = np.log([0.7, 0.3, 0.7, 0.3, 0.3])
    logprob = LogProb()
    x_ = Tensor(np.array([0, 1, 0, 1, 1]).astype(
        np.int32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpmf) < tol).all()


class KL(nn.Cell):
    """
    Test class: kl_loss between categorical distributions.
    """

    def __init__(self):
        super(KL, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.kl_loss('Categorical', x_)


def test_kl_loss():
    """
    Test kl_loss.
    """
    kl_loss = KL()
    output = kl_loss(Tensor([0.7, 0.3], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy()) < tol).all()


class Sampling(nn.Cell):
    """
    Test class: sampling of categorical distribution.
    """

    def __init__(self):
        super(Sampling, self).__init__()
        self.c = msd.Categorical([0.2, 0.1, 0.7], dtype=dtype.int32)
        self.shape = (2, 3)

    def construct(self):
        return self.c.sample(self.shape)


def test_sample():
    """
    Test sample.
    """
    with pytest.raises(NotImplementedError):
        sample = Sampling()
        sample()


class Basics(nn.Cell):
    """
    Test class: mean/var/mode of categorical distribution.
    """

    def __init__(self):
        super(Basics, self).__init__()
        self.c = msd.Categorical([0.2, 0.1, 0.7], dtype=dtype.int32)

    def construct(self):
        return self.c.mean(), self.c.var(), self.c.mode()


def test_basics():
    """
    Test mean/variance/mode.
    """
    basics = Basics()
    mean, var, mode = basics()
    expect_mean = 0 * 0.2 + 1 * 0.1 + 2 * 0.7
    expect_var = 0 * 0.2 + 1 * 0.1 + 4 * 0.7 - (expect_mean * expect_mean)
    expect_mode = 2
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(var.asnumpy() - expect_var) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()


class CDF(nn.Cell):
    """
    Test class: cdf of categorical distributions.
    """

    def __init__(self):
        super(CDF, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    expect_cdf = [0.7, 0.7, 1, 0.7, 1]
    x_ = Tensor(np.array([0, 0, 1, 0, 1]).astype(
        np.int32), dtype=dtype.float32)
    cdf = CDF()
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()


class LogCDF(nn.Cell):
    """
    Test class: log cdf of categorical distributions.
    """

    def __init__(self):
        super(LogCDF, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.log_cdf(x_)


def test_logcdf():
    """
    Test log_cdf.
    """
    expect_logcdf = np.log([0.7, 0.7, 1, 0.7, 1])
    x_ = Tensor(np.array([0, 0, 1, 0, 1]).astype(
        np.int32), dtype=dtype.float32)
    logcdf = LogCDF()
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()


class SF(nn.Cell):
    """
    Test class: survival function of categorical distributions.
    """

    def __init__(self):
        super(SF, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.survival_function(x_)


def test_survival():
    """
    Test survival function.
    """
    expect_survival = [0.3, 0., 0., 0.3, 0.3]
    x_ = Tensor(np.array([0, 1, 1, 0, 0]).astype(
        np.int32), dtype=dtype.float32)
    sf = SF()
    output = sf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()


class LogSF(nn.Cell):
    """
    Test class: log survival function of categorical distributions.
    """

    def __init__(self):
        super(LogSF, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        return self.c.log_survival(x_)


def test_log_survival():
    """
    Test log survival function.
    """
    expect_logsurvival = np.log([1., 0.3, 0.3, 0.3, 0.3])
    x_ = Tensor(np.array([-2, 0, 0, 0.5, 0.5]
                         ).astype(np.float32), dtype=dtype.float32)
    log_sf = LogSF()
    output = log_sf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()


class EntropyH(nn.Cell):
    """
    Test class: entropy of categorical distributions.
    """

    def __init__(self):
        super(EntropyH, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self):
        return self.c.entropy()


def test_entropy():
    """
    Test entropy.
    """
    cat_benchmark = stats.multinomial(n=1, p=[0.7, 0.3])
    expect_entropy = cat_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()


class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between categorical distributions.
    """

    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.c = msd.Categorical([0.7, 0.3], dtype=dtype.int32)

    def construct(self, x_):
        entropy = self.c.entropy()
        kl_loss = self.c.kl_loss('Categorical', x_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.c.cross_entropy('Categorical', x_)
        return h_sum_kl - cross_entropy


def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    prob = Tensor([0.7, 0.3], dtype=dtype.float32)
    diff = cross_entropy(prob)
    tol = 1e-6
    assert (np.abs(diff.asnumpy()) < tol).all()
