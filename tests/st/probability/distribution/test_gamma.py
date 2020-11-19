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
"""test cases for Gamma distribution"""
import numpy as np
from scipy import stats
from scipy import special
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Prob(nn.Cell):
    """
    Test class: probability of Gamma distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_pdf = gamma_benchmark.pdf([1.0, 2.0]).astype(np.float32)
    pdf = Prob()
    output = pdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Gamma distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_logpdf = gamma_benchmark.logpdf([1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    output = logprob(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()


class KL(nn.Cell):
    """
    Test class: kl_loss of Gamma distribution.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        return self.g.kl_loss('Gamma', x_, y_)


def test_kl_loss():
    """
    Test kl_loss.
    """
    concentration_a = np.array([3.0]).astype(np.float32)
    rate_a = np.array([4.0]).astype(np.float32)

    concentration_b = np.array([1.0]).astype(np.float32)
    rate_b = np.array([1.0]).astype(np.float32)

    expect_kl_loss = (concentration_a - concentration_b) * special.digamma(concentration_a) \
                     + special.gammaln(concentration_b) - special.gammaln(concentration_a) \
                     + concentration_b * np.log(rate_a) - concentration_b * np.log(rate_b) \
                     + concentration_a * (rate_b / rate_a - 1.)

    kl_loss = KL()
    concentration = Tensor(concentration_b, dtype=dtype.float32)
    rate = Tensor(rate_b, dtype=dtype.float32)
    output = kl_loss(concentration, rate)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Gamma distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self):
        return self.g.mean(), self.g.sd(), self.g.mode()

def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_mean = gamma_benchmark.mean().astype(np.float32)
    expect_sd = gamma_benchmark.std().astype(np.float32)
    expect_mode = [2.0]
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Gamma distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, concentration=None, rate=None):
        return self.g.sample(self.shape, concentration, rate)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    concentration = Tensor([2.0], dtype=dtype.float32)
    rate = Tensor([2.0, 2.0, 2.0], dtype=dtype.float32)
    sample = Sampling(shape, seed=seed)
    output = sample(concentration, rate)
    assert output.shape == (2, 3, 3)

class CDF(nn.Cell):
    """
    Test class: cdf of Gamma distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_cdf = gamma_benchmark.cdf([2.0]).astype(np.float32)
    cdf = CDF()
    output = cdf(Tensor([2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Mormal distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.log_cdf(x_)

def test_log_cdf():
    """
    Test log cdf.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_logcdf = gamma_benchmark.logcdf([2.0]).astype(np.float32)
    logcdf = LogCDF()
    output = logcdf(Tensor([2.0], dtype=dtype.float32))
    tol = 5e-5
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Gamma distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.survival_function(x_)

def test_survival():
    """
    Test log_survival.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_survival = gamma_benchmark.sf([2.0]).astype(np.float32)
    survival_function = SF()
    output = survival_function(Tensor([2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Gamma distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.g.log_survival(x_)

def test_log_survival():
    """
    Test log_survival.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_log_survival = gamma_benchmark.logsf([2.0]).astype(np.float32)
    log_survival = LogSF()
    output = log_survival(Tensor([2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_log_survival) < tol).all()

class EntropyH(nn.Cell):
    """
    Test class: entropy of Gamma distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self):
        return self.g.entropy()

def test_entropy():
    """
    Test entropy.
    """
    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_entropy = gamma_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Gamma distributions.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.g = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        entropy = self.g.entropy()
        kl_loss = self.g.kl_loss('Gamma', x_, y_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.g.cross_entropy('Gamma', x_, y_)
        return h_sum_kl - cross_entropy

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    concentration = Tensor([3.0], dtype=dtype.float32)
    rate = Tensor([2.0], dtype=dtype.float32)
    diff = cross_entropy(concentration, rate)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()

class Net(nn.Cell):
    """
    Test class: expand single distribution instance to multiple graphs
    by specifying the attributes.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.get_flags = msd.Gamma(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        kl = self.g.kl_loss('Gamma', x_, y_)
        prob = self.g.prob(kl)
        return prob

def test_multiple_graphs():
    """
    Test multiple graphs case.
    """
    prob = Net()
    concentration_a = np.array([3.0]).astype(np.float32)
    rate_a = np.array([1.0]).astype(np.float32)
    concentration_b = np.array([2.0]).astype(np.float32)
    rate_b = np.array([1.0]).astype(np.float32)
    ans = prob(Tensor(concentration_b), Tensor(rate_b))

    expect_kl_loss = (concentration_a - concentration_b) * special.digamma(concentration_a) \
                     + special.gammaln(concentration_b) - special.gammaln(concentration_a) \
                     + concentration_b * np.log(rate_a) - concentration_b * np.log(rate_b) \
                     + concentration_a * (rate_b / rate_a - 1.)

    gamma_benchmark = stats.gamma(np.array([3.0]))
    expect_prob = gamma_benchmark.pdf(expect_kl_loss).astype(np.float32)

    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expect_prob) < tol).all()
