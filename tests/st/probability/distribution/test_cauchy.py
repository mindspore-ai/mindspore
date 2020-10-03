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
"""test cases for Cauchy distribution"""
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
    Test class: probability of Cauchy distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_pdf = cauchy_benchmark.pdf([1.0, 2.0]).astype(np.float32)
    pdf = Prob()
    output = pdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Cauchy distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_logpdf = cauchy_benchmark.logpdf([1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    output = logprob(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class KL(nn.Cell):
    """
    Test class: kl_loss of Cauchy distribution.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.c = msd.Cauchy(np.array([3.]), np.array([4.]), dtype=dtype.float32)

    def construct(self, mu, s):
        return self.c.kl_loss('Cauchy', mu, s)

def test_kl_loss():
    """
    Test kl_loss.
    """
    loc_b = np.array([0.]).astype(np.float32)
    scale_b = np.array([1.]).astype(np.float32)

    loc_a = np.array([3.0]).astype(np.float32)
    scale_a = np.array([4.0]).astype(np.float32)

    sum_square = np.square(scale_a + scale_b)
    square_diff = np.square(loc_a - loc_b)
    expect_kl_loss = np.log(sum_square + square_diff) - \
                np.log(4.0 * scale_a  * scale_b)

    kl_loss = KL()
    loc = Tensor(loc_b, dtype=dtype.float32)
    scale = Tensor(scale_b, dtype=dtype.float32)
    output = kl_loss(loc, scale)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mode of Cauchy distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([2.0, 4.0]), dtype=dtype.float32)

    def construct(self):
        return self.c.mode()

def test_basics():
    """
    Test mode.
    """
    basics = Basics()
    mode = basics()
    expect_mode = np.array([3.0, 3.0])
    tol = 1e-6
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Cauchy distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, mean=None, sd=None):
        return self.c.sample(self.shape, mean, sd)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    mean = Tensor([2.0], dtype=dtype.float32)
    sd = Tensor([2.0, 2.0, 2.0], dtype=dtype.float32)
    sample = Sampling(shape, seed=seed)
    output = sample(mean, sd)
    assert output.shape == (2, 3, 3)

class CDF(nn.Cell):
    """
    Test class: cdf of Cauchy distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_cdf = cauchy_benchmark.cdf([1.0, 2.0]).astype(np.float32)
    cdf = CDF()
    output = cdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Cauchy distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.log_cdf(x_)

def test_log_cdf():
    """
    Test log cdf.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_logcdf = cauchy_benchmark.logcdf([1.0, 2.0]).astype(np.float32)
    logcdf = LogCDF()
    output = logcdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 5e-5
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Cauchy distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.survival_function(x_)

def test_survival():
    """
    Test log_survival.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_survival = cauchy_benchmark.sf([1.0, 2.0]).astype(np.float32)
    survival_function = SF()
    output = survival_function(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Cauchy distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.c.log_survival(x_)

def test_log_survival():
    """
    Test log_survival.
    """
    cauchy_benchmark = stats.cauchy(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_log_survival = cauchy_benchmark.logsf([1.0, 2.0]).astype(np.float32)
    log_survival = LogSF()
    output = log_survival(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_log_survival) < tol).all()

class EntropyH(nn.Cell):
    """
    Test class: entropy of Cauchy distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.c = msd.Cauchy(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self):
        return self.c.entropy()

def test_entropy():
    """
    Test entropy.
    """
    expect_entropy = np.log(4 * np.pi * np.array([[2.0], [4.0]]))
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Cauchy distributions.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.c = msd.Cauchy(np.array([3.]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, mu, s):
        entropy = self.c.entropy()
        kl_loss = self.c.kl_loss('Cauchy', mu, s)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.c.cross_entropy('Cauchy', mu, s)
        return h_sum_kl - cross_entropy

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    mean = Tensor([1.0], dtype=dtype.float32)
    sd = Tensor([1.0], dtype=dtype.float32)
    diff = cross_entropy(mean, sd)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
