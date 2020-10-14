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
"""test cases for Gumbel distribution"""
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
    Test class: probability of Gumbel distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    value = np.array([1.0, 2.0]).astype(np.float32)
    expect_pdf = gumbel_benchmark.pdf(value).astype(np.float32)
    pdf = Prob()
    output = pdf(Tensor(value, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Gumbel distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_logpdf = gumbel_benchmark.logpdf([1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    output = logprob(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class KL(nn.Cell):
    """
    Test class: kl_loss of Gumbel distribution.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([1.0, 2.0]), dtype=dtype.float32)

    def construct(self, loc_b, scale_b):
        return self.gum.kl_loss('Gumbel', loc_b, scale_b)

def test_kl_loss():
    """
    Test kl_loss.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([1.0, 2.0]).astype(np.float32)

    loc_b = np.array([1.0]).astype(np.float32)
    scale_b = np.array([1.0, 2.0]).astype(np.float32)

    expect_kl_loss = np.log(scale_b) - np.log(scale) +\
               np.euler_gamma * (scale / scale_b - 1.) +\
               np.expm1((loc_b - loc) / scale_b + special.loggamma(scale / scale_b + 1.))

    kl_loss = KL()
    loc_b = Tensor(loc_b, dtype=dtype.float32)
    scale_b = Tensor(scale_b, dtype=dtype.float32)
    output = kl_loss(loc_b, scale_b)
    tol = 1e-5
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Gumbel distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self):
        return self.gum.mean(), self.gum.sd(), self.gum.mode()

def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()

    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_mean = gumbel_benchmark.mean().astype(np.float32)
    expect_sd = gumbel_benchmark.std().astype(np.float32)
    expect_mode = np.array([[0.0], [0.0]]).astype(np.float32)
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Gumbel distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([1.0, 2.0, 3.0]), dtype=dtype.float32, seed=seed)
        self.shape = shape

    def construct(self):
        return self.gum.sample(self.shape)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    sample = Sampling(shape, seed=seed)
    output = sample()
    assert output.shape == (2, 3, 3)

class CDF(nn.Cell):
    """
    Test class: cdf of Gumbel distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.cdf(x_)

def test_cdf():
    """
    Test cdf.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_cdf = gumbel_benchmark.cdf([1.0, 2.0]).astype(np.float32)
    cdf = CDF()
    output = cdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Gumbel distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.log_cdf(x_)

def test_log_cdf():
    """
    Test log cdf.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_logcdf = gumbel_benchmark.logcdf([1.0, 2.0]).astype(np.float32)
    logcdf = LogCDF()
    output = logcdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-4
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Gumbel distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.survival_function(x_)

def test_survival():
    """
    Test log_survival.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_survival = gumbel_benchmark.sf([1.0, 2.0]).astype(np.float32)
    survival_function = SF()
    output = survival_function(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Gumbel distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.gum.log_survival(x_)

def test_log_survival():
    """
    Test log_survival.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_log_survival = gumbel_benchmark.logsf([1.0, 2.0]).astype(np.float32)
    log_survival = LogSF()
    output = log_survival(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 5e-4
    assert (np.abs(output.asnumpy() - expect_log_survival) < tol).all()

class EntropyH(nn.Cell):
    """
    Test class: entropy of Gumbel distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self):
        return self.gum.entropy()

def test_entropy():
    """
    Test entropy.
    """
    loc = np.array([0.0]).astype(np.float32)
    scale = np.array([[1.0], [2.0]]).astype(np.float32)
    gumbel_benchmark = stats.gumbel_r(loc, scale)
    expect_entropy = gumbel_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Gumbel distributions.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=dtype.float32)

    def construct(self, x_, y_):
        entropy = self.gum.entropy()
        kl_loss = self.gum.kl_loss('Gumbel', x_, y_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.gum.cross_entropy('Gumbel', x_, y_)
        return h_sum_kl - cross_entropy

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    loc = Tensor([1.0], dtype=dtype.float32)
    scale = Tensor([1.0], dtype=dtype.float32)
    diff = cross_entropy(loc, scale)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
