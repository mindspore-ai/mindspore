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
"""test cases for Exponential distribution"""
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
    Test class: probability of Exponential distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_pdf = expon_benchmark.pdf([-1.0, 0.0, 1.0]).astype(np.float32)
    pdf = Prob()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = pdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Exponential distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_logpdf = expon_benchmark.logpdf([0.5, 1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    x_ = Tensor(np.array([0.5, 1.0, 2.0]).astype(np.float32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class KL(nn.Cell):
    """
    Test class: kl_loss between Exponential distributions.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.e = msd.Exponential([1.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.kl_loss('Exponential', x_)

def test_kl_loss():
    """
    Test kl_loss.
    """
    rate_a = 1.5
    rate_b = np.array([0.5, 2.0]).astype(np.float32)
    expect_kl_loss = np.log(rate_a) - np.log(rate_b) + rate_b / rate_a  - 1.0
    kl = KL()
    output = kl(Tensor(rate_b, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Exponential distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.e = msd.Exponential([0.5], dtype=dtype.float32)

    def construct(self):
        return self.e.mean(), self.e.sd(), self.e.mode()

def test_basics():
    """
    Test mean/standard/mode deviation.
    """
    basics = Basics()
    mean, sd, mode = basics()
    expect_mean = 2.
    expect_sd = 2.
    expect_mode = 0.
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Exponential distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, rate=None):
        return self.e.sample(self.shape, rate)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    rate = Tensor([1.0, 2.0, 3.0], dtype=dtype.float32)
    sample = Sampling(shape, seed=seed)
    output = sample(rate)
    assert output.shape == (2, 3, 3)

class CDF(nn.Cell):
    """
    Test class: cdf of Exponential distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.cdf(x_)

def test_cdf():
    """
    Test cdf.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_cdf = expon_benchmark.cdf([-1.0, 0.0, 1.0]).astype(np.float32)
    cdf = CDF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Exponential distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.log_cdf(x_)

def test_log_cdf():
    """
    Test log_cdf.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_logcdf = expon_benchmark.logcdf([0.5, 1.0, 2.5]).astype(np.float32)
    logcdf = LogCDF()
    x_ = Tensor(np.array([0.5, 1.0, 2.5]).astype(np.float32), dtype=dtype.float32)
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Exponential distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.survival_function(x_)

def test_survival():
    """
    Test survival function.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_survival = expon_benchmark.sf([-1.0, 0.0, 1.0]).astype(np.float32)
    survival = SF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = survival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Exponential distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self, x_):
        return self.e.log_survival(x_)

def test_log_survival():
    """
    Test log survival function.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_logsurvival = expon_benchmark.logsf([-1.0, 0.0, 1.0]).astype(np.float32)
    logsurvival = LogSF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = logsurvival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()

class EntropyH(nn.Cell):
    """
    Test class: entropy of Exponential distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.e = msd.Exponential([[1.0], [0.5]], dtype=dtype.float32)

    def construct(self):
        return self.e.entropy()

def test_entropy():
    """
    Test entropy.
    """
    expon_benchmark = stats.expon(scale=[[1.0], [2.0]])
    expect_entropy = expon_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Exponential distribution.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.e = msd.Exponential([1.0], dtype=dtype.float32)

    def construct(self, x_):
        entropy = self.e.entropy()
        kl_loss = self.e.kl_loss('Exponential', x_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.e.cross_entropy('Exponential', x_)
        return h_sum_kl - cross_entropy

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    rate = Tensor([0.5], dtype=dtype.float32)
    diff = cross_entropy(rate)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
