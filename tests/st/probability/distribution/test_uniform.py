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
"""test cases for Uniform distribution"""
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
    Test class: probability of Uniform distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.u = msd.Uniform([0.0], [[1.0], [2.0]], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    uniform_benchmark = stats.uniform([0.0], [[1.0], [2.0]])
    expect_pdf = uniform_benchmark.pdf([-1.0, 0.0, 0.5, 1.0, 1.5, 3.0]).astype(np.float32)
    pdf = Prob()
    x_ = Tensor(np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 3.0]).astype(np.float32), dtype=dtype.float32)
    output = pdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Uniform distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.u = msd.Uniform([0.0], [[1.0], [2.0]], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    uniform_benchmark = stats.uniform([0.0], [[1.0], [2.0]])
    expect_logpdf = uniform_benchmark.logpdf([0.5]).astype(np.float32)
    logprob = LogProb()
    x_ = Tensor(np.array([0.5]).astype(np.float32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class KL(nn.Cell):
    """
    Test class: kl_loss between Uniform distributions.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.u = msd.Uniform([0.0], [1.5], dtype=dtype.float32)

    def construct(self, x_, y_):
        return self.u.kl_loss('Uniform', x_, y_)

def test_kl_loss():
    """
    Test kl_loss.
    """
    low_a = 0.0
    high_a = 1.5
    low_b = -1.0
    high_b = 2.0
    expect_kl_loss = np.log(high_b - low_b) - np.log(high_a - low_a)
    kl = KL()
    output = kl(Tensor(low_b, dtype=dtype.float32), Tensor(high_b, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd of Uniform distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.u = msd.Uniform([0.0], [3.0], dtype=dtype.float32)

    def construct(self):
        return self.u.mean(), self.u.sd()

def test_basics():
    """
    Test mean/standard deviation.
    """
    basics = Basics()
    mean, sd = basics()
    expect_mean = [1.5]
    expect_sd = np.sqrt([0.75])
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Uniform distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.u = msd.Uniform([0.0], [[1.0], [2.0]], seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, low=None, high=None):
        return self.u.sample(self.shape, low, high)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    low = Tensor([1.0], dtype=dtype.float32)
    high = Tensor([2.0, 3.0, 4.0], dtype=dtype.float32)
    sample = Sampling(shape, seed=seed)
    output = sample(low, high)
    assert output.shape == (2, 3, 3)

class CDF(nn.Cell):
    """
    Test class: cdf of Uniform distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.u = msd.Uniform([0.0], [1.0], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.cdf(x_)

def test_cdf():
    """
    Test cdf.
    """
    uniform_benchmark = stats.uniform([0.0], [1.0])
    expect_cdf = uniform_benchmark.cdf([-1.0, 0.5, 1.0, 2.0]).astype(np.float32)
    cdf = CDF()
    x_ = Tensor(np.array([-1.0, 0.5, 1.0, 2.0]).astype(np.float32), dtype=dtype.float32)
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Uniform distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.u = msd.Uniform([0.0], [1.0], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.log_cdf(x_)

class SF(nn.Cell):
    """
    Test class: survival function of Uniform distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.u = msd.Uniform([0.0], [1.0], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.survival_function(x_)

class LogSF(nn.Cell):
    """
    Test class: log survival function of Uniform distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.u = msd.Uniform([0.0], [1.0], dtype=dtype.float32)

    def construct(self, x_):
        return self.u.log_survival(x_)

class EntropyH(nn.Cell):
    """
    Test class: entropy of Uniform distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.u = msd.Uniform([0.0], [1.0, 2.0], dtype=dtype.float32)

    def construct(self):
        return self.u.entropy()

def test_entropy():
    """
    Test entropy.
    """
    uniform_benchmark = stats.uniform([0.0], [1.0, 2.0])
    expect_entropy = uniform_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross_entropy between Uniform distributions.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.u = msd.Uniform([0.0], [1.5], dtype=dtype.float32)

    def construct(self, x_, y_):
        entropy = self.u.entropy()
        kl_loss = self.u.kl_loss('Uniform', x_, y_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.u.cross_entropy('Uniform', x_, y_)
        return h_sum_kl - cross_entropy

def test_log_cdf():
    """
    Test log_cdf.
    """
    uniform_benchmark = stats.uniform([0.0], [1.0])
    expect_logcdf = uniform_benchmark.logcdf([0.5, 0.8, 2.0]).astype(np.float32)
    logcdf = LogCDF()
    x_ = Tensor(np.array([0.5, 0.8, 2.0]).astype(np.float32), dtype=dtype.float32)
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

def test_survival():
    """
    Test survival function.
    """
    uniform_benchmark = stats.uniform([0.0], [1.0])
    expect_survival = uniform_benchmark.sf([-1.0, 0.5, 1.0, 2.0]).astype(np.float32)
    survival = SF()
    x_ = Tensor(np.array([-1.0, 0.5, 1.0, 2.0]).astype(np.float32), dtype=dtype.float32)
    output = survival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

def test_log_survival():
    """
    Test log survival function.
    """
    uniform_benchmark = stats.uniform([0.0], [1.0])
    expect_logsurvival = uniform_benchmark.logsf([0.5, 0.8, -2.0]).astype(np.float32)
    logsurvival = LogSF()
    x_ = Tensor(np.array([0.5, 0.8, -2.0]).astype(np.float32), dtype=dtype.float32)
    output = logsurvival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    low_b = -1.0
    high_b = 2.0
    diff = cross_entropy(Tensor(low_b, dtype=dtype.float32), Tensor(high_b, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()
