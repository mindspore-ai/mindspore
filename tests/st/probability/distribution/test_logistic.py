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
"""test cases for Logistic distribution"""
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
    Test class: probability of Logistic distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_pdf = logistic_benchmark.pdf([1.0, 2.0]).astype(np.float32)
    pdf = Prob()
    output = pdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Logistic distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_logpdf = logistic_benchmark.logpdf([1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    output = logprob(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Logistic distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([2.0, 4.0]), dtype=dtype.float32)

    def construct(self):
        return self.l.mean(), self.l.sd(), self.l.mode()

def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()
    expect_mean = [3.0, 3.0]
    expect_sd = np.pi * np.array([2.0, 4.0]) / np.sqrt(np.array([3.0]))
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Logistic distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, mean=None, sd=None):
        return self.l.sample(self.shape, mean, sd)

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
    Test class: cdf of Logistic distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.cdf(x_)


def test_cdf():
    """
    Test cdf.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_cdf = logistic_benchmark.cdf([1.0, 2.0]).astype(np.float32)
    cdf = CDF()
    output = cdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Logistic distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.log_cdf(x_)

def test_log_cdf():
    """
    Test log cdf.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_logcdf = logistic_benchmark.logcdf([1.0, 2.0]).astype(np.float32)
    logcdf = LogCDF()
    output = logcdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 5e-5
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Logistic distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.survival_function(x_)

def test_survival():
    """
    Test log_survival.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_survival = logistic_benchmark.sf([1.0, 2.0]).astype(np.float32)
    survival_function = SF()
    output = survival_function(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Logistic distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self, x_):
        return self.l.log_survival(x_)

def test_log_survival():
    """
    Test log_survival.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_log_survival = logistic_benchmark.logsf([1.0, 2.0]).astype(np.float32)
    log_survival = LogSF()
    output = log_survival(Tensor([1.0, 2.0], dtype=dtype.float32))
    tol = 2e-5
    assert (np.abs(output.asnumpy() - expect_log_survival) < tol).all()

class EntropyH(nn.Cell):
    """
    Test class: entropy of Logistic distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.l = msd.Logistic(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    def construct(self):
        return self.l.entropy()

def test_entropy():
    """
    Test entropy.
    """
    logistic_benchmark = stats.logistic(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_entropy = logistic_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()
