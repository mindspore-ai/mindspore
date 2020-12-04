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
"""test cases for Poisson distribution"""
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
    Test class: probability of Poisson distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_pdf = poisson_benchmark.pmf([-1.0, 0.0, 1.0]).astype(np.float32)
    pdf = Prob()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = pdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Poisson distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_logpdf = poisson_benchmark.logpmf([1.0, 2.0]).astype(np.float32)
    logprob = LogProb()
    x_ = Tensor(np.array([1.0, 2.0]).astype(np.float32), dtype=dtype.float32)
    output = logprob(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Poisson distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.p = msd.Poisson([1.44], dtype=dtype.float32)

    def construct(self):
        return self.p.mean(), self.p.sd(), self.p.mode()

def test_basics():
    """
    Test mean/standard/mode deviation.
    """
    basics = Basics()
    mean, sd, mode = basics()
    expect_mean = 1.44
    expect_sd = 1.2
    expect_mode = 1
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Poisson distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.p = msd.Poisson([[1.0], [0.5]], seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, rate=None):
        return self.p.sample(self.shape, rate)

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
    Test class: cdf of Poisson distribution.
    """
    def __init__(self):
        super(CDF, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.cdf(x_)

def test_cdf():
    """
    Test cdf.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_cdf = poisson_benchmark.cdf([-1.0, 0.0, 1.0]).astype(np.float32)
    cdf = CDF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = cdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_cdf) < tol).all()

class LogCDF(nn.Cell):
    """
    Test class: log_cdf of Poisson distribution.
    """
    def __init__(self):
        super(LogCDF, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.log_cdf(x_)

def test_log_cdf():
    """
    Test log_cdf.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_logcdf = poisson_benchmark.logcdf([0.5, 1.0, 2.5]).astype(np.float32)
    logcdf = LogCDF()
    x_ = Tensor(np.array([0.5, 1.0, 2.5]).astype(np.float32), dtype=dtype.float32)
    output = logcdf(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logcdf) < tol).all()

class SF(nn.Cell):
    """
    Test class: survival function of Poisson distribution.
    """
    def __init__(self):
        super(SF, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.survival_function(x_)

def test_survival():
    """
    Test survival function.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_survival = poisson_benchmark.sf([-1.0, 0.0, 1.0]).astype(np.float32)
    survival = SF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = survival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_survival) < tol).all()

class LogSF(nn.Cell):
    """
    Test class: log survival function of Poisson distribution.
    """
    def __init__(self):
        super(LogSF, self).__init__()
        self.p = msd.Poisson([0.5], dtype=dtype.float32)

    def construct(self, x_):
        return self.p.log_survival(x_)

def test_log_survival():
    """
    Test log survival function.
    """
    poisson_benchmark = stats.poisson(mu=0.5)
    expect_logsurvival = poisson_benchmark.logsf([-1.0, 0.0, 1.0]).astype(np.float32)
    logsurvival = LogSF()
    x_ = Tensor(np.array([-1.0, 0.0, 1.0]).astype(np.float32), dtype=dtype.float32)
    output = logsurvival(x_)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logsurvival) < tol).all()
