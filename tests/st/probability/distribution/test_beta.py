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
"""test cases for Beta distribution"""
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
    Test class: probability of Beta distribution.
    """
    def __init__(self):
        super(Prob, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.b.prob(x_)

def test_pdf():
    """
    Test pdf.
    """
    beta_benchmark = stats.beta(np.array([3.0]), np.array([1.0]))
    expect_pdf = beta_benchmark.pdf([0.25, 0.75]).astype(np.float32)
    pdf = Prob()
    output = pdf(Tensor([0.25, 0.75], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_pdf) < tol).all()

class LogProb(nn.Cell):
    """
    Test class: log probability of Beta distribution.
    """
    def __init__(self):
        super(LogProb, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_):
        return self.b.log_prob(x_)

def test_log_likelihood():
    """
    Test log_pdf.
    """
    beta_benchmark = stats.beta(np.array([3.0]), np.array([1.0]))
    expect_logpdf = beta_benchmark.logpdf([0.25, 0.75]).astype(np.float32)
    logprob = LogProb()
    output = logprob(Tensor([0.25, 0.75], dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_logpdf) < tol).all()

class KL(nn.Cell):
    """
    Test class: kl_loss of Beta distribution.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        return self.b.kl_loss('Beta', x_, y_)

def test_kl_loss():
    """
    Test kl_loss.
    """
    concentration1_a = np.array([3.0]).astype(np.float32)
    concentration0_a = np.array([4.0]).astype(np.float32)

    concentration1_b = np.array([1.0]).astype(np.float32)
    concentration0_b = np.array([1.0]).astype(np.float32)

    total_concentration_a = concentration1_a + concentration0_a
    total_concentration_b = concentration1_b + concentration0_b
    log_normalization_a = np.log(special.beta(concentration1_a, concentration0_a))
    log_normalization_b = np.log(special.beta(concentration1_b, concentration0_b))
    expect_kl_loss = (log_normalization_b - log_normalization_a) \
                     - (special.digamma(concentration1_a) * (concentration1_b - concentration1_a)) \
                     - (special.digamma(concentration0_a) * (concentration0_b - concentration0_a)) \
                     + (special.digamma(total_concentration_a) * (total_concentration_b - total_concentration_a))

    kl_loss = KL()
    concentration1 = Tensor(concentration1_b, dtype=dtype.float32)
    concentration0 = Tensor(concentration0_b, dtype=dtype.float32)
    output = kl_loss(concentration1, concentration0)
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_kl_loss) < tol).all()

class Basics(nn.Cell):
    """
    Test class: mean/sd/mode of Beta distribution.
    """
    def __init__(self):
        super(Basics, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([3.0]), dtype=dtype.float32)

    def construct(self):
        return self.b.mean(), self.b.sd(), self.b.mode()

def test_basics():
    """
    Test mean/standard deviation/mode.
    """
    basics = Basics()
    mean, sd, mode = basics()
    beta_benchmark = stats.beta(np.array([3.0]), np.array([3.0]))
    expect_mean = beta_benchmark.mean().astype(np.float32)
    expect_sd = beta_benchmark.std().astype(np.float32)
    expect_mode = [0.5]
    tol = 1e-6
    assert (np.abs(mean.asnumpy() - expect_mean) < tol).all()
    assert (np.abs(mode.asnumpy() - expect_mode) < tol).all()
    assert (np.abs(sd.asnumpy() - expect_sd) < tol).all()

class Sampling(nn.Cell):
    """
    Test class: sample of Beta distribution.
    """
    def __init__(self, shape, seed=0):
        super(Sampling, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([1.0]), seed=seed, dtype=dtype.float32)
        self.shape = shape

    def construct(self, concentration1=None, concentration0=None):
        return self.b.sample(self.shape, concentration1, concentration0)

def test_sample():
    """
    Test sample.
    """
    shape = (2, 3)
    seed = 10
    concentration1 = Tensor([2.0], dtype=dtype.float32)
    concentration0 = Tensor([2.0, 2.0, 2.0], dtype=dtype.float32)
    sample = Sampling(shape, seed=seed)
    output = sample(concentration1, concentration0)
    assert output.shape == (2, 3, 3)

class EntropyH(nn.Cell):
    """
    Test class: entropy of Beta distribution.
    """
    def __init__(self):
        super(EntropyH, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self):
        return self.b.entropy()

def test_entropy():
    """
    Test entropy.
    """
    beta_benchmark = stats.beta(np.array([3.0]), np.array([1.0]))
    expect_entropy = beta_benchmark.entropy().astype(np.float32)
    entropy = EntropyH()
    output = entropy()
    tol = 1e-6
    assert (np.abs(output.asnumpy() - expect_entropy) < tol).all()

class CrossEntropy(nn.Cell):
    """
    Test class: cross entropy between Beta distributions.
    """
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.b = msd.Beta(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        entropy = self.b.entropy()
        kl_loss = self.b.kl_loss('Beta', x_, y_)
        h_sum_kl = entropy + kl_loss
        cross_entropy = self.b.cross_entropy('Beta', x_, y_)
        return h_sum_kl - cross_entropy

def test_cross_entropy():
    """
    Test cross_entropy.
    """
    cross_entropy = CrossEntropy()
    concentration1 = Tensor([3.0], dtype=dtype.float32)
    concentration0 = Tensor([2.0], dtype=dtype.float32)
    diff = cross_entropy(concentration1, concentration0)
    tol = 1e-6
    assert (np.abs(diff.asnumpy() - np.zeros(diff.shape)) < tol).all()

class Net(nn.Cell):
    """
    Test class: expand single distribution instance to multiple graphs
    by specifying the attributes.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.beta = msd.Beta(np.array([3.0]), np.array([1.0]), dtype=dtype.float32)

    def construct(self, x_, y_):
        kl = self.beta.kl_loss('Beta', x_, y_)
        prob = self.beta.prob(kl)
        return prob

def test_multiple_graphs():
    """
    Test multiple graphs case.
    """
    prob = Net()
    concentration1_a = np.array([3.0]).astype(np.float32)
    concentration0_a = np.array([1.0]).astype(np.float32)
    concentration1_b = np.array([2.0]).astype(np.float32)
    concentration0_b = np.array([1.0]).astype(np.float32)
    ans = prob(Tensor(concentration1_b), Tensor(concentration0_b))

    total_concentration_a = concentration1_a + concentration0_a
    total_concentration_b = concentration1_b + concentration0_b
    log_normalization_a = np.log(special.beta(concentration1_a, concentration0_a))
    log_normalization_b = np.log(special.beta(concentration1_b, concentration0_b))
    expect_kl_loss = (log_normalization_b - log_normalization_a) \
                     - (special.digamma(concentration1_a) * (concentration1_b - concentration1_a)) \
                     - (special.digamma(concentration0_a) * (concentration0_b - concentration0_a)) \
                     + (special.digamma(total_concentration_a) * (total_concentration_b - total_concentration_a))

    beta_benchmark = stats.beta(np.array([3.0]), np.array([1.0]))
    expect_prob = beta_benchmark.pdf(expect_kl_loss).astype(np.float32)

    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expect_prob) < tol).all()
