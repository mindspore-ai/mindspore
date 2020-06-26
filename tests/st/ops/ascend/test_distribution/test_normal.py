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
"""test cases for normal distribution"""
import numpy as np
from scipy import stats
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    """
    Test class: probability of normal distribution.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.n = nn.Normal(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    @ms_function
    def construct(self, x_):
        return self.n('prob', x_)

class Net1(nn.Cell):
    """
    Test class: log probability of normal distribution.
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.n = nn.Normal(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    @ms_function
    def construct(self, x_):
        return self.n('log_prob', x_)

class Net2(nn.Cell):
    """
    Test class: kl_loss of normal distribution.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.n = nn.Normal(np.array([3.0]), np.array([4.0]), dtype=dtype.float32)

    @ms_function
    def construct(self, x_, y_):
        return self.n('kl_loss', 'Normal', x_, y_)

class Net3(nn.Cell):
    """
    Test class: mean/sd of normal distribution.
    """
    def __init__(self):
        super(Net3, self).__init__()
        self.n = nn.Normal(np.array([3.0]), np.array([[2.0], [4.0]]), dtype=dtype.float32)

    @ms_function
    def construct(self):
        return self.n('mean'), self.n('sd')

def test_pdf():
    """
    Test pdf.
    """
    norm_benchmark = stats.norm(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_pdf = norm_benchmark.pdf([1.0, 2.0]).astype(np.float32)
    pdf = Net()
    output = pdf(Tensor([1.0, 2.0], dtype=dtype.float32))
    print("expected_pdf: ", expect_pdf)
    print("ans: ", output.asnumpy())
    tol = 1e-6
    assert (output.asnumpy() - expect_pdf < tol).all()

def test_log_likelihood():
    """
    Test log_pdf.
    """
    norm_benchmark = stats.norm(np.array([3.0]), np.array([[2.0], [4.0]]))
    expect_logpdf = norm_benchmark.logpdf([1.0, 2.0]).astype(np.float32)
    logprob = Net1()
    output = logprob(Tensor([1.0, 2.0], dtype=dtype.float32))
    print("expected_log_probability: ", expect_logpdf)
    print("ans: ", output.asnumpy())
    tol = 1e-6
    assert (output.asnumpy() - expect_logpdf < tol).all()

def test_kl_loss():
    """
    Test kl_loss.
    """
    mean_a = np.array([3.0]).astype(np.float32)
    sd_a = np.array([4.0]).astype(np.float32)

    mean_b = np.array([1.0]).astype(np.float32)
    sd_b = np.array([1.0]).astype(np.float32)

    diff_log_scale = np.log(sd_a) - np.log(sd_b)
    squared_diff = np.square(mean_a / sd_b - mean_b / sd_b)
    expect_kl_loss = 0.5 * squared_diff + 0.5 * np.expm1(2 * diff_log_scale) - diff_log_scale

    kl_loss = Net2()
    mean = Tensor(mean_b, dtype=dtype.float32)
    sd = Tensor(sd_b, dtype=dtype.float32)
    output = kl_loss(mean, sd)
    print("expected_kl_loss: ", expect_kl_loss)
    print("ans: ", output.asnumpy())
    tol = 1e-6
    assert (output.asnumpy() - expect_kl_loss < tol).all()

def test_basics():
    """
    Test mean/standard deviation.
    """
    basics = Net3()
    mean, sd = basics()
    print("mean is ", mean)
    print("sd is ", sd)
