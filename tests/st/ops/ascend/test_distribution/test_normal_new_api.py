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
"""test cases for new api of normal distribution"""
import numpy as np
from scipy import stats
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """
    Test class: new api of normal distribution.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.normal = msd.Normal(0., 1., dtype=dtype.float32)

    def construct(self, x_, y_):
        kl = self.normal.kl_loss('kl_loss', 'Normal', x_, y_)
        prob = self.normal.prob('prob', kl)
        return prob


def test_new_api():
    """
    Test new api of normal distribution.
    """
    prob = Net()
    mean_a = np.array([0.0]).astype(np.float32)
    sd_a = np.array([1.0]).astype(np.float32)
    mean_b = np.array([1.0]).astype(np.float32)
    sd_b = np.array([1.0]).astype(np.float32)
    ans = prob(Tensor(mean_b), Tensor(sd_b))

    diff_log_scale = np.log(sd_a) - np.log(sd_b)
    squared_diff = np.square(mean_a / sd_b - mean_b / sd_b)
    expect_kl_loss = 0.5 * squared_diff + 0.5 * \
        np.expm1(2 * diff_log_scale) - diff_log_scale

    norm_benchmark = stats.norm(np.array([0.0]), np.array([1.0]))
    expect_prob = norm_benchmark.pdf(expect_kl_loss).astype(np.float32)

    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expect_prob) < tol).all()
