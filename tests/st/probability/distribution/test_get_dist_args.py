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
"""test cases for Normal distribution"""
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net1(nn.Cell):
    """
    Test class: Normal distribution.  `dist_spec_args` are `mean`, `sd`.
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.normal = msd.Normal(dtype=dtype.float32)
        self.normal1 = msd.Normal(0.0, 1.0, dtype=dtype.float32)
        self.normal2 = msd.Normal(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value, mean, sd, mean_a, sd_a):
        args_list = self.normal.get_dist_args(mean, sd)
        prob = self.normal1.prob(value, *args_list)
        args_list1 = self.normal.get_dist_args()
        prob1 = self.normal2.prob(value, *args_list1)

        args_list2 = self.normal1.get_dist_args()
        dist_type = self.normal1.get_dist_type()
        kl_loss = self.normal2.kl_loss(dist_type, *args_list2)

        args_list3 = self.normal.get_dist_args(mean_a, sd_a)
        dist_type = self.normal1.get_dist_type()
        kl_loss1 = self.normal2.kl_loss(dist_type, *args_list3)
        return prob, prob1, kl_loss, kl_loss1

def test1():
    """
    Test Normal with two `dist_spec_args`.
    """
    net = Net1()
    mean = Tensor(3.0, dtype=dtype.float32)
    sd = Tensor(4.0, dtype=dtype.float32)
    mean_a = Tensor(0.0, dtype=dtype.float32)
    sd_a = Tensor(1.0, dtype=dtype.float32)
    value = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    ans, expected, ans1, expected1 = net(value, mean, sd, mean_a, sd_a)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected.asnumpy()) < tol).all()
    assert (np.abs(ans1.asnumpy() - expected1.asnumpy()) < tol).all()

class Net2(nn.Cell):
    """
    Test class: Exponential distribution.  `dist_spec_args` is `rate`.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.expon = msd.Exponential(dtype=dtype.float32)
        self.expon1 = msd.Exponential(1.0, dtype=dtype.float32)
        self.expon2 = msd.Exponential(2.0, dtype=dtype.float32)

    def construct(self, value, rate, rate1):
        args_list = self.expon.get_dist_args(rate)
        prob = self.expon1.prob(value, *args_list)
        args_list1 = self.expon.get_dist_args()
        prob1 = self.expon2.prob(value, *args_list1)

        args_list2 = self.expon1.get_dist_args()
        dist_type = self.expon1.get_dist_type()
        kl_loss = self.expon2.kl_loss(dist_type, *args_list2)

        args_list3 = self.expon.get_dist_args(rate1)
        dist_type = self.expon.get_dist_type()
        kl_loss1 = self.expon2.kl_loss(dist_type, *args_list3)
        return prob, prob1, kl_loss, kl_loss1

def test2():
    """
    Test Expomential with single `dist_spec_args`.
    """
    net = Net2()
    rate = Tensor(2.0, dtype=dtype.float32)
    rate1 = Tensor(1.0, dtype=dtype.float32)
    value = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    ans, expected, ans1, expected1 = net(value, rate, rate1)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected.asnumpy()) < tol).all()
    assert (np.abs(ans1.asnumpy() - expected1.asnumpy()) < tol).all()
