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
"""
Test nn.probability.distribution.gumbel.
"""
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype
from mindspore import Tensor

def test_gumbel_shape_errpr():
    """
    Invalid shapes.
    """
    with pytest.raises(ValueError):
        msd.Gumbel([[2.], [1.]], [[2.], [3.], [4.]], dtype=dtype.float32)

def test_type():
    with pytest.raises(TypeError):
        msd.Gumbel(0., 1., dtype=dtype.int32)

def test_name():
    with pytest.raises(TypeError):
        msd.Gumbel(0., 1., name=1.0)

def test_seed():
    with pytest.raises(TypeError):
        msd.Gumbel(0., 1., seed='seed')

def test_scale():
    with pytest.raises(ValueError):
        msd.Gumbel(0., 0.)
    with pytest.raises(ValueError):
        msd.Gumbel(0., -1.)

def test_arguments():
    """
    args passing during initialization.
    """
    l = msd.Gumbel([3.0], [4.0], dtype=dtype.float32)
    assert isinstance(l, msd.Distribution)


class GumbelProb(nn.Cell):
    """
    Gumbel distribution: initialize with loc/scale.
    """
    def __init__(self):
        super(GumbelProb, self).__init__()
        self.gumbel = msd.Gumbel(3.0, 4.0, dtype=dtype.float32)

    def construct(self, value):
        prob = self.gumbel.prob(value)
        log_prob = self.gumbel.log_prob(value)
        cdf = self.gumbel.cdf(value)
        log_cdf = self.gumbel.log_cdf(value)
        sf = self.gumbel.survival_function(value)
        log_sf = self.gumbel.log_survival(value)
        return prob + log_prob + cdf + log_cdf + sf + log_sf

def test_gumbel_prob():
    """
    Test probability functions: passing value through construct.
    """
    net = GumbelProb()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)

class KL(nn.Cell):
    """
    Test kl_loss.
    """
    def __init__(self):
        super(KL, self).__init__()
        self.gumbel = msd.Gumbel(3.0, 4.0)

    def construct(self, mu, s):
        kl = self.gumbel.kl_loss('Gumbel', mu, s)
        cross_entropy = self.gumbel.cross_entropy('Gumbel', mu, s)
        return kl + cross_entropy

def test_kl_cross_entropy():
    """
    Test kl_loss and cross_entropy.
    """
    from mindspore import context
    context.set_context(device_target="Ascend")
    net = KL()
    loc_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    scale_b = Tensor(np.array([1.0]).astype(np.float32), dtype=dtype.float32)
    ans = net(loc_b, scale_b)
    assert isinstance(ans, Tensor)


class GumbelBasics(nn.Cell):
    """
    Test class: basic loc/scale function.
    """
    def __init__(self):
        super(GumbelBasics, self).__init__()
        self.gumbel = msd.Gumbel(3.0, 4.0, dtype=dtype.float32)

    def construct(self):
        mean = self.gumbel.mean()
        sd = self.gumbel.sd()
        mode = self.gumbel.mode()
        entropy = self.gumbel.entropy()
        return mean + sd + mode + entropy

def test_bascis():
    """
    Test mean/sd/mode/entropy functionality of Gumbel.
    """
    net = GumbelBasics()
    ans = net()
    assert isinstance(ans, Tensor)


class GumbelConstruct(nn.Cell):
    """
    Gumbel distribution: going through construct.
    """
    def __init__(self):
        super(GumbelConstruct, self).__init__()
        self.gumbel = msd.Gumbel(3.0, 4.0)


    def construct(self, value):
        prob = self.gumbel('prob', value)
        prob1 = self.gumbel.prob(value)
        return prob + prob1

def test_gumbel_construct():
    """
    Test probability function going through construct.
    """
    net = GumbelConstruct()
    value = Tensor([0.5, 1.0], dtype=dtype.float32)
    ans = net(value)
    assert isinstance(ans, Tensor)
