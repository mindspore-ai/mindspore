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
""" test lazy adam """
import pytest
import numpy as np
from mindspore.nn.optim import LazyAdam, FTRL, Adam, ProximalAdagrad
import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.ops import operations as P

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(enable_sparse=True)
    yield
    context.set_context(enable_sparse=False)


class NetWithSparseGatherV2(nn.Cell):
    """ NetWithSparseGatherV2 definition """
    def __init__(self):
        super(NetWithSparseGatherV2, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight1")
        self.weight2 = Parameter(Tensor(np.ones([2, 1, 2]).astype((np.float32))), name="weight2")
        self.axis = 0
        self.gather = P.SparseGatherV2()

    def construct(self, indices, label):
        return self.gather(self.weight1, indices, self.axis) + self.weight2


def test_ftrl_target():
    """ test_ftrl_target """
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = FTRL(net.trainable_params(), weight_decay=0.9, loss_scale=2.0)
    if optimizer.target not in ('CPU', 'Ascend'):
        raise ValueError("The value must be 'CPU' or 'Ascend', but got value {}".format(optimizer.target))


def test_lazyadam_target():
    """ test_lazyadam_target """
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = LazyAdam(net.trainable_params(), learning_rate=0.1, weight_decay=0.9, loss_scale=2.0)
    if optimizer.target not in ('CPU', 'Ascend'):
        raise ValueError("The value must be 'CPU' or 'Ascend', but got value {}".format(optimizer.target))


def test_adam_target():
    """ test_adam_target """
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = Adam(net.trainable_params(), learning_rate=0.1, loss_scale=1024.0, weight_decay=0.9)
    if optimizer.target not in ('CPU', 'Ascend'):
        raise ValueError("The value must be 'CPU' or 'Ascend', but got value {}".format(optimizer.target))


def test_proximal_target():
    """ test_proximal_target """
    net = NetWithSparseGatherV2()
    net.set_train()

    optimizer = ProximalAdagrad(net.trainable_params(), weight_decay=0.9, loss_scale=1024.0)
    if optimizer.target not in ('CPU', 'Ascend'):
        raise ValueError("The value must be 'CPU' or 'Ascend', but got value {}".format(optimizer.target))
