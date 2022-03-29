# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator, compile_net

SEED_ = 1
SEED2_ = 1
alpha_ = Tensor(np.array([1.0]), mstype.float32)
beta_ = Tensor(np.array([1.0]), mstype.float32)


class Net(Cell):
    def __init__(self, seed, seed2, strategy=None):
        super(Net, self).__init__()
        self.gamma = P.Gamma(seed, seed2).shard(strategy)

    def construct(self, shape, alpha, beta):
        out = self.gamma(shape, alpha, beta)
        return out


def test_gamma_auto_parallel():
    """
    Features: test Gamma auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0, full_batch=True)
    net = Net(SEED_, SEED2_)
    shape = (4, 4, 4)
    compile_net(net, shape, alpha_, beta_)


def test_gamma_data_parallel():
    """
    Features: test Gamma data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    net = Net(SEED_, SEED2_)
    shape = (8, 8)
    phase = compile_net(net, shape, alpha_, beta_)

    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs("Gamma-0", {"seed": 2, "seed2": 2})


def test_gamma_model_parallel():
    """
    Features: test Gamma model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=5)
    shape = (8, 8)
    strategy = ((2, 2), (1,), (1,))
    net = Net(SEED_, SEED2_, strategy)
    phase = compile_net(net, shape, alpha_, beta_)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs("Gamma-0", {"seed": 3, "seed2": 3})


def test_gamma_strategy_error():
    """
    Features:test Gamma strategy error
    Description: invalid strategy
    Expectation: Raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    shape = (8, 8)
    strategy = ((2, 2), (2,), (1,))
    net = Net(SEED_, SEED2_, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, shape, alpha_, beta_)
