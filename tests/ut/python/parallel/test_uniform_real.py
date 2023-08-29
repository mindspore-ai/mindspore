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

from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


SEED_ = 1
SEED2_ = 1


class Net(Cell):
    def __init__(self, seed, seed2, strategy=None):
        super(Net, self).__init__()
        self.uniform_real = P.UniformReal(seed, seed2).shard(strategy)

    def construct(self, shape):
        out = self.uniform_real(shape)
        return out


def test_uniform_real_auto_parallel():
    """
    Features: test UniformReal auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(SEED_, SEED2_)
    shape = (4, 4, 4)
    compile_net(net, shape)


def test_uniform_real_data_parallel():
    """
    Features: test UniformReal data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    net = Net(SEED_, SEED2_)
    shape = (8, 8)
    phase = compile_net(net, shape)

    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs("UniformReal-0", {"seed": 2, "seed2": 2})


def test_uniform_real_model_parallel():
    """
    Features: test UniformReal model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=5)
    shape = (8, 8)
    strategy = ((2, 2),)
    net = Net(SEED_, SEED2_, strategy)
    phase = compile_net(net, shape)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_attrs("UniformReal-0", {"seed": 3, "seed2": 3})
