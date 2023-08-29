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
# ============================================================================
import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.train import Model


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.invert = P.Invert().shard(strategy)
        self.pop = P.PopulationCount().shard(strategy)
        self.cast = P.Cast().shard(strategy)
        self.relu = P.ReLU().shard(strategy)

    def construct(self, x, b):
        out = self.invert(x)
        out = self.pop(out)
        out = self.cast(out, ms.float32)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([16, 16]), dtype=ms.int16)
_w = Tensor(np.ones([16, 16]), dtype=ms.int16)
_b = Tensor(np.ones([16, 16]), dtype=ms.int16)


def compile_net(net):
    model = Model(net)
    model.predict(_x, _b)
    context.reset_auto_parallel_context()


def test_invert_population_count_semi():
    """
    Feature: semi auto parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((2, 4),)
    net = Net(strategy)
    compile_net(net)


def test_invert_population_count_auto():
    """
    Feature: auto parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      full_batch=True)
    strategy = None
    net = Net(strategy)
    compile_net(net)
