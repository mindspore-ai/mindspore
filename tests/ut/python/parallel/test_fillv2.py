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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fillv2 = P.FillV2()
        self.mul = P.Mul()

    def construct(self, y):
        x = self.fillv2(Tensor([64, 64], ms.int32), Tensor(1, ms.float32))
        out = self.mul(x, y)
        return out


def compile_graph(net, device_num, parallel_mode, y, search_mode="recursive_programming"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode,
                                      search_mode=search_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, y)
    return phase


def test_fillv2_semi_auto0():
    """
    Feature: distribute operator fillv2 in semi auto parallel.
    Description: fill net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    net = Net()
    y = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", y)
