# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.operations._inner_ops import Dequant
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class DequantNet(Cell):
    def __init__(self, sqrt_mode, relu_mode, strategy):
        super().__init__()
        self.dequant = Dequant(sqrt_mode, relu_mode).shard(strategy)

    def construct(self, data, dep_scale):
        out = self.dequant(data, dep_scale)
        return out


def test_dequant_1D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1), (1,))

    net = DequantNet(False, False, strategy)

    data = Parameter(Tensor(np.ones([4096, 512]), dtype=ms.int32), "data")
    dep_scale = Parameter(Tensor(np.ones([512]), dtype=ms.uint64, const_arg=True), "dep_scale")
    net.set_inputs(data, dep_scale)

    phase = compile_net(net, data, dep_scale)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1024, 512])
    assert validator.check_parameter_shape("dep_scale", [512])
