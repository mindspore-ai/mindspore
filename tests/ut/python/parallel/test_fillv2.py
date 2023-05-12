# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest

import mindspore as ms
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from .utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


shape_tuple = (512, 1024, 1024)
shape_tensor = Tensor(shape_tuple, ms.int32)


class Net(Cell):
    def __init__(self, shape, strategy=None):
        super(Net, self).__init__()
        self.fillv2 = P.FillV2().shard(strategy)
        self.value = Tensor(1, ms.float16)
        self.shape = shape

    def construct(self):
        output = self.fillv2(self.shape, self.value)
        return output


@pytest.mark.parametrize("shape", [shape_tuple, shape_tensor])
def test_fillv2_auto_parallel(shape):
    """
    Feature: test FillV2Info auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net(shape)
    compile_net(net)


@pytest.mark.parametrize("shape", [shape_tuple, shape_tensor])
def test_fillv2_data_parallel(shape):
    """
    Feature: test FillV2Info data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net(shape)
    compile_net(net)


@pytest.mark.parametrize("shape", [shape_tuple, shape_tensor])
def test_fillv2_model_parallel(shape):
    """
    Feature: test FillV2Info model parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 1), ())
    net = Net(shape, strategy)
    compile_net(net)


@pytest.mark.parametrize("shape", [shape_tuple, shape_tensor])
def test_fillv2_strategy_error(shape):
    """
    Feature: test invalid strategy for FillV2Info
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 2, 2), ())
    net = Net(shape, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net)
    context.reset_auto_parallel_context()
