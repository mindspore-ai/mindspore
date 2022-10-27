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

import numpy as np
import pytest
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter
import mindspore as ms
import mindspore.common.api as me
from mindspore.common.initializer import initializer
from mindspore.common import set_seed
from hccl_test.manage.api import Hccl


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, weight1, weight2):
        super().__init__()
        self.weight1 = Parameter(weight1, "w1")
        self.weight2 = Parameter(weight2, "w2")
        self.matmul1 = P.MatMul(transpose_a=False, transpose_b=True).shard(strategy1)
        self.matmul2 = P.MatMul(transpose_a=False, transpose_b=True).shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)

    def construct(self, x):
        out = self.matmul1(x, self.weight1)
        out = self.matmul2(out, self.weight2)
        out = self.relu(out)
        return out


def check_initializer_weight_slice(init_name="Uniform", using_seed=False):
    def get_slice(rank):
        if using_seed:
            set_seed(1)
        hccl = Hccl()
        rank_save = hccl.rank_id
        hccl.rank_id = rank
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(dataset_strategy="full_batch")
        context.set_auto_parallel_context(device_num=8, global_rank=rank)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        strategy1 = ((2, 1), (4, 1))
        strategy2 = ((2, 4),)
        context.set_context(mode=context.GRAPH_MODE)
        exe = me._cell_graph_executor

        x = Tensor(np.ones([32, 32]), dtype=ms.float32)
        weight1 = initializer(init_name, [32, 32], ms.float32)
        weight2 = initializer(init_name, [32, 32], ms.float32)
        net = Net(strategy1, strategy2, weight1, weight2)
        net.set_train()
        exe.compile(net, x, phase='train')
        hccl.rank_id = rank_save
        return net.parameters_dict()['w1'].data.asnumpy(), net.parameters_dict()['w2'].data.asnumpy()

    Tensor.delta_seed = 0
    w1_slice0, w2_slice0 = get_slice(0)
    Tensor.delta_seed = 0
    w1_slice1, _ = get_slice(1)
    Tensor.delta_seed = 0
    w1_slice4, _ = get_slice(4)
    slice_shape = w1_slice0.shape

    w1_slice0 = w1_slice0.flatten()
    w1_slice1 = w1_slice1.flatten()
    w1_slice4 = w1_slice4.flatten()
    w2_slice0 = w2_slice0.flatten()
    expect_slice_shape = (8, 32)

    assert expect_slice_shape == slice_shape
    assert all(w1_slice0 == w1_slice4)
    if init_name not in ["One", "Zero"]:
        assert any(w1_slice0 != w1_slice1)
        if using_seed:
            assert all(w1_slice0 == w2_slice0)
        else:
            assert any(w1_slice0 != w2_slice0)


initializers = ["Uniform", "Normal", "TruncatedNormal", "HeUniform", "HeNormal", "XavierUniform", "One", "Zero"]


def test_initializer_weight_slice():
    """
    Feature: test initializer in auto parallel with/without using set_seed.
    Description: test initializer in auto parallel with/without using set_seed.
    Expectation: without any assert error.
    """
    for init_name in initializers:
        check_initializer_weight_slice(init_name)
    for init_name in initializers:
        check_initializer_weight_slice(init_name, using_seed=True)


def test_wrong_order_set_parallel_mode_with_initializer():
    """
    Feature: test parameter initialize in auto parallel.
    Description: test parameter initialize in auto parallel applying initializer before setting auto parallel mode.
    Expectation: without any assert error.
    """
    weight1 = initializer("Normal", [32, 32], ms.float32)
    weight2 = initializer("Normal", [32, 32], ms.float32)
    strategy1 = ((2, 1), (4, 1))
    strategy2 = ((2, 4),)
    net = Net(strategy1, strategy2, weight1, weight2)
    exe = me._cell_graph_executor
    x = Tensor(np.ones([32, 32]), dtype=ms.float32)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    with pytest.raises(RuntimeError):
        exe.compile(net, x, phase='train')


def test_wrong_order_set_same_parallel_mode_with_initializer():
    """
    Feature: test parameter initialize in auto parallel.
    Description: test parameter initialize in auto parallel applying initializer after setting auto parallel mode.
    Expectation: without any assert error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    weight1 = initializer("Normal", [32, 32], ms.float32)
    weight2 = initializer("Normal", [32, 32], ms.float32)
    strategy1 = ((2, 1), (4, 1))
    strategy2 = ((2, 4),)
    net = Net(strategy1, strategy2, weight1, weight2)
    exe = me._cell_graph_executor
    x = Tensor(np.ones([32, 32]), dtype=ms.float32)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    exe.compile(net, x, phase='train')


def test_wrong_order_set_parallel_mode_without_initializer():
    """
    Feature: test parameter initialize in auto parallel.
    Description: test parameter initialize in auto parallel not using initializer.
    Expectation: without any assert error.
    """
    weight1 = Tensor(np.ones([32, 32]), ms.float32)
    weight2 = Tensor(np.ones([32, 32]), ms.float32)
    strategy1 = ((2, 1), (4, 1))
    strategy2 = ((2, 4),)
    net = Net(strategy1, strategy2, weight1, weight2)
    exe = me._cell_graph_executor
    x = Tensor(np.ones([32, 32]), dtype=ms.float32)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    exe.compile(net, x, phase='train')
