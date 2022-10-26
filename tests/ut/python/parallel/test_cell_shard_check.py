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
import numpy as np
import pytest

import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, context, Tensor, shard


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def set_context():
    context.set_context(mode=context.PYNATIVE_MODE)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, parallel_mode="auto_parallel", search_mode="sharding_propagation")


class NetMul(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        return self.mul(x, y)


class NetMatMul(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        return self.matmul(x, y)


class Net(nn.Cell):
    def __init__(self, in_strategy, out_strategy=None):
        super().__init__()
        self.mul_net = NetMul()
        self.matmul_net = NetMatMul()
        self.mul_net.shard(in_strategy=in_strategy, out_strategy=out_strategy)

    def construct(self, x, y):
        out1 = self.matmul_net(x, y)
        out2 = self.matmul_net(x, y)
        return self.mul_net(out1, out2)


def cell_shard_execution(in_strategy, out_strategy, error_log):
    net = Net(in_strategy, out_strategy)
    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([128, 128]), dtype=ms.float32)

    with pytest.raises(Exception) as err:
        _ = net(x, y)
    assert error_log in str(err.value)


def add_one_func(x):
    return x + 1


def test_in_strategy_numbers_check():
    """
    Feature: shard function for cell
    Description: inconsistent input number and in_strategy number
    Expectation: throw an exception indicating inconsistent input number and in_strategy number
    """
    set_context()
    in_strategy = ((8, 1), None, (1, 8))
    out_strategy = (None,)
    error_log = "Input numbers: 2 is not equal to in_strategy numbers: 3"
    cell_shard_execution(in_strategy, out_strategy, error_log)


def test_in_strategy_dimension_check():
    """
    Feature: shard function for cell
    Description: inconsistent input dimension and in_strategy dimension
    Expectation: throw an exception indicating inconsistent input_dimension and in_strategy dimension
    """
    set_context()
    in_strategy = ((8, 1, 1), None)
    out_strategy = (None, (8, 1))
    error_log = "Input dimension: 2 is not equal to in_strategy dimension: 3 at index 0"
    cell_shard_execution(in_strategy, out_strategy, error_log)


def test_in_strategy_format_check():
    """
    Feature: shard function for cell
    Description: unsupported in_strategy format
    Expectation: throw an exception indicating an supported in_strategy format
    """
    set_context()
    in_strategy = ([8, 1], None)
    out_strategy = (None,)
    error_log = "in_strategy should be a two-dimension tuple"
    cell_shard_execution(in_strategy, out_strategy, error_log)


def test_vmap_nested_shard_check():
    """
    Feature: vmap nested shard
    Description: test usage of vmap nested shard
    Expectation: throw an exception indicating that vmap nested shard is invalid usage
    """
    set_context()
    x = Tensor(np.ones([16, 8]), ms.float32)
    in_strategy = ((8,),)
    func = ops.vmap(shard(add_one_func, in_strategy=in_strategy), in_axes=0, out_axes=0)
    error_log = "The usage of vmap nested shard (e.g vmap(shard)) is not supported currently. Current FuncGraph:"
    with pytest.raises(Exception) as err:
        _ = func(x)
    assert error_log in str(err.value)


def test_shard_nested_shard_check():
    """
    Feature: shard nested shard
    Description: test usage of shard nested shard
    Expectation: throw an exception indicating that shard nested shard is invalid usage
    """
    set_context()
    x = Tensor(np.ones([16, 8]), ms.float32)
    in_strategy = ((8, 1),)
    func = shard(shard(add_one_func, in_strategy=in_strategy), in_strategy=in_strategy)
    error_log = "Nested use of shard (e.g shard(shard(...), ...) is not supported currently."
    with pytest.raises(Exception) as err:
        _ = func(x)
    assert error_log in str(err.value)
