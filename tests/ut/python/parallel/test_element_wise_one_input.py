# Copyright 2025 Huawei Technologies Co., Ltd
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

import re
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops.auto_generate.gen_ops_prim import SoftplusExt, ReLU, EluExt, LeakyReLUExt, \
                                                    Identity, NanToNum, RemainderTensorScalar, \
                                                    RemainderScalarTensor, Mul
from mindspore.parallel.shard import Layout

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class SoftplusNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.softplus_op = SoftplusExt().shard(strategy)
        else:
            self.softplus_op = SoftplusExt()

    def construct(self, input_data, beta=0.1, threshold=20.0):
        return self.softplus_op(input_data, beta, threshold)

class ReLUNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.relu_op = ReLU().shard(strategy)
        else:
            self.relu_op = ReLU()

    def construct(self, input_data):
        return self.relu_op(input_data)

class EluNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.elu_op = EluExt().shard(strategy)
        else:
            self.elu_op = EluExt()

    def construct(self, input_data):
        return self.elu_op(input_data)

class LeakyReLUNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.leaky_relu_op = LeakyReLUExt().shard(strategy)
        else:
            self.leaky_relu_op = LeakyReLUExt()

    def construct(self, input_data):
        return self.leaky_relu_op(input_data)

class IdentityNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.identity_op = Identity().shard(strategy)
        else:
            self.identity_op = Identity()

    def construct(self, input_data):
        return self.identity_op(input_data)

class NanToNumNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.nantonum_op = NanToNum().shard(strategy)
        else:
            self.nantonum_op = NanToNum()

    def construct(self, input_data):
        return self.nantonum_op(input_data)

class RemainderTSNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.remainderts_op = RemainderTensorScalar().shard(strategy)
        else:
            self.remainderts_op = RemainderTensorScalar()

    def construct(self, input_data, scalar):
        return self.remainderts_op(input_data, scalar)

class RemainderSTNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.remainderst_op = RemainderScalarTensor().shard(strategy)
        else:
            self.remainderst_op = RemainderScalarTensor()

    def construct(self, scalar, input_data):
        return self.remainderst_op(scalar, input_data)

def compile_graph(net, input_data, device_num=8, parallel_mode="semi_auto_parallel"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data)
    return phase

def test_softplus_default_values():
    """
    Feature: distribute operator softplus_ext in semi auto parallel.
    Description: Basic functionality with default parameters.
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = SoftplusNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data)

def test_softplus_shard_strategy_error():
    """
    Feature: test softplusext parallel error strategy.
    Description: Invalid shard strategy.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (1,))
    net = SoftplusNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_graph(net, input_data)

def test_softplus_auto_parallel():
    """
    Features: test softplusext auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_context(save_graphs=True)
    net = SoftplusNet()
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data, device_num=8, parallel_mode="auto_parallel")

def test_softplus_layout_extend():
    """
    Feature: test softplusext layout extend
    Description: layout extend
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout(("dp", "cp"), "None"),)
    net = SoftplusNet(int_layout)
    input_data = Tensor(np.zeros((16, 16)), dtype=ms.float32)
    compile_graph(net, input_data)

def test_relu_default_values():
    """
    Feature: distribute operator ReLU in semi auto parallel.
    Description: Basic functionality of ReLU.
    Expectation: Compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = ReLUNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data)

def test_relu_shard_strategy_error():
    """
    Feature: test parallel error strategy for ReLU.
    Description: Invalid shard strategy.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (1,))
    net = ReLUNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_graph(net, input_data)

def test_relu_auto_parallel():
    """
    Feature: test ReLU auto parallel.
    Description: auto parallel.
    Expectation: compile success.
    """
    context.set_context(save_graphs=True)
    net = ReLUNet()
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data, device_num=8, parallel_mode="semi_auto_parallel")

def test_relu_layout_extend():
    """
    Feature: test ReLU layout extend.
    Description: layout extend.
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout(("dp", "cp"), "None"),)
    net = ReLUNet(int_layout)
    input_data = Tensor(np.zeros((16, 16)), dtype=ms.float32)
    compile_graph(net, input_data)

def test_elu_default_values():
    """
    Feature: distribute operator EluExt in semi auto parallel.
    Description: Basic functionality of EluExt.
    Expectation: Compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = EluNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data)

def test_leakyrelu_default_values():
    """
    Feature: distribute operator LeakyReLUExt in semi auto parallel.
    Description: Basic functionality of LeakyReLUExt.
    Expectation: Compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = LeakyReLUNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data)

def test_identity_default_values():
    """
    Feature: distribute operator Identity in semi auto parallel.
    Description: Basic functionality of Identity.
    Expectation: Compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = IdentityNet(strategy=strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    compile_graph(net, input_data)

def test_nantonum_default_values():
    """
    Feature: distribute operator NanToNum in semi auto parallel.
    Description: Basic functionality of NanToNum.
    Expectation: Compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = NanToNumNet(strategy=strategy)
    input_data = Tensor(np.array([[1.0, 2.0, 3.0, 4.0],
                                  [np.inf, np.inf, np.inf, np.inf],
                                  [-np.inf, -np.inf, -np.inf, -np.inf],
                                  [np.nan, np.nan, np.nan, np.nan]]), dtype=ms.float32)
    compile_graph(net, input_data)

class Net(nn.Cell):
    def __init__(self, strategy):
        super().__init__()
        self.scalar = 1
        self.mul = Mul().shard(strategy)
        self.relu_op = ReLU()
        self.elu_op = EluExt()
        self.leaky_relu_op = LeakyReLUExt()
        self.softplus_op = SoftplusExt()
        self.remaindertensorscalar_op = RemainderTensorScalar()
        self.identity_op = Identity()
        self.nantonum_op = NanToNum()

    def construct(self, x):
        out = self.mul(x, x)
        out = self.relu_op(out)
        out = self.elu_op(out)
        out = self.leaky_relu_op(out)
        out = self.softplus_op(out)
        out = self.remaindertensorscalar_op(out, self.scalar)
        out = self.identity_op(out)
        out = self.nantonum_op(out)
        return out

def test_element_wise_single_input_ops():
    """
    Features: test sharding propagation for element wise ops with a single input.
    Description:
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4, 8), (1, 2, 4, 8))
    net = Net(strategy=strategy)
    _x = Tensor(np.ones([4, 8, 4, 8]), dtype=ms.float32)
    _cell_graph_executor.compile(net, _x, phase='train')

    # Get strategies
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("SoftplusExt", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("EluExt", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("LeakyReLUExt", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("ReLU", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("RemainderTensorScalar", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("Identity", k) is not None:
            assert v == [[1, 2, 4, 8]]
        elif re.search("NanToNum", k) is not None:
            assert v == [[1, 2, 4, 8]]

def test_remaindarTS_default_values():
    """
    Feature: distribute operator RemainderTensorScalar in semi auto parallel.
    Description: Basic functionality of RemainderTensorScalar.
    Expectation: Compile done without error.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy = ((1, 4),)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    scalar = 1
    net = RemainderTSNet(strategy=strategy)
    net.set_train()
    _cell_graph_executor.compile(net, input_data, scalar)

def test_remaindarST_default_values():
    """
    Feature: distribute operator RemainderScalarTensor in semi auto parallel.
    Description: Basic functionality of RemainderScalarTensor.
    Expectation: Compile done without error.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy = ((1, 4),)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    scalar = 1
    net = RemainderSTNet(strategy=strategy)
    net.set_train()
    _cell_graph_executor.compile(net, scalar, input_data)

def test_remainderST_auto_parallel():
    """
    Feature: test remainderST auto parallel.
    Description: auto parallel.
    Expectation: compile success.
    """
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel")
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    scalar = 1
    net = RemainderSTNet()
    net.set_train()
    _cell_graph_executor.compile(net, scalar, input_data)

def test_remainderST_layout_extend():
    """
    Feature: test remainderST layout extend.
    Description: layout extend.
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout("dp", "cp"),)
    net = RemainderSTNet(int_layout)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    scalar = 1
    _cell_graph_executor.compile(net, scalar, input_data)

def test_remainderTS_layout_extend():
    """
    Feature: test remainderTS layout extend.
    Description: layout extend.
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout("dp", "cp"),)
    net = RemainderTSNet(int_layout)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    scalar = 1
    _cell_graph_executor.compile(net, input_data, scalar)
