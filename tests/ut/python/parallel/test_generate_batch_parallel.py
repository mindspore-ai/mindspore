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
# ============================================================================


import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def test_concat():
    """
    Feature: Test Concat with axis=0 and generate batch parallel strategy.
    Description: axis=0, batch parallel strategy must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.concat = P.Concat()

        def construct(self, x, y):
            out = self.concat((x, y))
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((1, 1), (1, 1))'}
    assert validator.check_node_attrs("Concat-0", expect_strategy)


def test_batch_matmul():
    """
    Feature: Test BatchMatMul with 2-dim weight and generate batch parallel strategy.
    Description: batch parallel strategy of weight must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.batch_matmul = P.BatchMatMul()

        def construct(self, x, y):
            out = self.batch_matmul(x, y)
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    x = Tensor(np.ones([128, 128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((8, 1, 1), (1, 1))'}
    assert validator.check_node_attrs("BatchMatMul-0", expect_strategy)


def test_onehot():
    """
    Feature: Test OneHot with 2-dim input and generate batch parallel strategy.
    Description: batch parallel strategy must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.ont_hot = P.OneHot()
            self.on = Tensor(1.0, ms.float32)
            self.off = Tensor(0.0, ms.float32)

        def construct(self, x, y):
            out = self.ont_hot(x, 1, self.on, self.off)
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", full_batch=True)

    x = Tensor(np.ones([2, 128]), dtype=ms.int32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((1, 1, 1), (), ())'}
    assert validator.check_node_attrs("OneHot-0", expect_strategy)


def test_slice():
    """
    Feature: Test Slice with input no fully fetched and generate batch parallel strategy.
    Description: batch parallel strategy must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.slice = P.Slice()

        def construct(self, x, y):
            out = self.slice(x, (0, 0), (64, 128))
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((1, 1))'}
    assert validator.check_node_attrs("Slice-0", expect_strategy)


def test_strided_slice():
    """
    Feature: Test StridedSlice with input no fully fetched and generate batch parallel strategy.
    Description: batch parallel strategy must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.slice = P.StridedSlice()

        def construct(self, x, y):
            out = self.slice(x, (0, 0), (64, 128), (1, 1))
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((1, 1))'}
    assert validator.check_node_attrs("StridedSlice-0", expect_strategy)


def test_split():
    """
    Feature: Test Split with axis=0 and generate batch parallel strategy.
    Description: batch parallel strategy must be full one.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = P.Split(0, 2)

        def construct(self, x, y):
            out, _ = self.split(x)
            return out
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)
    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 128]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y)
    validator = ParallelValidator(net, phase)
    expect_strategy = {"gen_strategy": '((1, 1))'}
    assert validator.check_node_attrs("Split-0", expect_strategy)


def test_virtual_output():
    """
    Feature: Test virtualoutput with return parameter in predict mode.
    Description: No need to insert virtualoutput.
    Expectation: Successful graph compilation.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(np.ones([32, 32]), ms.float32))

        def construct(self):
            return self.param
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0, full_batch=True)
    net = Net()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    phase, _ = _cell_graph_executor.compile(net)
    validator = ParallelValidator(net, phase)
    sub_graph = {'Return-0': ['param']}
    assert validator.check_graph_structure(sub_graph)
