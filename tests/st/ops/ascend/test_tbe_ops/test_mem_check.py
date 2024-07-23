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
from tests.mark_utils import arg_mark
import os
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P


class GatherNet(nn.Cell):
    def __init__(self):
        super(GatherNet, self).__init__()
        self.gather = P.Gather()

    def construct(self, input_x, indices, axis):
        return self.gather(input_x, indices, axis)


class AddNet(nn.Cell):
    def __init__(self):
        super(AddNet, self).__init__()
        self.add = P.Add()

    def construct(self, x, y):
        return self.add(x, y)


class NetSort(nn.Cell):
    def __init__(self):
        super(NetSort, self).__init__()
        self.sort = ops.Sort()

    def construct(self, x, y):
        x += y
        y = self.sort(x)
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gather_mem_check():
    """
    Feature: mem check.
    Description: gather op mem check success.
    Expectation: the result equal to expect.
    """
    cfg_path = os.path.abspath("./test_mem_check.cfg")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        ascend_config={"ge_options":
                                       {"global": {"op_debug_config": cfg_path}}})
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]), mindspore.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6, 0, 2, 4, 2, 6, 0, 2, 4, 2, 6]), mindspore.int32)
    axis = 0
    output = ops.gather(input_params, input_indices, axis)
    expect_np = np.array([1., 3., 5., 3., 7., 1., 3., 5., 3., 7., 1., 3., 5., 3., 7.])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_add_mem_check():
    """
    Feature: mem check.
    Description: add op mem check success.
    Expectation: the result equal to expect.
    """
    cfg_path = os.path.abspath("./test_mem_check.cfg")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        ascend_config={"ge_options":
                                       {"global": {"op_debug_config": cfg_path}}})
    x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 1, 2]), mindspore.float32)
    y = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 1, 2]), mindspore.float32)
    net = AddNet()
    output = net(x, y)
    expect_np = np.array([2, 4, 6, 8, 10, 12, 14, 2, 4])
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sort_mem_check_graph():
    """
    Feature: mem check in graph mode.
    Description: sort op mem check fail.
    Expectation: RuntimeError.
    """
    cfg_path = os.path.abspath("./test_mem_check.cfg")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        ascend_config={"ge_options":
                                       {"global": {"op_debug_config": cfg_path}}})
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    tensor_x = Tensor(np.random.random([18018]), mindspore.float16)
    tensor_y = Tensor(np.random.random([18018]), mindspore.float16)
    net = NetSort()
    with pytest.raises(RuntimeError):
        y = net(tensor_x, tensor_y)
        print(y)


class NetSoftMax(nn.Cell):
    def __init__(self):
        super(NetSoftMax, self).__init__()
        self.softmax = P.SoftmaxCrossEntropyWithLogits()

    def construct(self, features, labels):
        return self.softmax(features, labels)


def test_softmax_mem_check_graph():
    """
    Feature: mem check in graph mode.
    Description: sort op mem check fail.
    Expectation: RuntimeError.
    skip because tbe compiler error
    """
    cfg_path = os.path.abspath("./test_mem_check.cfg")
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        ascend_config={"ge_options":
                                       {"global": {"op_debug_config": cfg_path}}})
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    features = np.random.randn(32, 1000).astype(np.float32)
    labels = np.random.randn(32, 1000).astype(np.float32)
    net_softmax = NetSoftMax()
    output = net_softmax(Tensor(features), Tensor(labels))
    print(output.asnumpy())
