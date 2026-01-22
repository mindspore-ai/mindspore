
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
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.parallel.shard import Layout

from .test_finegraint_micro_interleaved import GradWrap, NetWithLoss, NetWithAdd2


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self,
                 strategy1=None,
                 strategy2=None,
                 strategy3=None,
                 axis=0,
                 init_flag=True,
                 split_tuple=(4, 4),
                 split_string="manual_split",
                 param_shape=(8, 8)):
        super().__init__()
        self.gatherv2 = P.Gather().shard(strategy1)
        self.gatherv2.add_prim_attr(split_string, split_tuple)
        self.mul = P.Mul().shard(strategy2)
        self.reshape = P.Reshape()
        self.matmul = P.MatMul().shard(strategy3)
        self.matmul.add_prim_attr("forward_reduce_scatter", True)
        if init_flag:
            self.param = Parameter(initializer("ones", param_shape, ms.float32), name="gatherv2_param")
        else:
            self.param = Parameter(Tensor(np.ones(param_shape), dtype=ms.float32), name="gatherv2_param")
        self.mul_weight = Parameter(initializer("ones", (8, 8, 8), ms.float32), name="mul_weight")
        self.matmul_weight = Parameter(initializer("ones", (64, 16), ms.float32), name="matmul_weight")
        self.axis = axis

    def construct(self, x, b):
        out = self.gatherv2(self.param, x, self.axis)
        out = self.mul(out, self.mul_weight)
        out = self.reshape(out, (8, 64))
        out = self.matmul(out, self.matmul_weight)
        return out


class Net2(Cell):
    def __init__(self, add_weight, kernel_size=1, strides=1, pad_mode="valid", pad_list=0, ceil_mode=None,
                 data_format="NCDHW", strategy0=None, strategy1=None, strategy2=None):
        super().__init__()
        self.add = P.Add().shard(strategy0)
        self.max_pool = P.MaxPool3D(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode, pad_list=pad_list,
                                    ceil_mode=ceil_mode, data_format=data_format).shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)
        self.add_weight = Parameter(add_weight, "w1")

    def construct(self, x, b):
        out = self.add(x, self.add_weight)
        out = self.max_pool(out)
        out = self.relu(out)
        return out


class Net3(Cell):
    def __init__(self, conv2d_weight, out_channel, kernel_size, pad_mode, stride, dilation=1, group=1, pad=0,
                 strategy1=None, strategy2=None):
        super().__init__()
        self.conv2d = P.Conv2D(out_channel=out_channel, kernel_size=kernel_size, pad_mode=pad_mode, pad=pad,
                               stride=stride, dilation=dilation, group=group).shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.conv2d_weight = Parameter(conv2d_weight, "w1")

    def construct(self, x, b):
        out = self.conv2d(x, self.conv2d_weight)
        out = self.neg(out)
        return out


class Net4(nn.Cell):
    def __init__(self, in_layout, out_layout, self_define_shard=True):
        super().__init__()
        self.tensor_scatter_update = P.TensorScatterUpdate()
        self.tensor_scatter_update.shard(in_strategy=in_layout, out_strategy=out_layout)
        self.tensor_scatter_update.add_prim_attr("self_define_shard", self_define_shard)
        self.relu = P.ReLU()
        self.mul = P.Mul()

    def construct(self, input_x, indices, update):
        out = self.relu(input_x)
        out = self.tensor_scatter_update(out, indices, update)
        out = self.mul(out, 2)
        return out


def compile_net(net: Cell, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()


def test_exception_log_with_node_1():
    """
    Feature: test exception log with node
    Description: raise error and check error msg
    Expectation: specific error python code lines can be printed
    """
    # fork from test_manual_gatherv2.py::test_auto_parallel_error
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=2,
                                      global_rank=0)
    net = Net()
    _x = Tensor(np.ones([8, 8]), dtype=ms.int32)
    _b = Tensor(np.ones([64, 8]), dtype=ms.float32)
    with pytest.raises(RuntimeError) as exec_info:
        compile_net(net, _x, _b)
    error_info = str(exec_info.value)
    index = error_info.find('self.gatherv2(self.param, x, self.axis)')
    assert index != -1


def test_exception_log_with_node_2():
    """
    Feature: test exception log with node
    Description: raise error and check error msg
    Expectation: specific error python code lines can be printed
    """
    # fork from test_maxpool_3d.py::test_maxpool3d_valid_mode_output_shape_cannot_div_by_strategy
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1, 1, 8, 1),)
    strategy2 = ((1, 1, 1, 1, 1),)
    _x = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
    _w = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 16, 16, 16, 16]), dtype=ms.float32)
    net = Net2(_w, kernel_size=2, pad_mode="valid", strides=4,
               strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError) as exec_info:
        compile_net(net, _x, _b)
    error_info = str(exec_info.value)
    index = error_info.find('self.add(x, self.add_weight)')
    assert index != -1


def test_exception_log_with_node_3():
    """
    Feature: test exception log with node
    Description: raise error and check error msg
    Expectation: specific error python code lines can be printed
    """
    # fork from test_conv2d.py::test_conv2d_data_parallel_invalid_stride
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((8, 1, 1, 1), (1, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    _x = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    _w1 = Tensor(np.ones([8, 16, 2, 2]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 16, 8, 8]), dtype=ms.float32)
    net = Net3(_w1, out_channel=8, kernel_size=2, pad_mode="same", stride=(2, 2, 1, 1),
               strategy1=strategy1, strategy2=strategy2)
    with pytest.raises(RuntimeError) as exec_info:
        compile_net(net, _x, _b)
    error_info = str(exec_info.value)
    index = error_info.find('out = self.neg(out)')
    assert index != -1


def test_internal_exception_log_with_node_1():
    """
    Feature: test internal exception log with node
    Description: raise error and check error msg
    Expectation: specific error python code lines can be printed
    """
    # fork from test_finegraint_micro_interleaved.py::test_interleaved_with_add_failed
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    add_layout = (layout("None"), layout(("dp", "interleaved_parallel"), "None"))
    x = Tensor(np.ones([1024,]), dtype=ms.float32)
    bias = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAdd2(bias, add_layout)))
    with pytest.raises(RuntimeError) as exec_info:
        compile_net(net, x)
    error_info = str(exec_info.value)
    index = error_info.find('self.add(y, self.bias)')
    assert index != -1


def test_internal_exception_log_with_node_2():
    """
    Feature: test internal exception log with node
    Description: raise error and check error msg
    Expectation: specific error python code lines can be printed
    """
    # fork from test_config_layout_for_arbitrary_ops.py::test_config_layout_for_ops_not_dividable
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    layout = Layout((4, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = Net4(layout1, layout2, False)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    with pytest.raises(RuntimeError) as exec_info:
        compile_net(net, input_x, indices, update)
    error_info = str(exec_info.value)
    index = error_info.find('self.tensor_scatter_update(out, indices, update)')
    assert index != -1
