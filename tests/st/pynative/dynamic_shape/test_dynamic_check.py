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
# ============================================================================

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore.common.parameter import Parameter, ParameterTuple
from tests.st.pynative.utils.tools import clean_all_ir_files, count_ir_files_num, clear_folder, \
    get_flag_from_ir_file_line


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


def teardown_module():
    context.set_context(save_graphs=False)


class NetInner(nn.Cell):
    def __init__(self):
        super(NetInner, self).__init__()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()
        self.para1 = Parameter(Tensor([2, 3, 4, 5], ms.float32), name="para1")
        self.para2 = Parameter(Tensor([2, 3, 4, 5], ms.float32), name="para2")

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.addn((x, self.para1))
        x = self.relu(x)
        x = self.addn((x, self.para2))
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.addn = ops.AddN()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.log(x)
        x = self.exp(x)
        x = self.relu(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_input_shape_changed():
    """
    Feature: PyNative dynamic shape check.
    Description: Top cell input shape changed, net is dynamic.
    Expectation: Dynamic check is detected.
    """
    net = Net()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    context.set_context(save_graphs=True, save_graphs_path="ir")

    try:
        clean_all_ir_files("ir")
        # run first shape, save the first launch_bprop_graph.ir
        input_x = Tensor(np.random.rand(2, 3, 6, 4).astype(np.float32) * 2)
        input_y = Tensor(np.random.rand(2, 3, 6, 4).astype(np.float32) * 5)
        _ = grad_op(net)(input_x, input_y)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 1
        assert get_flag_from_ir_file_line("ir", "launch_bprop_graph", "enable_run_graph_by_single_op") == 0

        # run second shape, store 2 static shape
        input_x2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 2)
        input_y2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 5)
        _ = grad_op(net)(input_x2, input_y2)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 2
        assert get_flag_from_ir_file_line("ir", "launch_bprop_graph", "enable_run_graph_by_single_op") == 0

        clean_all_ir_files("ir")
        # run third shape, set top cell use dynamic, use func grad
        input_x = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 2)
        input_y = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 5)
        _ = grad_op(net)(input_x, input_y)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 0
    finally:
        clear_folder("ir")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_parameter_requires_grad_changed():
    """
    Feature: PyNative auto dynamic check.
    Description: Check parameter requires grad.
    Expectation: The result is dynamic.
    """
    net = NetInner()
    grad_op = ops.GradOperation(get_all=False, get_by_list=True, sens_param=False)
    net_params = ParameterTuple(net.trainable_params())

    # run first time
    input_x = Tensor([1, 2, 3, 4], ms.float32) * 2
    input_y = Tensor([1, 2, 3, 4], ms.float32) * 3
    grad1 = grad_op(net, net_params)(input_x, input_y)
    assert len(grad1) == 2
    assert np.allclose(grad1[0].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)
    assert np.allclose(grad1[1].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)
    # run second time
    for p in net_params:
        p.requires_grad = False
        break
    net_params = ParameterTuple(net.trainable_params())
    grad2 = grad_op(net, net_params)(input_x, input_y)
    assert len(grad2) == 1
    assert np.allclose(grad2[0].asnumpy(), Tensor(np.array([1, 1, 1, 1])).astype(np.float32).asnumpy(),
                       0.001, 0.001)


class BpropNet(nn.Cell):
    def __init__(self):
        super(BpropNet, self).__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.relu(x)
        return out

    def bprop(self, x, out, dout):
        grads = x * 2
        return (grads,)


class NetHasBprop(nn.Cell):
    def __init__(self):
        super(NetHasBprop, self).__init__()
        self.exp = ops.Exp()
        self.relu = nn.ReLU()
        self.inner = BpropNet()

    def construct(self, x):
        x = self.exp(x)
        x = self.inner(x)
        x = self.relu(x)
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_net_bprop_dynamic_shape():
    """
    Feature: PyNative bporp dynamic shape.
    Description: Net has custom bprop is dynamic.
    Expectation: The net is dynamic.
    """
    net = NetHasBprop()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    context.set_context(save_graphs=True, save_graphs_path="ir")

    try:
        clean_all_ir_files("ir")
        # run first shape, save the first launch_bprop_graph.ir
        input_x = Tensor(np.random.rand(2, 3, 6, 4).astype(np.float32) * 2)
        _ = grad_op(net)(input_x)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 1
        assert get_flag_from_ir_file_line("ir", "launch_bprop_graph", "enable_run_graph_by_single_op") == 1

        # run second shape, dynamic shape
        clean_all_ir_files("ir")
        input_x2 = Tensor(np.random.rand(2, 3, 6, 16).astype(np.float32) * 2)
        _ = grad_op(net)(input_x2)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 0

        # run third shape, dynamic shape
        input_x = Tensor(np.random.rand(2, 3, 6, 8).astype(np.float32) * 2)
        _ = grad_op(net)(input_x)
        assert count_ir_files_num("ir", "launch_bprop_graph") == 0
    finally:
        clear_folder("ir")
