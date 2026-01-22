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
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore import context
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from mindspore import Symbol
from mindspore.nn.wrap.cell_wrapper import _TrainGradAccuStepCell
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def compile_net(net, *inputs):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    phase, _ = _cell_graph_executor.compile(train_net, *inputs)
    return phase


def test_custom_op_all_tensor_all_tuple_param():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple_param(x, y, z, out, dout):
        dz = dout * (out / z[0])
        dq = dout * (out / (x + y[0]))
        dx = dq
        dy = dq
        return dx, (dy, dy), (dz, dz)

    def python_add_mul_tuple_param(x, y, z):
        q = x + y[0]
        out = q * z[0]
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.custom_weight1 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight1")
            self.custom_weight2 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight2")
            self.custom_weight3 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight3")
            self.custom_weight4 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight4")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple_param, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple_param)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, (self.custom_weight1, self.custom_weight2),
                                 (self.custom_weight3, self.custom_weight4))
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), (layout("dp", "None"), layout("dp", "None")),
        (layout("dp", "None"), layout("dp", "None")))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['_MirrorOperator-0'])
    assert validator.check_node_inputs_has('MakeTuple-1', ['_MirrorOperator-1', '_MirrorOperator-2'])
    assert validator.check_node_inputs_has('MakeTuple-2', ['_MirrorOperator-3', '_MirrorOperator-4'])


def test_custom_op_all_tensor_single_tuple_param():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple_param(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y[0]))
        dx = dq
        dy = dq
        return dx, (dy, dy), dz

    def python_add_mul_tuple_param(x, y, z):
        q = x + y[0]
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.custom_weight1 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight1")
            self.custom_weight2 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight2")
            self.custom_weight3 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight3")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple_param, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple_param)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, (self.custom_weight1, self.custom_weight2), self.custom_weight3)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), (layout("dp", "None"), layout("dp", "None")),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['_MirrorOperator-0'])
    assert validator.check_node_inputs_has('MakeTuple-1', ['_MirrorOperator-1', '_MirrorOperator-2'])
    assert validator.check_node_inputs_has('Custom-0', ['_MirrorOperator-3'])


def test_custom_op_all_tensor_no_tuple_param():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple_param(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul_tuple_param(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.custom_weight1 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight1")
            self.custom_weight2 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight2")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple_param, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple_param)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, self.custom_weight1, self.custom_weight2)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['_MirrorOperator-0'])
    assert validator.check_node_inputs_has('Custom-0', ['_MirrorOperator-1', '_MirrorOperator-2'])


def test_custom_op_all_tensor_no_tuple_param_grad_accu():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple_param(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul_tuple_param(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.custom_weight1 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight1")
            self.custom_weight2 = Parameter(Tensor(np.random.rand(8, 8).astype(np.float32)), name="custom_weight2")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple_param, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple_param)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, self.custom_weight1, self.custom_weight2)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(16, 1).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = nn.GradAccumulationCell(Net(in_layout, out_layout), 2)
    optimizer = Momentum(parallel_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = _TrainGradAccuStepCell(parallel_net, optimizer)
    train_net.set_train()
    phase, _ = _cell_graph_executor.compile(train_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Load-1', ['_MirrorMicroStepOperator-1'])
    assert validator.check_node_inputs_has('Load-2', ['_MirrorMicroStepOperator-2'])


def test_custom_op_all_tensor_single_tuple():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y[0]))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul_tuple(x, y, z):
        q = x + y[0]
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob_tuple)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, y, z)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    y = [Tensor(np.random.rand(8, 8).astype(np.float32))] * 2
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), (layout("dp", "None"), layout("dp", "None")),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, y, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MakeTuple-1', ['_GetTensorSlice-0', '_GetTensorSlice-1'])


def test_custom_op_all_tensor_all_tuple():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob_tuple(x, y, z, out, dout):
        dz = dout * (out / z[0])
        dq = dout * (out / (x[0] + y[0]))
        dx = dq
        dy = dq
        return (dx, dx), dy, dz

    def python_add_mul_tuple(x, y, z):
        q = x[0] + y[0]
        out = q * z[0]
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul_tuple, lambda x, y, z: x[0], lambda x, y, z: x[0],
                                      func_type="pyfunc",
                                      bprop=bprob_tuple)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op([x, x], y, z)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    y = [Tensor(np.random.rand(8, 8).astype(np.float32))] * 2
    z = [Tensor(np.random.rand(8, 8).astype(np.float32))] * 2
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        (layout("dp", "None"), layout("dp", "None")), (layout("dp", "None"), layout("dp", "None")),
        (layout("dp", "None"), layout("dp", "None")))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, y, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MakeTuple-1', ['_GetTensorSlice-0', '_GetTensorSlice-1'])
    assert validator.check_node_inputs_has('MakeTuple-2', ['_GetTensorSlice-2', '_GetTensorSlice-3'])


def test_custom_op_all_tensor_no_tuple():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, y, z)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    y = Tensor(np.random.rand(8, 8).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, y, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['_MirrorOperator-0'])


def test_custom_op_all_tensor_reduce():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, out, dout):
        dout = dout.expand_dims(-1)
        return (dout * x,)

    def python_add_mul(x):
        out = np.mean(x, -1)
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x: x[:-1], lambda x: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"),)
    out_layout = (layout("dp"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Custom-0', ['AllGather-0'])


def test_custom_op_one_scalar():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, 1, z)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"), layout("None"), layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Custom-0', ['AllGather-0', 'StridedSlice-1'])


def test_custom_op_two_scalar():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, 1, 2)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"), layout("None"), layout("None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Custom-0', ['AllGather-0'])


def test_custom_op_tuple_scalar():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y[0]))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y[0]
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, (1, 1), z)
            out = self.relu(out)
            return out

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (layout("dp", "None"), layout("None"), layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Custom-0', ['AllGather-0', 'StridedSlice-1'])


def test_custom_op_all_tensor_multi_out():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        return dout[0], dout[0], dout[0]

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out, q

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: (x, x),
                                      lambda x, y, z: (x, x),
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out, q = self.custom_op(x, y, z)
            out = self.relu(out)
            q = self.relu(q)
            result = out + q
            return result

    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    y = Tensor(np.random.rand(8, 8).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((4, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"), layout("dp", "None"))
    parallel_net = Net(in_layout, out_layout)
    phase = compile_net(parallel_net, x, y, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('MatMul-0', ['_MirrorOperator-0'])


def test_custom_op_all_tensor_no_tuple_dynamic():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(1, 8).astype(np.float32)), name="matmul_weight")
            self.matmul = P.MatMul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.matmul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, y, z)
            out = self.relu(out)
            return out

    s1 = Symbol(divisor=8)
    x = Tensor(shape=[s1, 1], dtype=ms.float32)
    y = Tensor(shape=[s1, 8], dtype=ms.float32)
    z = Tensor(shape=[s1, 8], dtype=ms.float32)

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch")
    layout = Layout((8, 1), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net = Net(in_layout, out_layout)
    compile_net(parallel_net, x, y, z)


def test_custom_op_all_tensor_no_tuple_pp():
    """
    Feature: Test custom op param and tensor (mirror).
    Description: Test custom op with tensor and parameter inputs.
    Expectation: allreduce inserted
    """

    def bprob(x, y, z, out, dout):
        dz = dout * (out / z)
        dq = dout * (out / (x + y))
        dx = dq
        dy = dq
        return dx, dy, dz

    def python_add_mul(x, y, z):
        q = x + y
        out = q * z
        return out

    class Net(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(Net, self).__init__()
            np.random.seed(5)
            self.matmul_weight = Parameter(Tensor(np.random.rand(4, 8).astype(np.float32)), name="mul_weight")
            self.mul = P.Mul()
            self.custom_op = P.Custom(python_add_mul, lambda x, y, z: x, lambda x, y, z: x,
                                      func_type="pyfunc",
                                      bprop=bprob)
            self.custom_op.shard(in_layout, out_layout)
            self.relu = P.ReLU()

        def construct(self, x, y, z):
            x = self.mul(x, self.matmul_weight)
            x = self.relu(x)
            out = self.custom_op(x, y, z)
            out = self.relu(out)
            return out

    class PPNet(nn.Cell):
        def __init__(self, in_layout=None, out_layout=None):
            super(PPNet, self).__init__()
            self.block = nn.CellList()
            self.num_block = 4
            for _ in range(self.num_block):
                net = Net(in_layout, out_layout)
                self.block.append(net)

        def construct(self, x, y, z):
            for i in range(self.num_block):
                x = self.block[i](x, y, z)
            return x

    def compile_net_pp(net, *inputs):
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        train_net = TrainOneStepCell(net, optimizer)
        train_net.set_train()
        phase, _ = _cell_graph_executor.compile(train_net, *inputs)
        return phase

    x = Tensor(np.random.rand(8, 8).astype(np.float32))
    y = Tensor(np.random.rand(8, 8).astype(np.float32))
    z = Tensor(np.random.rand(8, 8).astype(np.float32))
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy="full_batch",
                                      pipeline_stages=2)
    layout = Layout((2, 2), ("dp", "mp"))
    in_layout = (
        layout("dp", "None"), layout("dp", "None"),
        layout("dp", "None"))
    out_layout = (layout("dp", "None"),)
    parallel_net_pp = PPNet(in_layout, out_layout)
    parallel_net_pp.block[0].pipeline_stage = 0
    parallel_net_pp.block[1].pipeline_stage = 0
    parallel_net_pp.block[2].pipeline_stage = 1
    parallel_net_pp.block[3].pipeline_stage = 1
    parallel_net = nn.PipelineCell(parallel_net_pp, 2)
    phase = compile_net_pp(parallel_net, x, y, z)
    validator = ParallelValidator(parallel_net, phase)
    assert validator.check_node_inputs_has('Custom-0', ['AllGather-0'])
