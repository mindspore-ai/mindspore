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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.parallel.shard import Layout
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", dataset_strategy="full_batch"
    )


class Net(Cell):
    def __init__(
            self,
            mul_weight1,
            mul_weight2,
            transpose_b=False,
            strategy1=None,
            strategy2=None
    ):
        super().__init__()
        self.mul1 = P.MatMul(transpose_b=transpose_b).shard(in_strategy=strategy1)
        self.mul2 = P.MatMul(transpose_b=transpose_b).shard(in_strategy=strategy2)
        self.mul1.add_prim_attr("enable_nd_tp", True)
        self.mul2.add_prim_attr("enable_nd_tp", True)
        self.mul_weight1 = Parameter(mul_weight1, "w1")
        self.mul_weight2 = Parameter(mul_weight2, "w2")

    def construct(self, x):
        out = self.mul1(x, self.mul_weight1)
        out = self.mul2(out, self.mul_weight2)
        return out


class BatchNet(Cell):
    def __init__(
            self,
            batch_matmul_weight1,
            batch_matmul_weight2,
            transpose_b=False,
            strategy1=None,
            strategy2=None
    ):
        super().__init__()
        self.batch_matmul1 = P.BatchMatMul(transpose_b=transpose_b).shard(
            in_strategy=strategy1
        )
        self.batch_matmul2 = P.BatchMatMul(transpose_b=transpose_b).shard(
            in_strategy=strategy2
        )
        self.batch_matmul1.add_prim_attr("enable_nd_tp", True)
        self.batch_matmul2.add_prim_attr("enable_nd_tp", True)
        self.batch_matmul_weight1 = Parameter(batch_matmul_weight1, "w1")
        self.batch_matmul_weight2 = Parameter(batch_matmul_weight2, "w2")

    def construct(self, x):
        out = self.batch_matmul1(x, self.batch_matmul_weight1)
        out = self.batch_matmul2(out, self.batch_matmul_weight2)
        return out


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 64]), dtype=ms.float32)
_w1_trans = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w2_trans = Tensor(np.ones([128, 32, 64]), dtype=ms.float32)

_x_s = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w1_s = Tensor(np.ones([32, 64]), dtype=ms.float32)
_w2_s = Tensor(np.ones([64, 64]), dtype=ms.float32)
_w1_trans_s = Tensor(np.ones([64, 32]), dtype=ms.float32)
_w2_trans_s = Tensor(np.ones([32, 64]), dtype=ms.float32)

def compile_batch_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x)
    context.reset_auto_parallel_context()

def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x_s)
    context.reset_auto_parallel_context()


def test_batch_matmul_2D_TP():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two batchmatmul
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((1, 2, 4), ("b", "x", "y"))
    net = BatchNet(
        _w1,
        _w2,
        False,
        (layout("b", "x", "y"), layout("b", "y", "x")),
        (layout("b", "y", "x"), layout("b", "x", "y"))
    )
    compile_batch_net(net)


def test_batch_matmul_2D_TP_transpose_b():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two batchmatmul and the Weight is transposed.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((1, 2, 4), ("b", "x", "y"))
    net = BatchNet(
        _w1_trans,
        _w2_trans,
        True,
        (layout("b", "x", "y"), layout("b", "x", "y")),
        (layout("b", "y", "x"), layout("b", "y", "x"))
    )
    compile_batch_net(net)


def test_batch_matmul_2D_TP_reduce_1D():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two batchmatmul, but the one of the dimensions divide num is 1.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((1, 1, 8), ("b", "x", "y"))
    net = BatchNet(
        _w1,
        _w2,
        False,
        (layout("b", "x", "y"), layout("b", "y", "x")),
        (layout("b", "y", "x"), layout("b", "x", "y"))
    )
    compile_batch_net(net)


def test_batch_matmul_3D_TP():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 3D TP, two batchmatmul.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((2, 2, 4), ("z", "x", "y"))
    net = BatchNet(
        _w1,
        _w2,
        False,
        (layout("None", ("z", "x"), "y"), layout("None", ("y", "z"), "x")),
        (layout("None", ("z", "y"), "x"), layout("None", ("x", "z"), "y"))
    )
    compile_batch_net(net)


def test_batch_matmul_3D_TP_transpose_b():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 3D TP, two batchmatmul and the Weight is transposed.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((2, 2, 4), ("z", "x", "y"))
    net = BatchNet(
        _w1_trans,
        _w2_trans,
        True,
        (layout("None", ("z", "x"), "y"), layout("None", "x", ("y", "z"))),
        (layout("None", ("z", "y"), "x"), layout("None", "y", ("x", "z")))
    )
    compile_batch_net(net)


def test_batch_matmul_3D_TP_reduce_2D():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two batchmatmul, but the oz divide num is 1.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((1, 4, 4), ("z", "x", "y"))
    net = BatchNet(
        _w1,
        _w2,
        False,
        (layout("None", ("z", "x"), "y"), layout("None", ("y", "z"), "x")),
        (layout("None", ("z", "y"), "x"), layout("None", ("x", "z"), "y"))
    )
    compile_batch_net(net)


def test_matmul_2D_TP():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two matmul
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((2, 4), ("x", "y"))
    net = Net(
        _w1_s,
        _w2_s,
        False,
        (layout("x", "y"), layout("y", "x")),
        (layout("y", "x"), layout("x", "y"))
    )
    compile_net(net)


def test_matmul_2D_TP_transpose_b():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two matmul and the Weight is transposed.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((2, 4), ("x", "y"))
    net = Net(
        _w1_trans_s,
        _w2_trans_s,
        True,
        (layout("x", "y"), layout("x", "y")),
        (layout("y", "x"), layout("y", "x"))
    )
    compile_net(net)


def test_matmul_2D_TP_reduce_1D():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two matmul, but the one of the dimensions divide num is 1.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0
    )
    layout = Layout((1, 8), ("x", "y"))
    net = Net(
        _w1_s,
        _w2_s,
        False,
        (layout("x", "y"), layout("y", "x")),
        (layout("y", "x"), layout("x", "y"))
    )
    compile_net(net)


def test_matmul_3D_TP():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 3D TP, two matmul.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((2, 2, 4), ("x", "y", "z"))
    net = Net(
        _w1_s,
        _w2_s,
        False,
        (layout(("z", "x"), "y"), layout(("y", "z"), "x")),
        (layout(("z", "y"), "x"), layout(("x", "z"), "y"))
    )
    compile_net(net)


def test_matmul_3D_TP_transpose_b():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 3D TP, two matmul and the Weight is transposed.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((2, 2, 4), ("z", "x", "y"))
    net = Net(
        _w1_trans_s,
        _w2_trans_s,
        True,
        (layout(("z", "x"), "y"), layout("x", ("y", "z"))),
        (layout(("z", "y"), "x"), layout("y", ("x", "z")))
    )
    compile_net(net)


def test_matmul_3D_TP_reduce_2D():
    """
    Feature: 2D/3D tensor parllel.
    Description: test 2D TP, two matmul, but the oz divide num is 1.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0
    )
    layout = Layout((1, 4, 4), ("z", "x", "y"))
    net = Net(
        _w1_s,
        _w2_s,
        False,
        (layout(("z", "x"), "y"), layout(("y", "z"), "x")),
        (layout(("z", "y"), "x"), layout(("x", "z"), "y"))
    )
    compile_net(net)
