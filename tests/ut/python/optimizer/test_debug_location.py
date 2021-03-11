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

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.nn.optim import Momentum
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops._grad.grad_math_ops import binop_grad_common
from mindspore.ops._utils import get_broadcast_shape
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

context.set_context(mode=context.GRAPH_MODE)


class MockNeg(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init MockNeg"""
        self.init_prim_io_names(inputs=['x'], outputs=['y'])

    def infer_shape(self, input_x):
        return input_x

    def infer_dtype(self, input_x):
        raise TypeError("InferError")
        # return input_x


class MockSub(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self):
        """init MockSub"""
        self.init_prim_io_names(inputs=['x', 'y'], outputs=['output'])

    def infer_shape(self, x_shape, y_shape):
        return get_broadcast_shape(x_shape, y_shape)

    def infer_dtype(self, x_dtype, y_dtype):
        return x_dtype


@bprop_getters.register(MockSub)
def get_bprop_mock_sub(self):
    """Grad definition for `MockSub` operation."""
    neg_func = MockNeg()

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, neg_func(dout))

    return bprop


class Net(nn.Cell):
    def __init__(self, in_features, out_features):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.ones([out_features, in_features]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([out_features]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.add = P.Add()

    def construct(self, input_):
        output = self.add(self.matmul(input_, self.weight), self.bias)
        return output


class NetFP16(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NetFP16, self).__init__()
        self.weight = Parameter(Tensor(np.ones([out_features, in_features]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([out_features]).astype(np.float32)), name="bias")
        self.matmul = P.MatMul()
        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, input_):
        output = self.cast(
            self.add(self.matmul(self.cast(input_, mstype.float16), self.cast(self.weight, mstype.float16)),
                     self.cast(self.bias, mstype.float16)), mstype.float32)
        return output


def get_axis(x):
    shape = F.shape(x)
    length = F.tuple_len(shape)
    perm = F.make_range(0, length)
    return perm


class MSELoss(nn.Cell):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.square = P.Square()
        self.reduce_mean = P.ReduceMean()
        self.sub = MockSub()

    def construct(self, data, label):
        diff = self.sub(data, label)
        return self.reduce_mean(self.square(diff), get_axis(diff))


class NegCell(nn.Cell):
    def __init__(self):
        super(NegCell, self).__init__()
        self.neg = MockNeg()

    def construct(self, x):
        return self.neg(x)


class Net3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple = (NegCell(), nn.ReLU())

    def construct(self, x):
        for op in self.tuple:
            x = op(x)
        return x


def test_op_forward_infererror():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    with pytest.raises(TypeError):
        net(input_me)


class SequenceNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.seq = nn.SequentialCell([nn.AvgPool2d(3, 1), nn.ReLU(), nn.Flatten()])

    def construct(self, x):
        x = self.seq(x) + bbb
        return x


def test_sequential_resolve_error():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = SequenceNet()
    with pytest.raises(NameError):
        net(input_me)


def test_compile_grad_error():
    inputs = Tensor(np.ones([16, 16]).astype(np.float32))
    label = Tensor(np.zeros([16, 16]).astype(np.float32))
    lr = Tensor(np.ones([1], np.float32) * 0.1)
    net = NetFP16(16, 16)
    loss = MSELoss()
    optimizer = Momentum(net.trainable_params(), learning_rate=lr, momentum=0.9)

    net_with_loss = WithLossCell(net, loss)
    scale_manager = DynamicLossScaleManager()
    update_cell = scale_manager.get_update_cell()
    train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=update_cell)
    train_network.set_train()
    with pytest.raises(TypeError) as e:
        train_network(inputs, label)
        print(e)
