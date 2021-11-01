# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Generate the mindir for bprop"""
import os
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.ops as ops
from mindspore.ops.operations import _inner_ops as inner
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
import mindspore.ops._grad as g
from mindspore.ops.bprop_mindir import serializable_bprop_ops
from mindspore._c_expression import load_mindir


class Net(nn.Cell):
    def __init__(self, op):
        super(Net, self).__init__()
        self.op = op

    def construct(self, *inputs, a=0, b=1):
        c = a + b
        return c, self.op(*inputs)


class TupleInputNet(nn.Cell):
    def __init__(self, op):
        super(TupleInputNet, self).__init__()
        self.op = op

    def construct(self, x):
        return self.op((x,))


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, *inputs):
        gout = self.grad(self.network)(*inputs)
        return gout


def test_remove_mindir_dir():
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = bprop_installed_dir + "/../bprop_mindir"
    os.rename(bprop_mindir_export_dir, bprop_mindir_export_dir + "_bak")
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    relu = Net(P.ReLU())
    grad = GradNet(relu)
    grad.compile(x)
    os.rename(bprop_mindir_export_dir + "_bak", bprop_mindir_export_dir)


def test_load_mindir_dir():
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = bprop_installed_dir + "/../bprop_mindir"
    for op in serializable_bprop_ops:
        file_name = bprop_mindir_export_dir + "/" + op.name + "_bprop.mindir"
        graph = load_mindir(file_name)
        assert not graph is None


def test_relu():
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    relu = Net(P.ReLU())
    grad = GradNet(relu)
    grad.compile(x)


def test_identity():
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.int64))
    identity = Net(P.Identity())
    grad = GradNet(identity)
    grad.compile(x)


def test_range():
    x = Tensor(np.array([1, 2, 3, 2]).astype(np.int64))
    range_net = Net(inner.Range(1.0, 8.0, 2.0))
    grad = GradNet(range_net)
    grad.compile(x)


def test_ones_like():
    x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
    ones_like = Net(P.OnesLike())
    grad = GradNet(ones_like)
    grad.compile(x)


def test_zeros_like():
    x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
    zeros_like = Net(P.ZerosLike())
    grad = GradNet(zeros_like)
    grad.compile(x)


def test_argmax():
    x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
    argmax = Net(P.Argmax())
    grad = GradNet(argmax)
    grad.compile(x)


def test_argmin():
    x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
    argmin = Net(P.Argmin())
    grad = GradNet(argmin)
    grad.compile(x)


def test_broadcast():
    x = Tensor(np.array([1, 2, 5, 2]).astype(np.float32))
    broadcast = TupleInputNet(P.Broadcast(1))
    grad = GradNet(broadcast)
    grad.compile(x)


def test_is_finite():
    x = Tensor(np.ones([2, 4]).astype(np.int32))
    is_finite = Net(P.IsFinite())
    grad = GradNet(is_finite)
    grad.compile(x)


def test_approximate_equal():
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = Tensor(np.array([2, 4, 6]).astype(np.float32))
    approximate_equal = Net(P.ApproximateEqual(2.))
    grad = GradNet(approximate_equal)
    grad.compile(x, y)


def test_logical_not():
    x = Tensor(np.array([True, False, True]).astype(np.bool))
    logical_not = Net(P.LogicalNot())
    grad = GradNet(logical_not)
    grad.compile(x)


def test_sign():
    x = Tensor(np.array([[2.0, 0.0, -1.0]]).astype(np.float32))
    sign = Net(P.Sign())
    grad = GradNet(sign)
    grad.compile(x)


def test_round():
    x = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]).astype(np.float32))
    round_net = Net(P.Round())
    grad = GradNet(round_net)
    grad.compile(x)


def test_lin_space():
    start = Tensor(1, mstype.float32)
    stop = Tensor(10, mstype.float32)
    num = 5
    lin_space = Net(P.LinSpace())
    grad = GradNet(lin_space)
    grad.compile(start, stop, num)


def test_dropout_gen_mask():
    x = (2, 4, 2, 2)
    keep_prob = Tensor(1.0, mstype.float32)
    dropout_gen_mask = Net(P.DropoutGenMask(10, 28))
    grad = GradNet(dropout_gen_mask)
    grad.compile(x, keep_prob)


def test_onehot():
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    depth, on_value, off_value = 3, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)
    one_hot = Net(P.OneHot())
    grad = GradNet(one_hot)
    grad.compile(indices, depth, on_value, off_value)


def test_assign():
    class AssignNet(nn.Cell):
        def __init__(self):
            super(AssignNet, self).__init__()
            self.assign = P.Assign()
            self.variable = Parameter(Tensor([1.0], mstype.float32), name="variable")

        def construct(self, x):
            return self.assign(self.variable, x)

    value = Tensor([2.0], mstype.float32)
    assign = AssignNet()
    grad = GradNet(assign)
    grad.compile(value)


def test_assign_add():
    class AssignAddNet(nn.Cell):
        def __init__(self):
            super(AssignAddNet, self).__init__()
            self.assign_add = P.AssignAdd()
            self.variable = Parameter(initializer(1, [1], mstype.int64), name="global_step")

        def construct(self, x):
            return self.assign_add(self.variable, x)

    value = Tensor(np.ones([1]).astype(np.int64) * 100)
    assign_add = AssignAddNet()
    grad = GradNet(assign_add)
    grad.compile(value)


def test_assign_sub():
    class AssignSubNet(nn.Cell):
        def __init__(self):
            super(AssignSubNet, self).__init__()
            self.assign = P.AssignSub()
            self.variable = Parameter(initializer(1, [1], mstype.int32), name="global_step")

        def construct(self, x):
            return self.assign(self.variable, x)

    value = Tensor(np.ones([1]).astype(np.int32) * 100)
    assign_sub = AssignSubNet()
    grad = GradNet(assign_sub)
    grad.compile(value)


def test_iou():
    anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]).astype(np.float16))
    gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]).astype(np.float16))
    iou = Net(P.IOU())
    grad = GradNet(iou)
    grad.compile(anchor_boxes, gt_boxes)


def test_bn_training_reduce():
    x = Tensor(np.ones([128, 3, 32, 3]).astype(np.float32))
    bn_training_reduce = Net(P.BNTrainingReduce())
    grad = GradNet(bn_training_reduce)
    grad.compile(x)


def test_equal():
    x = Tensor([2.0], mstype.float32)
    y = Tensor([2.0], mstype.float32)
    equal = Net(P.Equal())
    grad = GradNet(equal)
    grad.compile(x, y)


def test_not_equal():
    x = Tensor([2.0], mstype.float32)
    y = Tensor([2.0], mstype.float32)
    not_equal = Net(P.NotEqual())
    grad = GradNet(not_equal)
    grad.compile(x, y)


def test_greater():
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    greater = Net(P.Greater())
    grad = GradNet(greater)
    grad.compile(x, y)


def test_greater_equal():
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    greater_equal = Net(P.GreaterEqual())
    grad = GradNet(greater_equal)
    grad.compile(x, y)


def test_less():
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    less = Net(P.Less())
    grad = GradNet(less)
    grad.compile(x, y)


def test_less_equal():
    x = Tensor(np.array([1, 2, 3]), mstype.int32)
    y = Tensor(np.array([1, 1, 4]), mstype.int32)
    less_equal = Net(P.LessEqual())
    grad = GradNet(less_equal)
    grad.compile(x, y)


def test_logical_and():
    x = Tensor(np.array([True, False, True]), mstype.bool_)
    y = Tensor(np.array([True, True, False]), mstype.bool_)
    logical_and = Net(P.LogicalAnd())
    grad = GradNet(logical_and)
    grad.compile(x, y)


def test_logical_or():
    x = Tensor(np.array([True, False, True]), mstype.bool_)
    y = Tensor(np.array([True, True, False]), mstype.bool_)
    logical_or = Net(P.LogicalOr())
    grad = GradNet(logical_or)
    grad.compile(x, y)


def test_reduce_all():
    x = Tensor(np.array([[True, False], [True, True]]))
    reduce_all = Net(P.ReduceAll(keep_dims=True))
    grad = GradNet(reduce_all)
    grad.compile(x)


def test_reduce_any():
    x = Tensor(np.array([[True, False], [True, True]]))
    reduce_all = Net(P.ReduceAny(keep_dims=True))
    grad = GradNet(reduce_all)
    grad.compile(x)


def test_dropout_do_mask():
    input_x = Tensor(np.ones([2, 2, 3]), mstype.float32)
    keep_prob = Tensor(0.5, mstype.float32)
    mask = Tensor(np.ones([2]), mstype.uint8)
    dropout_do_mask = Net(P.DropoutDoMask())
    grad = GradNet(dropout_do_mask)
    grad.compile(input_x, mask, keep_prob)
