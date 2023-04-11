# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
import mindspore.ops._grad_experimental as g
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops._grad_experimental.grad_base import bprop_getters, bprops
from mindspore._c_expression import _check_bprop_mindir
from mindspore import mutable
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self, op):
        super(Net, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


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


def grad_compile_(arg, operation):
    net = Net(operation)
    grad = GradNet(net)
    if isinstance(arg, (tuple, list)):
        grad.compile(*arg)
    else:
        grad.compile(arg)



def load_bprop_mindir(bprop_mindir_install_dir, bprop_map):
    for op in bprop_map.keys():
        if not isinstance(op, str):
            continue
        file_name = os.path.join(bprop_mindir_install_dir, op + "_bprop.mindir")
        if os.path.isfile(file_name):
            assert _check_bprop_mindir(op)


def test_load_mindir():
    """
    Feature: Bprop pre-compilation.
    Description: Load all the mindir files of serializable bprop.
    Expectation: All are loaded successfully.
    """
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = os.path.join(bprop_installed_dir, "..", "bprop_mindir")
    load_bprop_mindir(bprop_mindir_export_dir, bprop_getters)
    load_bprop_mindir(bprop_mindir_export_dir, bprops)


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
    start = Tensor(0.0, mstype.float32)
    limit = Tensor(10.0, mstype.float32)
    delta = Tensor(1.0, mstype.float32)
    range_net = Net(P.Range())
    grad = GradNet(range_net)
    grad.compile(start, limit, delta)


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
            self.assign(self.variable, x)
            return self.variable

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
            self.assign_add(self.variable, x)
            return self.variable

    value = Tensor(np.ones([1]).astype(np.int64) * 100)
    assign_add = AssignAddNet()
    grad = GradNet(assign_add)
    grad.compile(value)


def test_assign_sub():
    class AssignSubNet(nn.Cell):
        def __init__(self):
            super(AssignSubNet, self).__init__()
            self.assign_sub = P.AssignSub()
            self.variable = Parameter(initializer(1, [1], mstype.int32), name="global_step")

        def construct(self, x):
            self.assign_sub(self.variable, x)
            return self.variable

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


def test_select():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the select op.
    Expectation: Load the bprop mindir successfully.
    """
    input_cond = Tensor([True, False])
    x = Tensor(np.array([1, 2]), mstype.int32)
    y = Tensor(np.array([1, 1]), mstype.int32)
    select = Net(P.Select())
    grad = GradNet(select)
    grad.compile(input_cond, x, y)


def test_scatter_max():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the scatter_max op.
    Expectation: Load the bprop mindir successfully.
    """

    class ScatterMaxNet(nn.Cell):
        def __init__(self):
            super(ScatterMaxNet, self).__init__()
            self.scatter_max = P.ScatterMax()
            self.input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32),
                                     name="input_x")

        def construct(self, indices, updates):
            return self.scatter_max(self.input_x, indices, updates)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]) * 88, mstype.float32)
    scatter_max = ScatterMaxNet()
    grad = GradNet(scatter_max)
    grad.compile(indices, updates)


def test_scatter_min():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the scatter_min op.
    Expectation: Load the bprop mindir successfully.
    """

    class ScatterMinNet(nn.Cell):
        def __init__(self):
            super(ScatterMinNet, self).__init__()
            self.scatter_min = P.ScatterMin()
            self.input_x = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32),
                                     name="input_x")

        def construct(self, indices, updates):
            return self.scatter_min(self.input_x, indices, updates)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]) * 88, mstype.float32)
    scatter_min = ScatterMinNet()
    grad = GradNet(scatter_min)
    grad.compile(indices, updates)


def test_relu_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the relu_grad op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[[-1, 1, 10],
                           [1, -1, 1],
                           [10, 1, -1]]]]).astype(np.float32))
    relu = Net(P.ReLU())
    grad1 = GradNet(relu)
    grad2 = GradNet(grad1)
    grad2.compile(x)


def test_tuple_getitem():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tuple_getitem op.
    Expectation: Load the bprop mindir successfully.
    """

    class TupleGetitemNet(nn.Cell):
        def __init__(self):
            super(TupleGetitemNet, self).__init__()
            self.maxpool_arg = P.MaxPoolWithArgmax(pad_mode="VALID", kernel_size=2, strides=1)

        def construct(self, x):
            output = self.maxpool_arg(x)
            return output[0]

    x = Tensor(np.arange(1 * 3 * 3 * 4).reshape((1, 3, 3, 4)), mstype.float32)
    tuple_getitem = TupleGetitemNet()
    grad = GradNet(tuple_getitem)
    grad.compile(x)


def test_depend():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the depend op.
    Expectation: Load the bprop mindir successfully.
    """

    class DependNet(nn.Cell):
        def __init__(self):
            super(DependNet, self).__init__()
            self.softmax = P.Softmax()
            self.depend = ops.Depend()

        def construct(self, x, y):
            mul = x * y
            y = self.depend(y, mul)
            output = self.softmax(y)
            return output

    x = Tensor(np.ones([4, 5]), mstype.float32)
    y = Tensor(np.ones([4, 5]), mstype.float32)
    depend = DependNet()
    grad = GradNet(depend)
    grad.compile(x, y)


def test_switch():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the switch op.
    Expectation: Load the bprop mindir successfully.
    """

    class SwitchNet(nn.Cell):
        def construct(self, x, y):
            if x > y:
                return x
            return y

    x = Tensor(np.array([3]), mstype.float32)
    y = Tensor(np.array([2]), mstype.float32)
    switch_net = SwitchNet()
    grad = GradNet(switch_net)
    grad.compile(x, y)


def test_update_state():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the update_state op.
    Expectation: Load the bprop mindir successfully.
    """

    class UpdateStateNet(nn.Cell):
        def __init__(self):
            super(UpdateStateNet, self).__init__()
            self.assign_add = P.AssignAdd()
            self.variable = Parameter(initializer(1, [1], mstype.int64), name="global_step")

        def construct(self, x):
            self.assign_add(self.variable, x)
            return self.variable

    value = Tensor(np.ones([1]).astype(np.int64) * 100)
    update_state = UpdateStateNet()
    grad = GradNet(update_state)
    grad.compile(value)


def test_load():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the load op.
    Expectation: Load the bprop mindir successfully.
    """

    class LoadNet(nn.Cell):
        def __init__(self):
            super(LoadNet, self).__init__()
            self.add = P.Add()
            self.variable = Parameter(initializer(1, [1], mstype.int64), name="global_step")

        def construct(self, x):
            return self.add(self.variable, x)

    value = Tensor(np.ones([1]).astype(np.int64) * 100)
    load = LoadNet()
    grad = GradNet(load)
    grad.compile(value)


def test_floor_div():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the floor_div op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([2, 4, -1]), mstype.int32)
    y = Tensor(np.array([3, 3, 3]), mstype.int32)
    floor_div = Net(P.FloorDiv())
    grad = GradNet(floor_div)
    grad.compile(x, y)


def test_truncate_div():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the truncate_div op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([2, 4, -1]), mstype.int32)
    y = Tensor(np.array([3, 3, 3]), mstype.int32)
    truncate_div = Net(P.TruncateDiv())
    grad = GradNet(truncate_div)
    grad.compile(x, y)


def test_minimum():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the minimum op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1.0, 5.0, 3.0]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 6.0]), mstype.float32)
    minimum = Net(P.Minimum())
    grad = GradNet(minimum)
    grad.compile(x, y)


def test_maximum():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the maximum op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1.0, 5.0, 3.0]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 6.0]), mstype.float32)
    maximum = Net(P.Maximum())
    grad = GradNet(maximum)
    grad.compile(x, y)


def test_is_nan():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the is_nan op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mstype.float32)
    is_nan = Net(P.IsNan())
    grad = GradNet(is_nan)
    grad.compile(x)


def test_is_inf():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the is_inf op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mstype.float32)
    is_inf = Net(P.IsInf())
    grad = GradNet(is_inf)
    grad.compile(x)


def test_relu_v2():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the relu_v2 op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]]]), mstype.float32)
    relu_v2 = Net(P.ReLUV2())
    grad = GradNet(relu_v2)
    grad.compile(x)


def test_cast():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the cast op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1.0]), mstype.float32)
    cast = Net(P.Cast())
    grad = GradNet(cast)
    grad.compile(x, mstype.int32)


def test_split():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the split op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]), mstype.int32)
    split = Net(P.Split(1, 2))
    grad = GradNet(split)
    grad.compile(x)


def test_reshape():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the reshape op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    reshape = Net(P.Reshape())
    grad = GradNet(reshape)
    grad.compile(x, (3, 2))


def test_expand_dims():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the expand_dims op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[2, 2], [2, 2]]), mstype.float32)
    expand_dims = Net(P.ExpandDims())
    grad = GradNet(expand_dims)
    grad.compile(x, 0)


def test_squeeze():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the squeeze op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.ones(shape=[3, 2, 1]), mstype.float32)
    squeeze = Net(P.Squeeze(2))
    grad = GradNet(squeeze)
    grad.compile(x)


def test_flatten():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the flatten op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.ones(shape=[1, 2, 3, 4]), mstype.float32)
    flatten = Net(P.Flatten())
    grad = GradNet(flatten)
    grad.compile(x)


def test_tile():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tile op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]), mstype.float32)
    tile = Net(P.Tile())
    grad = GradNet(tile)
    grad.compile(x, (2, 3))


def test_embedding_lookup():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the embedding_lookup op.
    Expectation: Load the bprop mindir successfully.
    """
    input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mstype.float32)
    input_indices = Tensor(np.array([[5, 2], [8, 5]]), mstype.int32)
    offset = 4
    embedding_lookup = Net(P.EmbeddingLookup())
    grad = GradNet(embedding_lookup)
    grad.compile(input_params, input_indices, offset)


def test_padding():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the padding op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[8], [10]]), mstype.float32)
    padding = Net(P.Padding(4))
    grad = GradNet(padding)
    grad.compile(x)


def test_transpose():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the transpose op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mstype.float32)
    perm = (0, 2, 1)
    transpose = Net(P.Transpose())
    grad = GradNet(transpose)
    grad.compile(x, perm)


def test_concat():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the concat op.
    Expectation: Load the bprop mindir successfully.
    """
    x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
    concat = Net(P.Concat())
    grad = GradNet(concat)
    grad.compile(mutable((x1, x2)))


def test_slice():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the slice op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]]]).astype(np.int32))
    slice_net = Net(P.Slice())
    grad = GradNet(slice_net)
    grad.compile(x, (1, 0, 0), (1, 1, 3))


def test_gather():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the gather op.
    Expectation: Load the bprop mindir successfully.
    """
    input_params = Tensor(np.array([1, 2, 3, 4, 5, 6, 7]), mstype.float32)
    input_indices = Tensor(np.array([0, 2, 4, 2, 6]), mstype.int32)
    gather = Net(P.Gather())
    grad = GradNet(gather)
    grad.compile(input_params, input_indices, 0)


def test_gather_d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the gather_d op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)
    index = Tensor(np.array([[0, 0], [1, 0]]), mstype.int32)
    gather_d = Net(P.GatherD())
    grad = GradNet(gather_d)
    grad.compile(x, 1, index)


def test_sort():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the sort op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[8, 2, 1], [5, 9, 3], [4, 6, 7]]), mstype.float16)
    sort = Net(P.Sort())
    grad = GradNet(sort)
    grad.compile(x)


def test_reverse_v2():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the reverse_v2 op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mstype.int32)
    reverse_v2 = Net(P.ReverseV2(axis=[1]))
    grad = GradNet(reverse_v2)
    grad.compile(x)


def test_unstack():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the unstack op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))
    unstack = Net(P.Unstack())
    grad = GradNet(unstack)
    grad.compile(x)


def test_strided_slice():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the strided_slice op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]]], mstype.float32)
    strided_slice = Net(P.StridedSlice())
    grad = GradNet(strided_slice)
    grad.compile(x, (1, 0, 2), (3, 1, 3), (1, 1, 1))


def test_strided_slice_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the strided_slice_grad op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]]], mstype.float32)
    strided_slice = Net(P.StridedSlice())
    grad = GradNet(strided_slice)
    second_grad = GradNet(grad)
    second_grad.compile(x, (1, 0, 2), (3, 1, 3), (1, 1, 1))


def test_sparse_gather_v2():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the sparse_gather_v2 op.
    Expectation: Load the bprop mindir successfully.
    """
    input_params = Tensor(np.array([[1, 2, 7, 42], [3, 4, 54, 22], [2, 2, 55, 3]]), mstype.float32)
    input_indices = Tensor(np.array([1, 2]), mstype.int32)
    axis = 1
    sparse_gather_v2 = Net(P.SparseGatherV2())
    grad = GradNet(sparse_gather_v2)
    grad.compile(input_params, input_indices, axis)


def test_resize_nearest_neighbor():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the resize_nearest_neighbor op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), mstype.float32)
    resize_nearest_neighbor = Net(P.ResizeNearestNeighbor((2, 2)))
    grad = GradNet(resize_nearest_neighbor)
    grad.compile(x)


def test_gather_nd():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the gather_nd op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    gather_nd = Net(P.GatherNd())
    grad = GradNet(gather_nd)
    grad.compile(x, indices)


def test_scatter_nd():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the scatter_nd op.
    Expectation: Load the bprop mindir successfully.
    """
    indices = Tensor(np.array([[0], [2]]), mstype.int32)
    updates = Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2],
                                [3, 3, 3, 3], [4, 4, 4, 4]],
                               [[1, 1, 1, 1], [2, 2, 2, 2],
                                [3, 3, 3, 3], [4, 4, 4, 4]]]), mstype.float32)
    shape = (4, 4, 4)
    scatter_nd = Net(P.ScatterNd())
    grad = GradNet(scatter_nd)
    grad.compile(indices, updates, shape)


def test_scatter_nd_update():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the scatter_nd_update op.
    Expectation: Load the bprop mindir successfully.
    """
    np_x = np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
    input_x = Parameter(Tensor(np_x, mstype.float32), name="x")
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    scatter_nd_update = Net(P.ScatterNdUpdate())
    grad = GradNet(scatter_nd_update)
    grad.compile(input_x, indices, updates)


def test_scatter_non_aliasing_add():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the scatter_non_aliasing_add op.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32), name="x")
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    scatter_non_aliasing_add = Net(P.ScatterNonAliasingAdd())
    grad = GradNet(scatter_non_aliasing_add)
    grad.compile(input_x, indices, updates)


def test_tensor_scatter_update():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tensor_scatter_update op.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    update = Tensor(np.array([1.0, 2.2]), mstype.float32)
    tensor_scatter_update = Net(P.TensorScatterUpdate())
    grad = GradNet(tensor_scatter_update)
    grad.compile(input_x, indices, update)


def test_tensor_scatter_add():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tensor_scatter_add op.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [0, 0]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    tensor_scatter_add = Net(P.TensorScatterAdd())
    grad = GradNet(tensor_scatter_add)
    grad.compile(input_x, indices, updates)


def test_space_to_depth():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the space_to_depth op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.random.rand(1, 3, 2, 2), mstype.float32)
    block_size = 2
    space_to_depth = Net(P.SpaceToDepth(block_size))
    grad = GradNet(space_to_depth)
    grad.compile(x)


def test_depth_to_space():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the depth_to_space op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.random.rand(1, 12, 1, 1), mstype.float32)
    block_size = 2
    depth_to_space = Net(P.DepthToSpace(block_size))
    grad = GradNet(depth_to_space)
    grad.compile(x)


def test_diag_part():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the diag_part op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor([[1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 4]])
    diag_part = Net(P.DiagPart())
    grad = GradNet(diag_part)
    grad.compile(x)


def test_space_to_batch_nd():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the space_to_batch_nd op.
    Expectation: Load the bprop mindir successfully.
    """
    block_size = 2
    paddings = [[0, 0], [0, 0]]
    x = Tensor(np.array([[[[1, 2], [3, 4]]]]), mstype.float32)
    space_to_batch_nd = Net(P.SpaceToBatchND(block_size, paddings))
    grad = GradNet(space_to_batch_nd)
    grad.compile(x)


def test_batch_to_space_nd():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the batch_to_space_nd op.
    Expectation: Load the bprop mindir successfully.
    """
    block_size = 2
    crops = [[0, 0], [0, 0]]
    x = Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]), mstype.float32)
    batch_to_space_nd = Net(P.BatchToSpaceND(block_size, crops))
    grad = GradNet(batch_to_space_nd)
    grad.compile(x)


def test_broadcast_to():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the broadcast_to op.
    Expectation: Load the bprop mindir successfully.
    """
    shape = (2, 3)
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    broadcast_to = Net(P.BroadcastTo(shape))
    grad = GradNet(broadcast_to)
    grad.compile(x)


def test_reverse_sequence():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the reverse_sequence op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mstype.float32)
    seq_lengths = Tensor(np.array([1, 2, 3]))
    reverse_sequence = Net(P.ReverseSequence(seq_dim=1))
    grad = GradNet(reverse_sequence)
    grad.compile(x, seq_lengths)


def test_trans_shape():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the trans_shape op.
    Expectation: Load the bprop mindir successfully.
    """
    shape = (3, 1)
    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    trans_shape = Net(P.TransShape())
    grad = GradNet(trans_shape)
    grad.compile(x, shape)


def test_trans_unique():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the unique op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1, 2, 5, 2]), mstype.int32)
    unique = Net(P.Unique())
    grad = GradNet(unique)
    grad.compile(x)


def test_masked_select():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the masked_select op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1, 2, 3, 4]), mstype.int32)
    mask = Tensor(np.array([1, 0, 1, 0]), mstype.bool_)
    masked_select = Net(P.MaskedSelect())
    grad = GradNet(masked_select)
    grad.compile(x, mask)


def test_non_zero():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the non_zero op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[1, 0], [-5, 0]]]), mstype.int32)
    grad = GradNet(ops.nonzero)
    grad.compile(x)


def test_bias_add_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the bias_add op.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.arange(6).reshape((2, 3)), mstype.float32)
    bias = Tensor(np.random.random(3).reshape((3,)), mstype.float32)
    bias_add = Net(P.BiasAdd())
    grad = GradNet(bias_add)
    grad.compile(input_x, bias)


def test_bias_add_grad_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the bias_add_grad op.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.arange(6).reshape((2, 3)), mstype.float32)
    bias = Tensor(np.random.random(3).reshape((3,)), mstype.float32)
    bias_add = Net(P.BiasAdd())
    grad = GradNet(bias_add)
    second_grad = GradNet(grad)
    second_grad.compile(input_x, bias)


def test_conv2d_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the conv2d op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
    weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
    conv2d = Net(P.Conv2D(out_channel=32, kernel_size=3))
    grad = GradNet(conv2d)
    grad.compile(x, weight)


def test_conv3d_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the conv3d op.
    Expectation: Load the bprop mindir successfully.
    """

    x = Tensor(np.ones([16, 3, 10, 32, 32]), mindspore.float16)
    weight = Tensor(np.ones([32, 3, 4, 3, 3]), mindspore.float16)
    conv3d = Net(P.Conv3D(out_channel=32, kernel_size=(4, 3, 3)))
    grad = GradNet(conv3d)
    grad.compile(x, weight)


def test_conv3d_transpose_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the conv3d transpose op.
    Expectation: Load the bprop mindir successfully.
    """
    dout = Tensor(np.ones([32, 16, 10, 32, 32]), mindspore.float16)
    weight = Tensor(np.ones([16, 3, 4, 6, 2]), mindspore.float16)
    conv3d_transpose = Net(P.Conv3DTranspose(in_channel=16, out_channel=3, kernel_size=(4, 6, 2)))
    grad = GradNet(conv3d_transpose)
    grad.compile(dout, weight)


def test_depth_wise_conv2d_native():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the depth_wise_conv2d_native op.
    Expectation: Load the bprop mindir successfully.
    """
    w = Tensor(np.ones([1, 3, 3, 3], dtype=np.float32))
    x = Tensor(np.ones([1, 3, 16, 16], np.float32))
    net = Net(P.DepthwiseConv2dNative(channel_multiplier=3, kernel_size=(3, 3)))
    grad = GradNet(net)
    grad.compile(x, w)


def test_max_pool3d_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the max_pool3d_grad.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.arange(1 * 2 * 2 * 2 * 3).reshape((1, 2, 2, 2, 3)), mindspore.float32)
    net = Net(ops.MaxPool3D(kernel_size=2, strides=1, pad_mode="valid"))
    grad = GradNet(net)
    grad.compile(x)

    second_grad = GradNet(grad)
    second_grad.compile(x)

    third_grad = GradNet(second_grad)
    third_grad.compile(x)


def test_adaptive_max_pool2d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the adaptive_max_pool2d.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]), mindspore.float32)
    net = Net(ops.AdaptiveMaxPool2D((None, 2)))
    grad = GradNet(net)
    grad.compile(input_x)


def test_avg_pool():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the avg_pool.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float32)
    net = Net(ops.AvgPool(pad_mode="VALID", kernel_size=2, strides=1))
    grad = GradNet(net)
    grad.compile(x)


def test_adaptive_avg_pool2d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the avg_pool.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float32)
    net = Net(ops.AvgPool(pad_mode="VALID", kernel_size=2, strides=1))
    grad = GradNet(net)
    grad.compile(x)


def test_avg_pool_3d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the avg_pool_3d.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float32)
    avgpool_op = ops.AvgPool(pad_mode="VALID", kernel_size=2, strides=1)
    grad_compile_(x, avgpool_op)


def test_mish():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the Mish.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    mish = ops.Mish()
    grad_compile_(x, mish)


def test_selu():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the SeLU.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    selu = ops.SeLU()
    grad_compile_(input_x, selu)


def test_mul_no_none():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the mul_no_none.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[-1.0, 6.0, np.inf], [np.nan, -7.0, 4.0]]), mindspore.float32)
    y = Tensor(np.array([[-1.0, 4.0, 0], [0, -3.0, 1.0]]), mindspore.float32)
    mul_no_nan = ops.MulNoNan()
    grad_compile_([x, y], mul_no_nan)


def test_relu6():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the relu6.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    relu6 = ops.ReLU6()
    grad_compile_(input_x, relu6)


def test_hswish():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the HSwish.
    Expectation: Load the bprop mindir successfully.
    """
    hswish = ops.HSwish()
    input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
    grad_compile_(input_x, hswish)


def test_hsigmoid():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the HSigmoid.
    Expectation: Load the bprop mindir successfully.
    """
    hsigmoid = ops.HSigmoid()
    input_x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
    grad_compile_(input_x, hsigmoid)


def test_elu():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the Elu.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    elu = ops.Elu()
    grad_compile_(input_x, elu)


def test_sigmoid():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the sigmoid.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    sigmoid = ops.Sigmoid()
    grad_compile_(input_x, sigmoid)


def test_sigmoid_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the sigmoid_grad.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    sigmoid_grad = G.SigmoidGrad()
    grad_compile_([input_x, input_x], sigmoid_grad)



def test_log_softmax():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the log_softmax.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    log_softmax = ops.LogSoftmax()
    grad_compile_(logits, log_softmax)


def test_softplus():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the softplus.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    softplus = ops.Softplus()
    grad_compile_(input_x, softplus)


def test_softsign():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the softsign.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([0, -1, 2, 30, -30]), mindspore.float32)
    softsign = ops.Softsign()
    grad_compile_(input_x, softsign)


def test_tanh():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tanh.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    tanh = ops.Tanh()
    grad_compile_(input_x, tanh)


def test_tanh_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the tanh_grad.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    tanh_grad = G.TanhGrad()
    grad_compile_([input_x, input_x], tanh_grad)


def test_gelu():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the gelu.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    gelu = ops.GeLU()
    grad_compile_(x, gelu)



def test_fast_gelu():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the fast_gelu.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    fast_gelu = ops.FastGeLU()
    grad_compile_(x, fast_gelu)



def test_instance_norm():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the instance_norm.
    Expectation: Load the bprop mindir successfully.
    """
    class InstanceNormNet(nn.Cell):
        def __init__(self):
            super(InstanceNormNet, self).__init__()
            self.instance_norm = ops.operations.nn_ops.InstanceNorm()
            self.gamma = Parameter(Tensor(np.ones([64]), mindspore.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([64]), mindspore.float32), name="beta")
            self.mean = Parameter(Tensor(np.ones([64]), mindspore.float32), name="mean")
            self.variance = Parameter(Tensor(np.ones([64]), mindspore.float32), name="variance")
        def construct(self, input_x):
            out = self.instance_norm(input_x, self.gamma, self.beta, self.mean, self.variance)
            return out
    input_x = Tensor(np.ones([128, 64, 32, 64]), mindspore.float32)
    net = InstanceNormNet()
    grad_net = GradNet(net)
    grad_net.compile(input_x)


def test_batch_norm():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the batch_norm.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.ones([2, 2]), mindspore.float32)
    scale = Tensor(np.ones([2]), mindspore.float32)
    bias = Tensor(np.ones([2]), mindspore.float32)
    mean = Tensor(np.ones([2]), mindspore.float32)
    variance = Tensor(np.ones([2]), mindspore.float32)
    batch_norm = ops.BatchNorm()
    grad_compile_([input_x, scale, bias, mean, variance], batch_norm)


def test_batch_norm_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the batch_norm_grad.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.ones([2, 2]), mindspore.float32)
    scale = Tensor(np.ones([2]), mindspore.float32)
    bias = Tensor(np.ones([2]), mindspore.float32)
    mean = Tensor(np.ones([2]), mindspore.float32)
    variance = Tensor(np.ones([2]), mindspore.float32)
    batch_norm = G.BatchNormGrad()
    grad_compile_([input_x, input_x, scale, bias, mean, variance], batch_norm)


def test_layer_norm():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the layer_norm.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    layer_norm = ops.LayerNorm()
    grad_compile_([input_x, gamma, beta], layer_norm)


def test_layer_norm_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the layer_norm.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
    gamma = Tensor(np.ones([3]), mindspore.float32)
    beta = Tensor(np.ones([3]), mindspore.float32)
    variance = Tensor(np.array([6.6, 6.6]), mindspore.float32)
    layer_norm_grad = G.LayerNormGrad()
    grad_compile_([input_x, input_x, gamma, beta, variance], layer_norm_grad)


def test_l2_normalize():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the l2_normalize.
    Expectation: Load the bprop mindir successfully.
    """
    l2_normalize = ops.L2Normalize()
    x = Tensor(np.random.randint(-256, 256, (2, 3, 4)), mindspore.float32)
    grad_compile_(x, l2_normalize)


def test_softmax_cross():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the softmax_cross.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor([[2, 4, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
    labels = Tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], mindspore.float32)
    softmax_cross = ops.SoftmaxCrossEntropyWithLogits()
    grad_compile_([logits, labels], softmax_cross)


def test_nll_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the nll_loss.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor(np.array([[0.5488135, 0.71518934],
                              [0.60276335, 0.5448832],
                              [0.4236548, 0.6458941]]).astype(np.float32))
    labels = Tensor(np.array([0, 0, 0]).astype(np.int32))
    weight = Tensor(np.array([0.3834415, 0.79172504]).astype(np.float32))
    nll_loss = ops.NLLLoss(reduction="mean")
    grad_compile_([logits, labels, weight], nll_loss)


def test_sparse_softmax_cross():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the sparse_softmax_cross.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
    labels = Tensor([0, 1], mindspore.int32)
    sparse_softmax_cross = ops.SparseSoftmaxCrossEntropyWithLogits()
    grad_compile_([logits, labels], sparse_softmax_cross)


def test_resize_bilinear():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the resize_bilinear.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mindspore.float32)
    resize_bilinear = ops.ResizeBilinear((5, 5))
    grad_compile_(x, resize_bilinear)


def test_one_hot():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the onehot.
    Expectation: Load the bprop mindir successfully.
    """
    indices = Tensor(np.array([0, 1, 2]), mindspore.int32)
    depth, on_value, off_value = 3, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
    onehot = ops.OneHot()
    grad_compile_([indices, depth, on_value, off_value], onehot)


def test_topk():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the topk.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor([1, 2, 3, 4, 5], mindspore.float16)
    k = 3
    grad_compile_((input_x, k), ops.TopK(sorted=True))


def test_smooth_l1_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the smooth_l1_loss.
    Expectation: Load the bprop mindir successfully.
    """
    loss = ops.SmoothL1Loss()
    logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
    labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
    grad_compile_((logits, labels), loss)


def test_l2_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the l2_loss.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([1, 2, 3]), mindspore.float16)
    l2_loss = ops.L2Loss()
    grad_compile_(input_x, l2_loss)


def test_rnn_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the rnn_loss.
    Expectation: Load the bprop mindir successfully.
    """
    b, t, u, v = 1, 2, 3, 5
    acts = np.random.random((b, t, u, v)).astype(np.float32)
    labels = np.array([[1, 2]]).astype(np.int32)
    input_length = np.array([t] * b).astype(np.int32)
    label_length = np.array([len(l) for l in labels]).astype(np.int32)
    rnnt_loss = ops.RNNTLoss(blank_label=0)
    grad_compile_((Tensor(acts), Tensor(labels), Tensor(input_length), Tensor(label_length)), rnnt_loss)


def test_prelu():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the prelu.
    Expectation: Load the bprop mindir successfully.
    """
    prelu = ops.PReLU()
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)), mindspore.float32)
    weight = Tensor(np.array([0.1, 0.6, -0.3]), mindspore.float32)
    grad_compile_((x, weight), prelu)


def test_dynamic_rnn():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the dynamic_rnn.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.random.rand(2, 16, 64).astype(np.float16))
    w = Tensor(np.random.rand(96, 128).astype(np.float16))
    b = Tensor(np.random.rand(128).astype(np.float16))
    init_h = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
    init_c = Tensor(np.random.rand(1, 16, 32).astype(np.float16))
    dynamic_rnn = ops.DynamicRNN()
    grad_compile_((x, w, b, None, init_h, init_c), dynamic_rnn)


def test_dynamic_gru_v2():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the dynamic_gru_v2.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.random.rand(2, 8, 64).astype(np.float16))
    weight_i = Tensor(np.random.rand(64, 48).astype(np.float16))
    weight_h = Tensor(np.random.rand(16, 48).astype(np.float16))
    bias_i = Tensor(np.random.rand(48).astype(np.float16))
    bias_h = Tensor(np.random.rand(48).astype(np.float16))
    init_h = Tensor(np.random.rand(8, 16).astype(np.float16))
    dynamic_gru_v2 = ops.DynamicGRUV2()
    grad_compile_((x, weight_i, weight_h, bias_i, bias_h, None, init_h), dynamic_gru_v2)


def test_sigmoid_cross_entropy_with_logits():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the SigmoidCrossEntropyWithLogits.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
    labels = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
    sigmoid = ops.SigmoidCrossEntropyWithLogits()
    grad_compile_((logits, labels), sigmoid)


def test_pad_op():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the pad.
    Expectation: Load the bprop mindir successfully.
    """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    pad_op = ops.Pad(((1, 2), (2, 1)))
    grad_compile_(input_x, pad_op)


def test_mirror_pad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the mirror_pad.
    Expectation: Load the bprop mindir successfully.
    """
    mode = "REFLECT"
    pad = ops.MirrorPad(mode=mode)
    paddings = Tensor([[1, 1], [2, 2]])
    input_x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    grad_compile_((input_x, paddings), pad)


def test_roi_align():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the roi_align.
    Expectation: Load the bprop mindir successfully.
    """
    features = Tensor(np.array([[[[1., 2.], [3., 4.]]]]), mindspore.float32)
    rois = Tensor(np.array([[0, 0.2, 0.3, 0.2, 0.3]]), mindspore.float32)
    roi_align = ops.ROIAlign(2, 2, 0.5, 2)
    grad_compile_((features, rois), roi_align)


def test_conv2d_transpose_input():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the conv2d_transpose_input.
    Expectation: Load the bprop mindir successfully.
    """
    dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
    weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
    x = Tensor(np.ones([10, 32, 32, 32]))
    conv2d_transpose_input = ops.Conv2DTranspose(out_channel=32, kernel_size=3)
    grad_compile_((dout, weight, ops.shape(x)), conv2d_transpose_input)


def test_binary_cross_entropy():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the binary_cross_entropy.
    Expectation: Load the bprop mindir successfully.
    """
    class TestNet(nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            self.binary_cross_entropy = ops.BinaryCrossEntropy()
        def construct(self, logits, labels, weight):
            result = self.binary_cross_entropy(logits, labels, weight)
            return result
    net = TestNet()
    logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
    labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
    weight = Tensor(np.array([1, 2, 2]), mindspore.float32)
    grad = GradNet(net)
    grad.compile(logits, labels, weight)


def test_bce_with_logits_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the BCE_with_logits_loss.
    Expectation: Load the bprop mindir successfully.
    """
    logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
    label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
    weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
    pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
    loss = ops.BCEWithLogitsLoss()
    grad_compile_((logits, label, weight, pos_weight), loss)


def test_kldiv_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the kldiv_loss.
    Expectation: Load the bprop mindir successfully.
    """
    kldiv_loss = ops.KLDivLoss(reduction='sum')
    logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
    labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
    grad_compile_((logits, labels), kldiv_loss)


def test_dropout():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the dropout.
    Expectation: Load the bprop mindir successfully.
    """
    dropout = ops.Dropout3D(keep_prob=0.5)
    x = Tensor(np.ones([1, 2, 3, 4, 5]), mindspore.float32)
    grad_compile_(x, dropout)


def test_dropout_grad():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the dropout_grad.
    Expectation: Load the bprop mindir successfully.
    """
    dropout_grad = G.DropoutGrad(keep_prob=0.5)
    x = Tensor((20, 16, 50, 50), mindspore.float32)
    grad_compile_((x, x), dropout_grad)


def test_dropout_2d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the dropout2d.
    Expectation: Load the bprop mindir successfully.
    """
    from mindspore.ops.operations.nn_ops import Dropout2D
    dropout = Dropout2D(keep_prob=0.5)
    x = Tensor(np.ones([2, 1, 2, 3]), mindspore.float32)
    grad_compile_(x, dropout)


def test_ctc_loss():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the ctc_loss.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[0.3, 0.6, 0.6],
                          [0.4, 0.3, 0.9]],
                         [[0.9, 0.4, 0.2],
                          [0.9, 0.9, 0.1]]]).astype(np.float32))
    labels_indices = Tensor(np.array([[0, 0], [1, 0]]), mindspore.int64)
    labels_values = Tensor(np.array([2, 2]), mindspore.int32)
    sequence_length = Tensor(np.array([2, 2]), mindspore.int32)
    ctc_loss = ops.CTCLoss()
    grad_compile_((x, labels_indices, labels_values, sequence_length), ctc_loss)


def test_deformed_convolution():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the deformed_convolution.
    Expectation: Load the bprop mindir successfully.
    """
    kh, kw = 3, 3
    kernel_size = (kh, kw)
    strides = (1, 1, 1, 1)
    padding = (0, 0, 0, 0)
    dilations = (1, 1, 1, 1)
    deformable_groups = 1
    modulated = True
    deformable_offsets = ops.DeformableOffsets(strides, padding, kernel_size, dilations,
                                               "NCHW", deformable_groups, modulated)
    x = Tensor(np.ones((4, 3, 10, 10)), mstype.float32)
    offerts = Tensor(np.ones((5, 3 * kh * kw, 8, 8)), mstype.float32)
    grad_compile_([x, offerts], deformable_offsets)


def test_lrn():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the lrn.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([[[[0.1], [0.2]],
                          [[0.3], [0.4]]]]), mindspore.float32)
    lrn = ops.LRN()
    grad_compile_(x, lrn)


def test_conv_filter():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the Conv2D_backprop_filter.
    Expectation: Load the bprop mindir successfully.
    """
    class TestNet(nn.Cell):
        def __init__(self):
            super(TestNet, self).__init__()
            out_channel = 4
            kernel_size = 1
            self.conv_filter = G.Conv2DBackpropFilter(out_channel,
                                                      kernel_size,
                                                      pad_mode="valid",
                                                      pad=0,
                                                      mode=1,
                                                      stride=1,
                                                      dilation=1,
                                                      group=1)
            self.w = Parameter(initializer(Tensor(np.array([[[
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]]]]).astype(np.float32)), [1, 1, 3, 3]), name='w')
            self.x = Parameter(initializer(Tensor(np.array([[[
                [3, 0, 1, 2, 7, 4],
                [1, 5, 8, 9, 3, 1],
                [2, 7, 2, 5, 1, 3],
                [0, 1, 3, 1, 7, 8],
                [4, 2, 1, 6, 2, 8],
                [2, 4, 5, 2, 3, 9]]]]).astype(np.float32)), [1, 1, 6, 6]), name='x')
            self.out = Parameter(initializer(Tensor(np.array([[[
                [-5, -4, 0, 8],
                [-10, -2, 2, 3],
                [0, -2, -4, -7],
                [-3, -2, -3, -16]]]]).astype(np.float32)), [1, 1, 4, 4]), name='y')
            self.get_shape = P.Shape()

        def construct(self):
            return self.conv_filter(self.out, self.x, self.get_shape(self.w))

    net = TestNet()
    grad = GradNet(net)
    grad.compile()


def test_up_sample_nearest_3d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the up_sample_nearest_3d.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
               .reshape([1, 1, 2, 2, 4]), mstype.float32)
    output_size = [3, 4, 5]
    net = ops.UpsampleNearest3D(output_size=output_size)
    grad_compile_(x, net)


def test_upsample_trilinear_3d():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the upsample_trilinear_3D.
    Expectation: Load the bprop mindir successfully.
    """
    op = ops.UpsampleTrilinear3D(output_size=[4, 64, 48])
    grad_compile_(Tensor(input_data=np.random.randn(2, 3, 4, 512, 256)), op)


def test_add():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the add op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.ones([1]).astype(np.int32) * 100)
    y = Tensor(np.ones([1]).astype(np.int32) * 100)
    add = Net(P.Add())
    grad = GradNet(add)
    grad.compile(x, y)


def test_neg():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the batch neg op.
    Expectation: Load the bprop mindir successfully.
    """
    x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
    neg_net = Net(ops.Neg())
    grad = GradNet(neg_net)
    grad.compile(x)
