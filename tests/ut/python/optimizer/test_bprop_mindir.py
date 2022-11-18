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

import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
import mindspore.ops._grad as g
from mindspore.ops._grad.grad_base import bprop_getters, bprops
from mindspore._c_expression import _check_bprop_mindir
from mindspore import mutable


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
