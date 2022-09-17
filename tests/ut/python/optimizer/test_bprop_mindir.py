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

import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.ops.functional as F
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops.bprop_mindir import serializable_bprop_ops
from mindspore._c_expression import load_mindir
import mindspore.ops._grad as g


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


def test_load_mindir_dir():
    """
    Feature: Bprop pre-compilation.
    Description: Load all the mindir files of serializable bprop.
    Expectation: All are loaded successfully.
    """
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = bprop_installed_dir + "/../bprop_mindir"
    for op in serializable_bprop_ops:
        if isinstance(op, str):
            op_name = op
        else:
            op_name = op.__name__
        file_name = bprop_mindir_export_dir + "/" + op_name + "_bprop.mindir"
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


def test_stop_gradient():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the stop_gradient op.
    Expectation: Load the bprop mindir successfully.
    """

    class StopGradientNet(nn.Cell):
        def construct(self, x, y):
            c = x * y
            c_s = F.stop_gradient(c)
            return c_s

    x = Tensor(np.ones([4, 5]), mstype.float32)
    y = Tensor(np.ones([4, 5]), mstype.float32)
    stop_gradient = StopGradientNet()
    grad = GradNet(stop_gradient)
    grad.compile(x, y)


def test_switch():
    """
    Feature: Bprop pre-compilation.
    Description: Compile the backward graph for the switch op.
    Expectation: Load the bprop mindir successfully.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

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
