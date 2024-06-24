# Copyright 2023 Huawei Technologies Co., Ltd
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
import collections
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.ops.operations._sequence_ops import SequenceUnstack
from sequence_help import context_prepare, GradOfFirstInput

Res = collections.namedtuple('Res', ['out', 'y', 'out_n', 'y_n'])

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
context_prepare()


class SequenceUnstackNet(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.sequence_unstack = SequenceUnstack(axis=axis)

    def construct(self, input_x):
        return self.sequence_unstack(input_x)


class SequenceUnstackDynamicRankNet(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducemean = ops.ReduceMean(keep_dims=False)
        self.sequence_unstack = SequenceUnstack(axis=axis)

    def construct(self, input_x, random_indices):
        unique_indices = self.relu(random_indices)
        reduced_input_x = self.reducemean(input_x.astype(mstype.float32), unique_indices)
        reduced_input_x = reduced_input_x.astype(input_x.dtype)
        return self.sequence_unstack(reduced_input_x)


class UnstackNet(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.unstack = P.Unstack(axis=axis)

    def construct(self, input_x):
        return self.unstack(input_x)


class UnstackDynamicRankNet(nn.Cell):
    def __init__(self, axis):
        super().__init__()
        self.relu = ops.ReLU()
        self.reducemean = ops.ReduceMean(keep_dims=False)
        self.unstack = P.Unstack(axis=axis)

    def construct(self, input_x, random_indices):
        unique_indices = self.relu(random_indices)
        reduced_input_x = self.reducemean(input_x.astype(mstype.float32), unique_indices)
        reduced_input_x = reduced_input_x.astype(input_x.dtype)
        return self.unstack(reduced_input_x)


def to_numpy(out_res):
    out_np = []
    for out in out_res:
        out_np.append(out.asnumpy())
    return out_np


def input_grad_comp(x_input, net, out_grad_np):
    grad_net = GradOfFirstInput(net)
    grad_net.set_train()
    out_grad_tensor = ()
    out_grad_np_len = len(out_grad_np)
    for index in range(out_grad_np_len):
        out_grad_tensor += (Tensor(out_grad_np[index]),)
    res_input_grad = grad_net(x_input, out_grad_tensor)
    return res_input_grad


def allclose_nparray_sequence(res, y):
    for res_item, y_item in zip(res, y):
        assert np.allclose(y_item.asnumpy(), res_item.asnumpy(), 1e-5, 1e-5, equal_nan=True)


def unstack_dynamic_shape_impl(input_x, axis):
    input_x_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
    net = UnstackNet(axis)
    net.set_inputs(input_x_dyn)
    out = net(input_x)
    return out


def sequence_unstack_dynamic_shape_impl(input_x, axis):
    input_x_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
    net = SequenceUnstackNet(axis)
    net.set_inputs(input_x_dyn)
    out = net(input_x)
    return out


def dynamic_rank_impl(input_x, input_n, axis):
    random_indices_np = np.unique(np.random.randint(0, 1, (2,)).astype(np.int32))
    random_indices = Tensor(random_indices_np)

    input_n = ops.concat((input_n, input_n))
    new_shape_n = (2, input_n.shape[0] // 2, *input_n.shape[1:])
    input_n = input_n.reshape(*new_shape_n)

    input_x = ops.concat((input_x, input_x))
    new_shape = (2, input_x.shape[0] // 2, *input_x.shape[1:])
    input_x = input_x.reshape(*new_shape)
    input_x_dyn = Tensor(shape=[None for _ in input_x.shape], dtype=input_x.dtype)
    random_indices_dyn = Tensor(shape=[None for _ in random_indices.shape], dtype=random_indices.dtype)
    net = SequenceUnstackDynamicRankNet(axis)
    net.set_inputs(input_x_dyn, random_indices_dyn)
    out = net(input_x, random_indices)
    out_n = net(input_n, random_indices)
    y = UnstackDynamicRankNet(axis)(input_x, random_indices)
    y_n = UnstackDynamicRankNet(axis)(input_n, random_indices)
    res = Res(out, y, out_n, y_n)
    return res


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_shape_float32():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float32
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    res = sequence_unstack_dynamic_shape_impl(x_data, 0)
    y = unstack_dynamic_shape_impl(x_data, 0)
    allclose_nparray_sequence(res, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_shape_float64():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float64
    start = Tensor(1, dtype)
    limit = Tensor(61, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5, 2)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    res = sequence_unstack_dynamic_shape_impl(x_data, 1)
    y = unstack_dynamic_shape_impl(x_data, 1)
    allclose_nparray_sequence(res, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_shape_int32():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.int32
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    res = sequence_unstack_dynamic_shape_impl(x_data, 2)
    y = unstack_dynamic_shape_impl(x_data, 2)
    allclose_nparray_sequence(res, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_rank_int32():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.int32
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    shape_x_n = (3, 2, 4)
    limit_n = Tensor(25, dtype)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    x_data_n = ops.reshape(ops.range(start, limit_n, delta), shape_x_n).astype(dtype)
    res, y, res_n, y_n = dynamic_rank_impl(x_data, x_data_n, 0)
    allclose_nparray_sequence(res, y)
    allclose_nparray_sequence(res_n, y_n)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_rank_float32():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float32
    start = Tensor(1, dtype)
    limit = Tensor(61, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5, 2)
    shape_x_n = (3, 2, 4, 3)
    limit_n = Tensor(73, dtype)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    x_data_n = ops.reshape(ops.range(start, limit_n, delta), shape_x_n).astype(dtype)
    res, y, res_n, y_n = dynamic_rank_impl(x_data, x_data_n, 1)
    allclose_nparray_sequence(res, y)
    allclose_nparray_sequence(res_n, y_n)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack_dynamic_rank_float64():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float64
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    shape_x_n = (3, 2, 4)
    limit_n = Tensor(25, dtype)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x).astype(dtype)
    x_data_n = ops.reshape(ops.range(start, limit_n, delta), shape_x_n).astype(dtype)
    res, y, res_n, y_n = dynamic_rank_impl(x_data, x_data_n, 2)
    allclose_nparray_sequence(res, y)
    allclose_nparray_sequence(res_n, y_n)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_unstack0():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float32
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x)
    sequence_unstack_net = SequenceUnstackNet(axis=0)
    res = sequence_unstack_net(x_data)
    y = UnstackNet(0)(x_data)
    allclose_nparray_sequence(res, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_seq_tensor_unstack1():
    """
    Feature: test sequence unstack op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    dtype = mstype.float64
    start = Tensor(1, dtype)
    limit = Tensor(121, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5, 4)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x)
    sequence_unstack_net = SequenceUnstackNet(axis=2)
    res = sequence_unstack_net(x_data)
    y = ops.unstack(x_data, axis=2)
    allclose_nparray_sequence(res, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_in_seq_grad_other():
    """
    Feature: test sequence unstack grad op
    Description: inputs are dynamic sequence.
    Expectation: the result match with tuple result
    """
    axis = 2
    dtype = mstype.float64
    start = Tensor(1, dtype)
    limit = Tensor(31, dtype)
    delta = Tensor(1, dtype)
    shape_x = (2, 3, 5)
    x_data = ops.reshape(ops.range(start, limit, delta), shape_x)

    seq_net = SequenceUnstackNet(axis=axis)
    seq_out_grad_np = seq_net(x_data)
    sequence_unstack_res = input_grad_comp(x_data, seq_net, seq_out_grad_np)

    net = UnstackNet(axis=axis)
    out_grad_np = net(x_data)
    unstack_res = input_grad_comp(x_data, net, out_grad_np)
    allclose_nparray_sequence(sequence_unstack_res, unstack_res)
