# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test array ops """
import functools
import numpy as np
import pytest
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import prim_attr_register
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.primitive import PrimitiveWithInfer
from mindspore.ops.signature import sig_rw, sig_dtype, make_sig


from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config

context.set_context(mode=context.PYNATIVE_MODE)


def test_expand_dims():
    input_tensor = Tensor(np.array([[2, 2], [2, 2]]))
    expand_dims = P.ExpandDims()
    output = expand_dims(input_tensor, 0)
    assert output.asnumpy().shape == (1, 2, 2)


def test_cast():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_x = Tensor(input_np)
    td = ms.int32
    cast = P.Cast()
    result = cast(input_x, td)
    expect = input_np.astype(np.int32)
    assert np.all(result.asnumpy() == expect)


def test_ones():
    ones = P.Ones()
    output = ones((2, 3), mstype.int32)
    assert output.asnumpy().shape == (2, 3)
    assert np.sum(output.asnumpy()) == 6


def test_ones_1():
    ones = P.Ones()
    output = ones(2, mstype.int32)
    assert output.asnumpy().shape == (2,)
    assert np.sum(output.asnumpy()) == 2


def test_zeros():
    zeros = P.Zeros()
    output = zeros((2, 3), mstype.int32)
    assert output.asnumpy().shape == (2, 3)
    assert np.sum(output.asnumpy()) == 0


def test_zeros_1():
    zeros = P.Zeros()
    output = zeros(2, mstype.int32)
    assert output.asnumpy().shape == (2,)
    assert np.sum(output.asnumpy()) == 0


@non_graph_engine
def test_reshape():
    input_tensor = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]))
    shp = (3, 2)
    reshape = P.Reshape()
    output = reshape(input_tensor, shp)
    assert output.asnumpy().shape == (3, 2)


def test_transpose():
    input_tensor = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
    perm = (0, 2, 1)
    expect = np.array([[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]])

    transpose = P.Transpose()
    output = transpose(input_tensor, perm)
    assert np.all(output.asnumpy() == expect)


def test_squeeze():
    input_tensor = Tensor(np.ones(shape=[3, 2, 1]))
    squeeze = P.Squeeze(2)
    output = squeeze(input_tensor)
    assert output.asnumpy().shape == (3, 2)


def test_invert_permutation():
    invert_permutation = P.InvertPermutation()
    x = (3, 4, 0, 2, 1)
    output = invert_permutation(x)
    expect = (2, 4, 3, 0, 1)
    assert np.all(output == expect)


def test_select():
    select = P.Select()
    cond = Tensor(np.array([[True, False, False], [False, True, True]]))
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    y = Tensor(np.array([[7, 8, 9], [10, 11, 12]]))
    output = select(cond, x, y)
    expect = np.array([[1, 8, 9], [10, 5, 6]])
    assert np.all(output.asnumpy() == expect)


def test_argmin_invalid_output_type():
    P.Argmin(-1, mstype.int64)
    P.Argmin(-1, mstype.int32)
    with pytest.raises(TypeError):
        P.Argmin(-1, mstype.float32)
    with pytest.raises(TypeError):
        P.Argmin(-1, mstype.float64)
    with pytest.raises(TypeError):
        P.Argmin(-1, mstype.uint8)
    with pytest.raises(TypeError):
        P.Argmin(-1, mstype.bool_)


class CustomOP(PrimitiveWithInfer):
    __mindspore_signature__ = (sig_dtype.T, sig_dtype.T, sig_dtype.T1,
                               sig_dtype.T1, sig_dtype.T2, sig_dtype.T2,
                               sig_dtype.T2, sig_dtype.T3, sig_dtype.T4)

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        raise NotImplementedError


class CustomOP2(PrimitiveWithInfer):
    __mindspore_signature__ = (
        make_sig('p1', sig_rw.RW_WRITE, dtype=sig_dtype.T),
        make_sig('p2', dtype=sig_dtype.T),
        make_sig('p3', dtype=sig_dtype.T),
    )

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, p1, p2, p3):
        raise NotImplementedError


class CustNet1(Cell):
    def __init__(self):
        super(CustNet1, self).__init__()
        self.op = CustomOP()
        self.t1 = Tensor(np.ones([2, 2]), dtype=ms.int32)
        self.t2 = Tensor(np.ones([1, 5]), dtype=ms.float16)
        self.int1 = 3
        self.float1 = 5.1

    def construct(self):
        x = self.op(self.t1, self.t1, self.int1,
                    self.float1, self.int1, self.float1,
                    self.t2, self.t1, self.int1)
        return x


class CustNet2(Cell):
    def __init__(self):
        super(CustNet2, self).__init__()
        self.op = CustomOP2()
        self.t1 = Tensor(np.ones([2, 2]), dtype=ms.int32)
        self.t2 = Tensor(np.ones([1, 5]), dtype=ms.float16)
        self.int1 = 3

    def construct(self):
        return self.op(self.t1, self.t2, self.int1)


class CustNet3(Cell):
    def __init__(self):
        super(CustNet3, self).__init__()
        self.op = P.ReduceSum()
        self.t1 = Tensor(np.ones([2, 2]), dtype=ms.int32)
        self.t2 = Tensor(np.ones([1, 5]), dtype=ms.float16)
        self.t2 = 1

    def construct(self):
        return self.op(self.t1, self.t2)


class MathBinaryNet1(Cell):
    def __init__(self):
        super(MathBinaryNet1, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.max = P.Maximum()
        self.number = 3

    def construct(self, x):
        return self.add(x, self.number) + self.mul(x, self.number) + self.max(x, self.number)


class MathBinaryNet2(Cell):
    def __init__(self):
        super(MathBinaryNet2, self).__init__()
        self.less_equal = P.LessEqual()
        self.greater = P.Greater()
        self.logic_or = P.LogicalOr()
        self.logic_and = P.LogicalAnd()
        self.number = 3
        self.flag = True

    def construct(self, x):
        ret_less_equal = self.logic_and(self.less_equal(x, self.number), self.flag)
        ret_greater = self.logic_or(self.greater(x, self.number), self.flag)
        return self.logic_or(ret_less_equal, ret_greater)


class BatchToSpaceNet(Cell):
    def __init__(self):
        super(BatchToSpaceNet, self).__init__()
        block_size = 2
        crops = [[0, 0], [0, 0]]
        self.batch_to_space = P.BatchToSpace(block_size, crops)

    def construct(self, x):
        return self.batch_to_space(x)


class SpaceToBatchNet(Cell):
    def __init__(self):
        super(SpaceToBatchNet, self).__init__()
        block_size = 2
        paddings = [[0, 0], [0, 0]]
        self.space_to_batch = P.SpaceToBatch(block_size, paddings)

    def construct(self, x):
        return self.space_to_batch(x)


class PackNet(Cell):
    def __init__(self):
        super(PackNet, self).__init__()
        self.stack = P.Stack()

    def construct(self, x):
        return self.stack((x, x))


class UnpackNet(Cell):
    def __init__(self):
        super(UnpackNet, self).__init__()
        self.unstack = P.Unstack()

    def construct(self, x):
        return self.unstack(x)


class SpaceToDepthNet(Cell):
    def __init__(self):
        super(SpaceToDepthNet, self).__init__()
        block_size = 2
        self.space_to_depth = P.SpaceToDepth(block_size)

    def construct(self, x):
        return self.space_to_depth(x)


class DepthToSpaceNet(Cell):
    def __init__(self):
        super(DepthToSpaceNet, self).__init__()
        block_size = 2
        self.depth_to_space = P.DepthToSpace(block_size)

    def construct(self, x):
        return self.depth_to_space(x)


class BatchToSpaceNDNet(Cell):
    def __init__(self):
        super(BatchToSpaceNDNet, self).__init__()
        block_shape = [2, 2]
        crops = [[0, 0], [0, 0]]
        self.batch_to_space_nd = P.BatchToSpaceND(block_shape, crops)

    def construct(self, x):
        return self.batch_to_space_nd(x)


class SpaceToBatchNDNet(Cell):
    def __init__(self):
        super(SpaceToBatchNDNet, self).__init__()
        block_shape = [2, 2]
        paddings = [[0, 0], [0, 0]]
        self.space_to_batch_nd = P.SpaceToBatchND(block_shape, paddings)

    def construct(self, x):
        return self.space_to_batch_nd(x)


class TensorShapeNet(Cell):
    def __init__(self):
        super(TensorShapeNet, self).__init__()
        self.shape = P.TensorShape()
        self.unique = P.Unique()

    def construct(self, x):
        x, _ = self.unique(x)
        return self.shape(x)


class UniqueFunc1(Cell):
    def __init__(self):
        super(UniqueFunc1, self).__init__()
        self.unique = ops.unique

    def construct(self, x):
        y, idx = self.unique(x)
        return y, idx


class UniqueFunc2(Cell):
    def __init__(self):
        super(UniqueFunc2, self).__init__()
        self.unique = ops.unique

    def construct(self, x):
        y, idx = self.unique(x)
        return y, idx


class ArgMaxWithValueFunc(Cell):
    def __init__(self):
        super(ArgMaxWithValueFunc, self).__init__()
        self.argmax_ = ops.function.max

    def construct(self, x):
        return self.argmax_(x, axis=0, keep_dims=False)


class ArgMinWithValueFunc(Cell):
    def __init__(self):
        super(ArgMinWithValueFunc, self).__init__()
        self.argmin_ = ops.function.min

    def construct(self, x):
        return self.argmin_(x, axis=0, keep_dims=False)


class AminmaxFunc(Cell):
    def __init__(self):
        super(AminmaxFunc, self).__init__()
        self.aminmax_ = ops.function.array_func.aminmax

    def construct(self, x):
        return self.aminmax_(x, axis=0, keepdims=False)


class RangeNet(Cell):
    def __init__(self):
        super(RangeNet, self).__init__()
        self.range_ops = P.Range()

    def construct(self, start, limit, delta):
        return self.range_ops(start, limit, delta)


class NetForFlattenConcat(Cell):
    def __init__(self):
        super(NetForFlattenConcat, self).__init__()
        self.flatten_concat = inner.FlattenConcat()

    def construct(self, x1, x2, x3):
        return self.flatten_concat([x1, x2, x3])


class MaskedFillFunc(Cell):
    def __init__(self):
        super(MaskedFillFunc, self).__init__()
        self.maskedfill_ = ops.function.masked_fill

    def construct(self, x, mask, value):
        y = self.maskedfill_(x, mask, value)
        return y


test_case_array_ops = [
    ('CustNet1', {
        'block': CustNet1(),
        'desc_inputs': []}),
    ('CustNet2', {
        'block': CustNet2(),
        'desc_inputs': []}),
    ('CustNet3', {
        'block': CustNet3(),
        'desc_inputs': []}),
    ('Unique', {
        'block': UniqueFunc1(),
        'desc_inputs': [Tensor(np.array([2, 2, 1]), dtype=ms.int32)]}),
    ('Unique', {
        'block': UniqueFunc2(),
        'desc_inputs': [Tensor(np.array([[2, 2], [1, 3]]), dtype=ms.int32)]}),
    ('MathBinaryNet1', {
        'block': MathBinaryNet1(),
        'desc_inputs': [Tensor(np.ones([2, 2]), dtype=ms.int32)]}),
    ('MathBinaryNet2', {
        'block': MathBinaryNet2(),
        'desc_inputs': [Tensor(np.ones([2, 2]), dtype=ms.int32)]}),
    ('BatchToSpaceNet', {
        'block': BatchToSpaceNet(),
        'desc_inputs': [Tensor(np.array([[[[1]]], [[[2]]], [[[3]]], [[[4]]]]).astype(np.float16))]}),
    ('SpaceToBatchNet', {
        'block': SpaceToBatchNet(),
        'desc_inputs': [Tensor(np.array([[[[1, 2], [3, 4]]]]).astype(np.float16))]}),
    ('PackNet', {
        'block': PackNet(),
        'desc_inputs': [Tensor(np.array([[[1, 2], [3, 4]]]).astype(np.float16))]}),
    ('UnpackNet', {
        'block': UnpackNet(),
        'desc_inputs': [Tensor(np.array([[1, 2], [3, 4]]).astype(np.float16))]}),
    ('SpaceToDepthNet', {
        'block': SpaceToDepthNet(),
        'desc_inputs': [Tensor(np.random.rand(1, 3, 2, 2).astype(np.float16))]}),
    ('DepthToSpaceNet', {
        'block': DepthToSpaceNet(),
        'desc_inputs': [Tensor(np.random.rand(1, 12, 1, 1).astype(np.float16))]}),
    ('SpaceToBatchNDNet', {
        'block': SpaceToBatchNDNet(),
        'desc_inputs': [Tensor(np.random.rand(1, 1, 2, 2).astype(np.float16))]}),
    ('BatchToSpaceNDNet', {
        'block': BatchToSpaceNDNet(),
        'desc_inputs': [Tensor(np.random.rand(4, 1, 1, 1).astype(np.float16))]}),
    ('RangeNet', {
        'block': RangeNet(),
        'desc_inputs': [Tensor(np.array(1.0), ms.float32),
                        Tensor(np.array(8.0), ms.float32),
                        Tensor(np.array(2.0), ms.float32)]}),
    ('ArgMaxWithValue', {
        'block': ArgMaxWithValueFunc(),
        'desc_inputs': [Tensor(np.array([1., 2., 4., 3.]), ms.float32)]}),
    ('ArgMinWithValue', {
        'block': ArgMinWithValueFunc(),
        'desc_inputs': [Tensor(np.array([1., 4., 2., 3.]), ms.float32)]}),
    ('Aminmax', {
        'block': AminmaxFunc(),
        'desc_inputs': [Tensor(np.array([1., 2., 4., 3.]), ms.float32)]}),
    ('FlattenConcat', {
        'block': NetForFlattenConcat(),
        'desc_inputs': [Tensor(np.array([1], np.float32)),
                        Tensor(np.array([2], np.float32)),
                        Tensor(np.array([3], np.float64))]}),
    ('MaskedFill', {
        'block': MaskedFillFunc(),
        'desc_inputs': [Tensor(np.array([[3.0, 2.0, 1.0]]), mstype.float32),
                        Tensor(np.array([[True, True, False]]), mstype.bool_),
                        Tensor(5.0, mstype.float32)],
        'desc_bprop': [Tensor(np.array([[3.0, 2.0, 1.0]]), mstype.float32)]}),
    ('TensorShapeNet', {'block': TensorShapeNet(), 'desc_inputs': [Tensor(np.array([1, 2, 3, 2]), ms.int32)]})
]

test_case_lists = [test_case_array_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)


# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case


raise_set = [
    ('Squeeze_1_Error', {
        'block': (lambda x: P.Squeeze(axis=1.2), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5]))]}),
    ('Squeeze_2_Error', {
        'block': (lambda x: P.Squeeze(axis=((1.2, 1.3))), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5]))]}),
    ('ReduceSum_Error', {
        'block': (lambda x: P.ReduceSum(keep_dims=1), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.ones(shape=[3, 1, 5]))]}),
    ('TensorShapeNet_Error', {'block': (lambda x: P.TensorSHape(), {'exception': TypeError}),
                              'desc_inputs': [(Tensor(np.ones(shape=[3, 1, 5])), Tensor(np.ones(shape=[3, 1, 5])))]})
]


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return raise_set
