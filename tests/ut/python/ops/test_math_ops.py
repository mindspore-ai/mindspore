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
""" test math ops """
import functools

import numpy as np

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config

context.set_context(mode=context.GRAPH_MODE)

# pylint: disable=W0613
# pylint: disable=W0231
# W0613: unused-argument
# W0231: super-init-not-called

grad = C.GradOperation()


def test_multiply():
    """ test_multiply """
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]))
    input_y = Tensor(np.array([[0.1, 0.3, -3.6], [0.4, 0.5, -3.2]]))

    mul = P.Mul()
    result = mul(input_x, input_y)
    expect = np.array([[-0.01, 0.09, -12.96], [0.16, 0.25, 10.24]])
    diff = result.asnumpy() - expect
    error = np.ones(shape=[2, 3]) * 1.0e-6
    assert np.all(diff < error)
    assert np.all(-diff < error)


def test_sub():
    """ test_sub """
    input_x = Tensor(np.ones(shape=[3]))
    input_y = Tensor(np.zeros(shape=[3]))

    sub = P.Sub()
    result = sub(input_x, input_y)
    expect = np.ones(shape=[3])
    assert np.all(result.asnumpy() == expect)


def test_square():
    """ test_square """
    input_tensor = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    square = P.Square()
    result = square(input_tensor)
    expect = np.array([[1, 4, 9], [16, 25, 36]])
    assert np.all(result.asnumpy() == expect)


def test_sqrt():
    """ test_sqrt """
    input_tensor = Tensor(np.array([[4, 4], [9, 9]]))

    sqrt = P.Sqrt()
    expect = np.array([[2, 2], [3, 3]])
    result = sqrt(input_tensor)
    assert np.all(result.asnumpy() == expect)


class PowNet(nn.Cell):
    def __init__(self):
        super(PowNet, self).__init__()
        self.pow = P.Pow()

    def construct(self, x, y):
        return self.pow(x, y)


def test_pow():
    """ test_pow """
    input_tensor = Tensor(np.array([[2, 2], [3, 3]]))
    power = Tensor(np.array(3.0, np.int64))
    power2 = Tensor(np.array(True, np.bool))
    testpow = P.Pow()
    expect = np.array([[8, 8], [27, 27]])
    result = testpow(input_tensor, power)
    assert np.all(result.asnumpy() == expect)
    net = PowNet()
    net(input_tensor, power2)


def test_exp():
    """ test_exp """
    input_tensor = Tensor(np.array([[2, 2], [3, 3]]))
    testexp = P.Exp()
    result = testexp(input_tensor)
    expect = np.exp(np.array([[2, 2], [3, 3]]))
    assert np.all(result.asnumpy() == expect)


def test_realdiv():
    """ test_realdiv """
    x = Tensor(2048.0)
    y = Tensor(128.0)
    div = P.RealDiv()
    result = div(x, y)
    x = x.asnumpy()
    y = y.asnumpy()
    expect = x / y
    assert np.all(result.asnumpy() == expect)


def test_eye():
    """ test_eye """
    x = np.arange(3)
    expect = np.ones_like(x)
    expect = np.diag(expect)
    eye = P.Eye()
    eye_output = eye(3, 3, ms.float32)
    assert np.all(eye_output.asnumpy() == expect)


class VirtualLossGrad(PrimitiveWithInfer):
    """ VirtualLossGrad definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLossGrad"""

    def __call__(self, x, out, dout):
        raise NotImplementedError

    def infer_shape(self, x_shape, out_shape, dout_shape):
        return x_shape

    def infer_dtype(self, x_dtype, out_dtype, dout_dtype):
        return x_dtype


class VirtualLoss(PrimitiveWithInfer):
    """ VirtualLoss definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLoss"""

    def __call__(self, x):
        raise NotImplementedError

    def get_bprop(self):
        loss_grad = VirtualLossGrad()

        def bprop(x, out, dout):
            dx = loss_grad(x, out, dout)
            return (dx,)

        return bprop

    def infer_shape(self, x_shape):
        return [1]

    def infer_dtype(self, x_dtype):
        return x_dtype


class NetWithLoss(nn.Cell):
    """ NetWithLoss definition """

    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad(self.network)(x, y, b)


class MatMulNet(nn.Cell):
    """ MatMulNet definition """

    def __init__(self):
        super(MatMulNet, self).__init__()
        self.matmul = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x, y, b):
        return self.biasAdd(self.matmul(x, y), b)


class NetWithLossSub(nn.Cell):
    """ NetWithLossSub definition """

    def __init__(self, network):
        super(NetWithLossSub, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrapSub(nn.Cell):
    """ GradWrapSub definition """

    def __init__(self, network):
        super(GradWrapSub, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad(self.network)(x, y)


class SubNet(nn.Cell):
    """ SubNet definition """

    def __init__(self):
        super(SubNet, self).__init__()
        self.sub = P.Sub()

    def construct(self, x, y):
        return self.sub(x, y)


class NpuFloatNet(nn.Cell):
    """ NpuFloat definition """

    def __init__(self):
        super(NpuFloatNet, self).__init__()
        self.mul = P.Mul()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.fill = P.Fill()
        self.shape_op = P.Shape()
        self.select = P.Select()
        self.less = P.Less()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.neg = P.Neg()

    def construct(self, x):
        init = self.alloc_status()
        clear_status = self.clear_status(init)
        x = F.depend(x, clear_status) # let x depend on clear_status
        res = self.sub(x, self.neg(x))
        init = F.depend(init, res) # let get_status depend on res
        get_status = self.get_status(init)
        init = F.depend(init, get_status) # let reduce_sum depend on get_statusk
        flag_sum = self.reduce_sum(init, (0,))
        base = self.cast(self.fill(self.dtype(res), self.shape_op(res), 0.0), self.dtype(flag_sum))
        cond = self.less(base, flag_sum)
        out = self.select(cond, self.cast(base, self.dtype(res)), res)
        return out


class DiagNet(nn.Cell):
    """ DiagNet definition """

    def __init__(self):
        super(DiagNet, self).__init__()
        self.fill = P.Fill()
        self.diag = P.Diag()

    def construct(self, x):
        return x - self.diag(self.fill(mstype.float32, (3,), 1.0))


class NetWithLossCumSum(nn.Cell):
    """ NetWithLossCumSum definition """

    def __init__(self, network):
        super(NetWithLossCumSum, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, input_):
        predict = self.network(input_)
        return self.loss(predict)


class GradWrapCumSum(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrapCumSum, self).__init__()
        self.network = network

    def construct(self, input_):
        return grad(self.network)(input_)


class NetCumSum(nn.Cell):
    """ NetCumSum definition """

    def __init__(self):
        super(NetCumSum, self).__init__()
        self.cumsum = P.CumSum()
        self.axis = 1

    def construct(self, input_):
        return self.cumsum(input_, self.axis)


class SignNet(nn.Cell):
    def __init__(self):
        super(SignNet, self).__init__()
        self.sign = P.Sign()

    def construct(self, x):
        return self.sign(x)


class AssignAdd(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = P.AssignAdd()
        self.inputdata = Parameter(initializer(1, [1], ms.float32), name="global_step")

    def construct(self, input_):
        self.inputdata = input_
        return self.op(self.inputdata, input_)


class FloorNet(nn.Cell):
    def __init__(self):
        super(FloorNet, self).__init__()
        self.floor = P.Floor()

    def construct(self, x):
        return self.floor(x)


class Log1pNet(nn.Cell):
    def __init__(self):
        super(Log1pNet, self).__init__()
        self.log1p = P.Log1p()

    def construct(self, x):
        return self.log1p(x)


class ErfcNet(nn.Cell):
    def __init__(self):
        super(ErfcNet, self).__init__()
        self.erfc = P.Erfc()

    def construct(self, x):
        return self.erfc(x)


test_case_math_ops = [
    ('MatMulGrad', {
        'block': GradWrap(NetWithLoss(MatMulNet())),
        'desc_inputs': [Tensor(np.ones([3, 3]).astype(np.int32)),
                        Tensor(np.ones([3, 3]).astype(np.int32)),
                        Tensor(np.ones([3]).astype(np.int32))],
        'desc_bprop': [Tensor(np.ones([3, 3]).astype(np.int32)),
                       Tensor(np.ones([3, 3]).astype(np.int32)),
                       Tensor(np.ones([3]).astype(np.int32))],
        'skip': ['backward']}),
    ('CumSumGrad', {
        'block': GradWrapCumSum(NetWithLossCumSum(NetCumSum())),
        'desc_inputs': [Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float16))],
        'desc_bprop': [Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float16))],
        'skip': ['backward']}),
    ('Diag', {
        'block': DiagNet(),
        'desc_inputs': [Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32))],
        'skip': ['backward']}),
    ('SubBroadcast', {
        'block': GradWrapSub(NetWithLossSub(SubNet())),
        'desc_inputs': [Tensor(np.ones([5, 3])), Tensor(np.ones([8, 5, 3]))],
        'desc_bprop': [Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.float32))],
        'skip': ['backward']}),
    ('NpuFloat_NotOverflow', {
        'block': NpuFloatNet(),
        'desc_inputs': [Tensor(np.full((8, 5, 3, 1), 655, dtype=np.float16), dtype=ms.float16)],
        'desc_bprop': [Tensor(np.full((8, 5, 3, 1), 655, dtype=np.float16), dtype=ms.float16)],
        'skip': ['backward']}),
    ('NpuFloat_Overflow', {
        'block': NpuFloatNet(),
        'desc_inputs': [Tensor(np.full((8, 5, 3, 1), 65504, dtype=np.float16), dtype=ms.float16)],
        'desc_bprop': [Tensor(np.full((8, 5, 3, 1), 65504, dtype=np.float16), dtype=ms.float16)],
        'skip': ['backward']}),
    ('Sign', {
        'block': SignNet(),
        'desc_inputs': [Tensor(np.array([[1., 0., -2.]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1., 0., -2.]], np.float32))],
        'skip': ['backward']}),
    ('Floor', {
        'block': FloorNet(),
        'desc_inputs': [Tensor(np.array([[1., 0., -2.]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1., 0., -2.]], np.float32))],
        'skip': ['backward']}),
    ('Log1p', {
        'block': Log1pNet(),
        'desc_inputs': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
        'skip': ['backward']}),
    ('Erfc', {
        'block': ErfcNet(),
        'desc_inputs': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
    }),
]

test_case_lists = [test_case_math_ops]
test_exec_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case


raise_set = [
    ('StridedSlice_1_Error', {
        'block': (lambda x: P.StridedSlice(begin_mask="1"), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('StridedSlice_2_Error', {
        'block': (lambda x: P.StridedSlice(end_mask="1"), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('StridedSlice_3_Error', {
        'block': (lambda x: P.StridedSlice(ellipsis_mask=1.1), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('StridedSlice_4_Error', {
        'block': (lambda x: P.StridedSlice(new_axis_mask="1.1"), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('AssignAdd_Error', {
        'block': (P.AssignAdd(), {'exception': ValueError}),
        'desc_inputs': [[1]]}),
]


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return raise_set
