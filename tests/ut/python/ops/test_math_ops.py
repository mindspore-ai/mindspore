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
from mindspore import ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations._grad_ops import IgammaGradA
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore.ops.operations.math_ops import Zeta, Igamma, Igammac, BatchMatMul
from mindspore.ops.operations.math_ops import MatrixTriangularSolve
from mindspore.ops.operations.sparse_ops import DenseToDenseSetOperation
from mindspore.ops.operations.sparse_ops import DenseToSparseSetOperation
from mindspore.ops.function.math_func import inplace_index_add, polar

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
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


class OrgqrFunc(nn.Cell):
    def __init__(self):
        super(OrgqrFunc, self).__init__()
        self.orgqr_ = ops.function.math_func.orgqr

    def construct(self, x, tau):
        y = self.orgqr_(x, tau)
        return y


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


class FmaxFunc(nn.Cell):
    def __init__(self):
        super(FmaxFunc, self).__init__()
        self.fmax_ = ops.function.math_func.fmax

    def construct(self, x1, x2):
        return self.fmax_(x1, x2)


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
        self.op(self.inputdata, input_)
        return self.inputdata


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


class LdexpFunc(nn.Cell):
    def __init__(self):
        super(LdexpFunc, self).__init__()
        self.ldexp = ops.ldexp

    def construct(self, x, other):
        return self.ldexp(x, other)


class BlockDiagFunc(nn.Cell):
    def __init__(self):
        super(BlockDiagFunc, self).__init__()
        self.block_diag = ops.block_diag

    def construct(self, x1, x2, x3, x4, x5):
        return self.block_diag(x1, x2, x3, x4, x5)


class AtLeast1DFunc(nn.Cell):
    def __init__(self):
        super(AtLeast1DFunc, self).__init__()
        self.atleast_1d = ops.atleast_1d

    def construct(self, x1, x2, x3):
        return self.atleast_1d([x1, x2, x3])


class DstackFunc(nn.Cell):
    def __init__(self):
        super(DstackFunc, self).__init__()
        self.dstack = ops.dstack

    def construct(self, x1, x2):
        return self.dstack([x1, x2])


class DiffFunc(nn.Cell):
    def __init__(self):
        super(DiffFunc, self).__init__()
        self.diff = ops.diff

    def construct(self, x):
        return self.diff(x)


class AtLeast2DFunc(nn.Cell):
    def __init__(self):
        super(AtLeast2DFunc, self).__init__()
        self.atleast_2d = ops.atleast_2d

    def construct(self, x1, x2, x3):
        return self.atleast_2d([x1, x2, x3])


class CartesianProdFunc(nn.Cell):
    def __init__(self):
        super(CartesianProdFunc, self).__init__()
        self.cartesian_prod = ops.cartesian_prod

    def construct(self, x1, x2):
        return self.cartesian_prod(x1, x2)


class AtLeast3DFunc(nn.Cell):
    def __init__(self):
        super(AtLeast3DFunc, self).__init__()
        self.atleast_3d = ops.atleast_3d

    def construct(self, x1, x2, x3):
        return self.atleast_3d([x1, x2, x3])


class VstackFunc(nn.Cell):
    def __init__(self):
        super(VstackFunc, self).__init__()
        self.vstack = ops.vstack

    def construct(self, x1, x2):
        return self.vstack([x1, x2])


class CombinationsFunc(nn.Cell):
    def __init__(self):
        super(CombinationsFunc, self).__init__()
        self.combinations = ops.combinations

    def construct(self, x):
        return self.combinations(x)


class DistFunc(nn.Cell):
    def __init__(self):
        super(DistFunc, self).__init__()
        self.dist = ops.dist

    def construct(self, input_x, input_y):
        return self.dist(input_x, input_y)


class CopysignFunc(nn.Cell):
    def __init__(self):
        super(CopysignFunc, self).__init__()
        self.copysign = ops.copysign

    def construct(self, x, other):
        return self.copysign(x, other)


class HannWindowFunc(nn.Cell):
    def __init__(self):
        super(HannWindowFunc, self).__init__()
        self.hann_window = ops.hann_window

    def construct(self, window_length):
        return self.hann_window(window_length)


class HypotFunc(nn.Cell):
    def __init__(self):
        super(HypotFunc, self).__init__()
        self.hypot_ = ops.function.hypot

    def construct(self, x1, x2):
        y = self.hypot_(x1, x2)
        return y


class NanSumFunc(nn.Cell):
    def __init__(self):
        super(NanSumFunc, self).__init__()
        self.nansum = ops.function.math_func.nansum

    def construct(self, x, axes):
        y = self.nansum(x, axes)
        return y


class HeavisideFunc(nn.Cell):
    def __init__(self):
        super(HeavisideFunc, self).__init__()
        self.heaviside_ = ops.function.heaviside

    def construct(self, x, values):
        y = self.heaviside_(x, values)
        return y


class LogAddExpFunc(nn.Cell):
    def __init__(self):
        super(LogAddExpFunc, self).__init__()
        self.logaddexp = ops.logaddexp

    def construct(self, x1, x2):
        y = self.logaddexp(x1, x2)
        return y


class LogAddExp2Func(nn.Cell):
    def __init__(self):
        super(LogAddExp2Func, self).__init__()
        self.logaddexp2 = ops.logaddexp2

    def construct(self, x1, x2):
        y = self.logaddexp2(x1, x2)
        return y


class KaiserWindowFunc(nn.Cell):
    def __init__(self):
        super(KaiserWindowFunc, self).__init__()
        self.kaiser_window = ops.kaiser_window

    def construct(self, window_length):
        return self.kaiser_window(window_length)


class AddmvFunc(nn.Cell):
    def __init__(self):
        super(AddmvFunc, self).__init__()
        self.addmv = ops.addmv

    def construct(self, x, mat, vec, beta=1, alpha=1):
        y = self.addmv(x, mat, vec, beta, alpha)
        return y


class AddrFunc(nn.Cell):
    def __init__(self):
        super(AddrFunc, self).__init__()
        self.addr = ops.addr

    def construct(self, x, vec1, vec2, beta=1, alpha=1):
        y = self.addr(x, vec1, vec2, beta, alpha)
        return y


class MvFunc(nn.Cell):
    def __init__(self):
        super(MvFunc, self).__init__()
        self.mv = ops.mv

    def construct(self, mat, vec):
        return self.mv(mat, vec)


class OuterFunc(nn.Cell):
    def __init__(self):
        super(OuterFunc, self).__init__()
        self.outer = ops.outer

    def construct(self, x1, x2):
        return self.outer(x1, x2)


class Exp2Func(nn.Cell):
    def __init__(self):
        super(Exp2Func, self).__init__()
        self.exp2 = ops.exp2

    def construct(self, x):
        y = self.exp2(x)
        return y


class Deg2radNet(nn.Cell):
    def __init__(self):
        super(Deg2radNet, self).__init__()
        self.deg2rad = ops.deg2rad

    def construct(self, x):
        return self.deg2rad(x)


class InplaceIndexAddFunc(nn.Cell):
    def __init__(self):
        super(InplaceIndexAddFunc, self).__init__()
        self.inplace_index_add = inplace_index_add
        self.var = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)))

    def construct(self, indices, updates, axis=1):
        return self.inplace_index_add(self.var, indices, updates, axis)


class IsRealFunc(nn.Cell):
    def __init__(self):
        super(IsRealFunc, self).__init__()
        self.isreal = ops.isreal

    def construct(self, x):
        y = self.isreal(x)
        return y


class LcmFunc(nn.Cell):
    def __init__(self):
        super(LcmFunc, self).__init__()
        self.lcm = ops.function.lcm

    def construct(self, x1, x2):
        return self.lcm(x1, x2)


class GcdFunc(nn.Cell):
    def __init__(self):
        super(GcdFunc, self).__init__()
        self.gcd = ops.function.gcd

    def construct(self, x1, x2):
        return self.gcd(x1, x2)


class Rad2degNet(nn.Cell):
    def __init__(self):
        super(Rad2degNet, self).__init__()
        self.rad2deg = ops.rad2deg

    def construct(self, x):
        return self.rad2deg(x)


class BaddbmmNet(nn.Cell):
    def __init__(self, beta=1, alpha=1):
        super(BaddbmmNet, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.baddbmm = ops.baddbmm

    def construct(self, x, batch1, batch2):
        return self.baddbmm(x, batch1, batch2, self.beta, self.alpha)


class Log2Net(nn.Cell):
    def __init__(self):
        super(Log2Net, self).__init__()
        self.log2 = ops.log2

    def construct(self, x):
        return self.log2(x)


class Log10Net(nn.Cell):
    def __init__(self):
        super(Log10Net, self).__init__()
        self.log10 = ops.log10

    def construct(self, x):
        return self.log10(x)


class FminFunc(nn.Cell):
    def __init__(self):
        super(FminFunc, self).__init__()
        self.fmin_ = ops.function.math_func.fmin

    def construct(self, x1, x2):
        return self.fmin_(x1, x2)


class FracNet(nn.Cell):
    def __init__(self):
        super(FracNet, self).__init__()
        self.frac = ops.frac

    def construct(self, x):
        return self.frac(x)


class KronFunc(nn.Cell):
    def __init__(self):
        super(KronFunc, self).__init__()
        self.kron = ops.kron

    def construct(self, x, y):
        return self.kron(x, y)


class PolarFunc(nn.Cell):
    def __init__(self):
        super(PolarFunc, self).__init__()
        self.polar = polar

    def construct(self, x, y):
        return self.polar(x, y)


class Rot90Func(nn.Cell):
    def __init__(self):
        super(Rot90Func, self).__init__()
        self.rot90 = ops.rot90
        self.k = 0
        self.dims = (0, 1)

    def construct(self, x):
        return self.rot90(x, self.k, self.dims)


class RemainderNet(nn.Cell):
    def __init__(self):
        super(RemainderNet, self).__init__()
        self.remainder = ops.remainder

    def construct(self, x, y):
        return self.remainder(x, y)


class TrapzFunc(nn.Cell):
    def __init__(self):
        super(TrapzFunc, self).__init__()
        self.trapz = ops.trapz

    def construct(self, y, x=None, dx=1.0, dim=-1):
        out = self.trapz(y, x, dx, dim)
        return out


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
    ('InplaceIndexAdd', {
        'block': InplaceIndexAddFunc(),
        'desc_inputs': [Tensor(np.array([0, 1, 2], np.int32)),
                        Tensor(np.array([[0.5, 1.0, 1.5], [1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32))]}),
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
    ('Ldexp', {
        'block': LdexpFunc(),
        'desc_inputs': [Tensor(np.array([1.]), dtype=ms.float32),
                        Tensor(np.array([1, 2, 3, 4]), dtype=ms.int32)],
        'skip': ['backward']
    }),
    ('BlockDiag', {
        'block': BlockDiagFunc(),
        'desc_inputs': [Tensor(np.array([[4], [3], [2]]), ms.int32),
                        Tensor(np.array([7, 6, 5]), ms.int32),
                        Tensor(np.array(1), ms.int32),
                        Tensor(np.array([[5, 4, 3], [2, 1, 0]]), ms.int32),
                        Tensor(np.array([[8, 7], [7, 8]]), ms.int32)]
    }),
    ('AtLeast1D', {
        'block': AtLeast1DFunc(),
        'desc_inputs': [Tensor(np.array([[1, 1, 1], [1, 1, 1]]), ms.float64),
                        Tensor(np.array(1), ms.float64),
                        Tensor(np.array([1, 1, 1, 1, 1]), ms.float64)]
    }),
    ('Dstack', {
        'block': DstackFunc(),
        'desc_inputs': [Tensor(np.array([1, 2, 3]), ms.float32),
                        Tensor(np.array([4, 5, 6]), ms.float32)]
    }),
    ('Diff', {
        'block': DiffFunc(),
        'desc_inputs': [Tensor(np.array([1, 3, -1, 0, 4]), ms.int32)]
    }),
    ('AtLeast2D', {
        'block': AtLeast2DFunc(),
        'desc_inputs': [Tensor(np.array([[1, 1, 1], [1, 1, 1]]), ms.float64),
                        Tensor(np.array(1), ms.float64),
                        Tensor(np.array([1, 1, 1, 1, 1]), ms.float64)]
    }),
    ('CartesianProd', {
        'block': CartesianProdFunc(),
        'desc_inputs': [Tensor(np.array([1, 2]), ms.int32),
                        Tensor(np.array([5]), ms.int32)]
    }),
    ('AtLeast3D', {
        'block': AtLeast3DFunc(),
        'desc_inputs': [Tensor(np.array([[1, 1, 1], [1, 1, 1]]), ms.float64),
                        Tensor(np.array(1), ms.float64),
                        Tensor(np.array([1, 1, 1, 1, 1]), ms.float64)]
    }),
    ('Vstack', {
        'block': VstackFunc(),
        'desc_inputs': [Tensor(np.array([1, 2, 3]), ms.int32),
                        Tensor(np.array([4, 5, 6]), ms.int32)]
    }),
    ('Combinations', {
        'block': CombinationsFunc(),
        'desc_inputs': [Tensor(np.array([1, 3, -1, 0, 4]), ms.int32)]
    }),
    ('Dist', {
        'block': DistFunc(),
        'desc_inputs': [Tensor(np.array([[[1, 1], [2, 2]]]), ms.float32),
                        Tensor(np.array([[[3, 3], [3, 3]]]), ms.float32)]
    }),
    ('Copysign', {
        'block': CopysignFunc(),
        'desc_inputs': [Tensor(np.array([[0.3, -0.7], [0.5, 0.5]])),
                        Tensor(np.array([[-0.4, 0.6], [0.4, -0.6]]))]
    }),
    ('HannWindow', {
        'block': HannWindowFunc(),
        'desc_inputs': [5]
    }),
    ('LogAddExp2', {
        'block': LogAddExp2Func(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float16)), Tensor(np.array([2.0], np.float16))],
        'desc_bprop': [Tensor(np.array([1.0, 2.0, 3.0], np.float16)), Tensor(np.array([2.0], np.float16))],
    }),
    ('KaiserWindow', {
        'block': KaiserWindowFunc(),
        'desc_inputs': [5]
    }),
    ('LogAddExp', {
        'block': LogAddExpFunc(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float16)), Tensor(np.array([2.0], np.float16))],
        'desc_bprop': [Tensor(np.array([1.0, 2.0, 3.0], np.float16)), Tensor(np.array([2.0], np.float16))],
    }),
    ('Mv', {
        'block': MvFunc(),
        'desc_inputs': [Tensor(np.array([[3., 4.], [1., 6.], [1., 3.]])),
                        Tensor(np.array([1., 2.]))],
        'desc_bprop': [Tensor(np.array([[3., 4.], [1., 6.], [1., 3.]])),
                       Tensor(np.array([1., 2.]))],
    }),
    ('Addr', {
        'block': AddrFunc(),
        'desc_inputs': [Tensor(np.array([[0., 0.], [0., 0.], [0., 0.]])),
                        Tensor(np.array([1., 2., 3.])),
                        Tensor(np.array([1., 2.]))],
        'desc_bprop': [Tensor(np.array([[0., 0.], [0., 0.], [0., 0.]])),
                       Tensor(np.array([1., 2., 3.])),
                       Tensor(np.array([1., 2.]))],
    }),
    ('Outer', {
        'block': OuterFunc(),
        'desc_inputs': [Tensor(np.array([1., 2., 3.])),
                        Tensor(np.array([1., 2., 3.]))],
        'desc_bprop': [Tensor(np.array([1., 2., 3.])),
                       Tensor(np.array([1., 2., 3.]))],
        'skip': ['backward']
    }),
    ('Addmv', {
        'block': AddmvFunc(),
        'desc_inputs': [Tensor(np.array([1, 1])),
                        Tensor(np.array([[1, 2, 1], [1, 1, 1]])),
                        Tensor(np.array([1, 1, 1]))],
        'desc_bprop': [Tensor(np.array([1, 1])),
                       Tensor(np.array([[1, 2, 1], [1, 1, 1]])),
                       Tensor(np.array([1, 1, 1]))],
    }),
    ('Exp2', {
        'block': Exp2Func(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float16))],
    }),
    ('Trapz', {
        'block': TrapzFunc(),
        'desc_inputs': [Tensor(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.float32))],
        'desc_bprop': [Tensor(np.array([2, 8, 14], np.float32))],
    }),
    ('DenseToDenseSetOperation', {
        'block': DenseToDenseSetOperation(set_operation="a-b", validate_indices=True),
        'desc_inputs': [Tensor(np.array([[1, 2, 4], [3, 4, 5]], np.int32)),
                        Tensor(np.array([[3, 2, 5], [1, 4, 7]], np.int32))],
        'skip': ['backward']
    }),
    ('DenseToSparseSetOperation', {
        'block': DenseToSparseSetOperation(set_operation="a-b", validate_indices=True),
        'desc_inputs': [Tensor(np.array([[1, 2, 4], [3, 4, 5]], np.int32)),
                        Tensor(np.array([[0, 1], [1, 0]], np.int64)),
                        Tensor(np.array([1, 6], np.int32)),
                        Tensor(np.array([2, 3], np.int64))],
        'skip': ['backward']
    }),
    ('Deg2rad', {
        'block': Deg2radNet(),
        'desc_inputs': [Tensor(np.array([[90.0, -90.0], [180.0, -180.0], [270.0, -270.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[90.0, -90.0], [180.0, -180.0], [270.0, -270.0]], np.float32))],
    }),
    ('IsReal', {
        'block': IsRealFunc(),
        'desc_inputs': [Tensor([1, 1+1j, 2+0j])],
    }),
    ('Rad2deg', {
        'block': Rad2degNet(),
        'desc_inputs': [Tensor(np.array([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]], np.float32))],
        'desc_bprop': [Tensor(np.array([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]], np.float32))],
    }),
    ('Baddbmm', {
        'block': BaddbmmNet(),
        'desc_inputs': [Tensor(np.ones([1, 3, 3]).astype(np.float32)),
                        Tensor(np.ones([1, 3, 4]).astype(np.float32)),
                        Tensor(np.ones([1, 4, 3]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('Log2', {
        'block': Log2Net(),
        'desc_inputs': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))]}),
    ('Log10', {
        'block': Log10Net(),
        'desc_inputs': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))],
        'desc_bprop': [Tensor(np.array([[1.0, 2.0, 4.0]], np.float32))]}),
    ('Frac', {
        'block': FracNet(),
        'desc_inputs': [Tensor(np.array([2, 4.2, -2.5], np.float32))],
        'desc_bprop': [Tensor(np.array([2, 4.2, -2.5], np.float32))],
    }),
    ('Kron', {
        'block': KronFunc(),
        'desc_inputs': [Tensor(np.array([[0, 1, 2], [3, 4, 5]]).astype(np.float32)),
                        Tensor(np.array([[-1, -2, -3], [-4, -6, -8]]).astype(np.float32))],
        'skip': ['backward']}),
    ('Polar', {
        'block': PolarFunc(),
        'desc_inputs': [Tensor(np.array([[0, 1, 2], [3, 4, 5]]).astype(np.float32)),
                        Tensor(np.array([[-1, -2, -3], [-4, -6, -8]]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([1+2j, 2+3j, 3+4j], np.complex64))],
    }),
    ('Rot90', {
        'block': Rot90Func(),
        'desc_inputs': [Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))],
        'skip': ['backward']}),
    ('Remainder', {
        'block': RemainderNet(),
        'desc_inputs': [Tensor(np.array([-1.0, 5.0, 6.0]), ms.float32), Tensor(np.array([3.0, 2.0, 3.0]), ms.float32)],
        'skip': ['backward']}),
    ('NanSum', {
        'block': NanSumFunc(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float32)), int(0)],
        'desc_bprop': [Tensor(np.array([1.0, 2.0, 3.0], np.float32))]}),
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
    ('Ldexp_Error1', {
        'block': (LdexpFunc(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.array([[1., 1.], [1., 2.], [1., 3.]]), dtype=ms.float32),
                        Tensor(np.array([1, 2, 3]), dtype=ms.int32)],
        'skip': ['backward']}),
    ('Ldexp_Error2', {
        'block': (LdexpFunc(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randn(5, 2), dtype=mstype.float16),
                        Tensor(np.random.randn(5), dtype=mstype.float16)],
        'skip': ['backward']}),
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
    ('Hypot', {
        'block': HypotFunc(),
        'desc_inputs': [Tensor(np.array([3, 5, 7]).astype(np.float32)),
                        Tensor(np.array([4, 12, 24]).astype(np.float32))]}),
    ('Heaviside', {
        'block': HeavisideFunc(),
        'desc_inputs': [Tensor(np.array([4, 4, 12]).astype(np.float32)),
                        Tensor(np.array([4, 8, 12]).astype(np.float32))],
        'skip': ['backward']}),
    ('Trunc', {
        'block': P.Trunc(),
        'desc_inputs': [Tensor(np.array([[1.1, 2.2, -4.1]], np.float32))],
        'skip': ['backward']}),
    ('MatrixTriangularSolve', {
        'block': MatrixTriangularSolve(adjoint=False, lower=True),
        'desc_inputs': [Tensor(np.array([4, 4, 4]).astype(np.float32)),
                        Tensor(np.array([4, 4, 4]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([4, 4, 4]).astype(np.float32))]}),
    ('Gcd', {
        'block': GcdFunc(),
        'desc_inputs': [Tensor(np.array([2, 5, 8]).astype(np.int32)),
                        Tensor(np.array([4, 3, 12]).astype(np.int32))],
        'skip': ['backward']}),
    ('Fmin', {
        'block': FminFunc(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float32)),
                        Tensor(np.array([2.0, 1.0, 4.0], np.float32))],
        'desc_bprop': [Tensor(np.array([1.0, 2.0, 3.0], np.float32))]}),
    ('Zeta', {
        'block': Zeta(),
        'desc_inputs': [Tensor(np.array([1, 1, 1, 1], np.float32)),
                        Tensor([0.5, 0.5, 0.5, 0.5], mstype.float32)]}),
    ('Fmax', {
        'block': FmaxFunc(),
        'desc_inputs': [Tensor(np.array([1.0, 2.0, 3.0], np.float32)),
                        Tensor(np.array([2.0, 1.0, 4.0], np.float32))],
        'desc_bprop': [Tensor(np.array([1.0, 2.0, 3.0], np.float32))]}),
    ('Lcm', {
        'block': LcmFunc(),
        'desc_inputs': [Tensor(np.array([2, 5, 8]).astype(np.int32)),
                        Tensor(np.array([4, 3, 12]).astype(np.int32))],
        'skip': ['backward']}),
    ('Igamma', {
        'block': Igamma(),
        'desc_inputs': [Tensor(np.array([1.1, 2.2, -4.1], np.float32)),
                        Tensor(np.array([0.2, 1.2, 2.1], np.float32))],
        'desc_bprop': [Tensor(np.array([2, 3], np.float32)),
                       Tensor(np.array([2, 3], np.float32))],
        'skip': ['backward']}),
    ('Igammac', {
        'block': Igammac(),
        'desc_inputs': [Tensor(np.array([1.1, 2.2, -4.1], np.float32)),
                        Tensor(np.array([0.2, 1.2, 2.1], np.float32))],
        'desc_bprop': [Tensor(np.array([2, 3], np.float32)),
                       Tensor(np.array([2, 3], np.float32))],
        'skip': ['backward']}),
    ('IgammaGradA', {
        'block': IgammaGradA(),
        'desc_inputs': [Tensor(np.array([1.1, 2.2, 8.1, 2.1], np.float32)),
                        Tensor(np.array([0.2, 1.2, 2.1, 3.4], np.float32))],
        'skip': ['backward']}),
    ('Outer_Error', {
        'block': (OuterFunc(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.array([[1., 1.], [1., 2.], [1., 3.]]), dtype=ms.float32),
                        Tensor(np.array([1, 2, 3]), dtype=ms.int32)],
        'skip': ['backward']}),
    ('Deg2rad_1_Error', {
        'block': (lambda x: Deg2radNet(), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('Deg2rad_2_Error', {
        'block': (lambda x: Deg2radNet(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.array([[90, -90], [180, -180], [270, -270]], np.int32))]}),
    ('Rad2deg_1_Error', {
        'block': (lambda x: Rad2degNet(), {'exception': TypeError}),
        'desc_inputs': [0]}),
    ('Rad2deg_2_Error', {
        'block': (lambda x: Rad2degNet(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.array([[3, -3], [6, -6], [1, -1]], np.int32))]}),
    ('Baddbmm_Error', {
        'block': (BaddbmmNet(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.ones([1, 3, 3]).astype(np.float32)),
                        Tensor(np.ones([1, 3, 4]).astype(np.float32)),
                        [1, 2]],
        'skip': ['backward']}),
    ('Log2_Error_2', {
        'block': (Log2Net(), {'exception;': TypeError}),
        'desc_inputs': [Tensor(np.array([[1, 2, 4]], np.int32))],
        'skip': ['backward']}),
    ('Log2_Error_1', {
        'block': (Log2Net(), {'exception;': TypeError}),
        'desc_inputs': [[1]],
        'skip': ['backward']}),
    ('Log10_Error_1', {
        'block': (Log10Net(), {'exception': TypeError}),
        'desc_inputs': [[1]],
        'skip': ['backward']}),
    ('Log10_Error_2', {
        'block': (Log10Net(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.array([[1, 2, 4]], np.int32))],
        'skip': ['backward']}),
    ('Kron_1_Error', {
        'block': (KronFunc(), {'exception': TypeError}),
        'desc_inputs': [[-5, -3, -1, 1, 3, 5], [-5, -3, -1, 1, 3, 5]],
        'skip': ['backward']}),
    ('Kron_2_Error', {
        'block': (KronFunc(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.random.randn(2, 5), dtype=mstype.float64),
                        Tensor(np.random.randn(2, 5), dtype=mstype.float64)],
        'skip': ['backward']}),
    ('Kron_3_Error', {
        'block': (KronFunc(), {'exception': RuntimeError}),
        'desc_inputs': [Tensor(np.random.randn(2, 2, 3, 2, 3), dtype=mstype.float16),
                        Tensor(np.random.randn(3, 2), dtype=mstype.float16)],
        'skip': ['backward']}),
    ('Rot90_1_Error', {
        'block': (Rot90Func(), {'exception': TypeError}),
        'desc_inputs': [[-5, -3, -1, 1, 3, 5]],
        'skip': ['backward']}),
    ('Rot90_2_Error', {
        'block': (Rot90Func(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.array([0]), dtype=mstype.float16)],
        'skip': ['backward']}),
    ('Orgqr', {
        'block': OrgqrFunc(),
        'desc_inputs': [Tensor(np.array([[-114.6, 10.9, 1.1],
                                         [-0.304, 38.07, 69.38],
                                         [-0.45, -0.17, 62.0]]).astype(np.float32)),
                        Tensor(np.array([1.55, 1.94, 0.0]).astype(np.float32))
                        ],
        'skip': ['backward']}),
    ('Remainder_Error_1', {
        'block': (RemainderNet(), {'exception': TypeError}),
        'desc_inputs': [Tensor(np.array([-4.0, 5.0, 6.0]), ms.float32),
                        [3.0, 2.0, 3.0]],
        'skip': ['backward']}),
    ('Remainder_Error_2', {
        'block': (RemainderNet(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.array([-4.0, 5.0, 6.0]), ms.float32),
                        Tensor(np.array([3.0, 2.0]), ms.float32)],
        'skip': ['backward']}),
    ('BatchMatMul', {
        'block': BatchMatMul(),
        'desc_inputs': [Tensor(np.ones([2, 4, 2, 2]).astype(np.float32)),
                        Tensor(np.ones([2, 4, 2, 2]).astype(np.float32))],
        'desc_bprop': [Tensor(np.ones([2, 4, 2, 2]).astype(np.float32))]}),
]


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return raise_set
