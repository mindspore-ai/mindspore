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
""" test ops """
import functools
import pytest

import numpy as np

import mindspore.nn as nn
import mindspore.ops.composite as C
from mindspore import Tensor
from mindspore import ops, Parameter, context
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _quant_ops as Q
from mindspore.ops.operations import nn_ops as nps
from mindspore.nn.layer import normalization
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import (pipeline_for_compile_forward_ge_graph_for_case_by_case_config,
            pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
from ....mindspore_test_framework.pipeline.gradient.compile_gradient \
    import pipeline_for_compile_grad_ge_graph_for_case_by_case_config
from ....ops_common import convert

grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


class InputBackward(nn.Cell):
    def __init__(self, network):
        super(InputBackward, self).__init__()
        self.network = network
        self.network.set_train()
        self.grad = grad_all_with_sens

    def construct(self, x1, x2, x3, sens):
        return self.grad(self.network)(x1, x2, x3, sens)


class NetForTupleInput(nn.Cell):
    def __init__(self, op):
        super(NetForTupleInput, self).__init__()
        self.op = op

    def construct(self, x1, x2):
        return self.op((x1, x2))


class StridedSlicessdNet(nn.Cell):
    def __init__(self):
        super(StridedSlicessdNet, self).__init__()
        self.rank = P.Rank()

    def construct(self, x1):
        return P.StridedSlice(1, 1, 0, self.rank(x1), 0)(x1, (0, 0), (0, 0), (1, 1))


class NetForConcat(nn.Cell):
    def __init__(self):
        super(NetForConcat, self).__init__()
        self.concat = P.Concat()

    def construct(self, x1):
        return self.concat((x1, x1))


class NetForConcat1(nn.Cell):
    def __init__(self):
        super(NetForConcat1, self).__init__()
        self.concat = P.Concat()

    def construct(self, x1, x2):
        return self.concat((x1, x2))


class NetForConcat2(nn.Cell):
    def __init__(self):
        super(NetForConcat2, self).__init__()
        self.concat = P.Concat(axis=2)

    def construct(self, x1, x2):
        return self.concat((x1, x2))


class NetForConcat3(nn.Cell):
    def __init__(self):
        super(NetForConcat3, self).__init__()
        self.concat = P.Concat(axis=0)

    def construct(self, x1, x2, x3):
        return self.concat((x1, x2, x3))


class NetForConcat4(nn.Cell):
    def __init__(self):
        super(NetForConcat4, self).__init__()
        self.concat = P.Concat(axis=-1)

    def construct(self, x1, x2, x3):
        return self.concat((x1, x2, x3))


class NetForStackInput(nn.Cell):
    def __init__(self, op):
        super(NetForStackInput, self).__init__()
        self.op = op
        self.mul = P.Mul()

    def construct(self, *args):
        t = ()
        for element in args:
            t = t + (self.mul(element, element),)
        return self.op(t)


class NetForUnpackInput(nn.Cell):
    def __init__(self, op):
        super(NetForUnpackInput, self).__init__()
        self.op = op
        self.mul = P.Mul()

    def construct(self, x1):
        return self.op((self.mul(x1, x1)))


class NetForFlatten(nn.Cell):
    def __init__(self):
        super(NetForFlatten, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x, y):
        return self.flatten(x) + y


class NetForFlatten0D(nn.Cell):
    def __init__(self):
        super(NetForFlatten0D, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        return self.flatten(x)


class NetForFlattenComposed(nn.Cell):
    # make flatten op together with other ops for testing flatten grad
    def __init__(self):
        super(NetForFlattenComposed, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x, y):
        return self.flatten(x + x) + y


class ArgmaxNet(nn.Cell):
    def __init__(self):
        super(ArgmaxNet, self).__init__()
        self.argmax = P.Argmax(axis=1)

    def construct(self, input_):
        return self.argmax(input_)


class ArgminNet(nn.Cell):
    def __init__(self):
        super(ArgminNet, self).__init__()
        self.argmin = P.Argmin(axis=1)

    def construct(self, input_):
        return self.argmin(input_)


class CumSumNet(nn.Cell):
    def __init__(self):
        super(CumSumNet, self).__init__()
        self.cumsum = P.CumSum()
        self.axis = 1

    def construct(self, input_):
        return self.cumsum(input_, self.axis)


class SummaryNet(nn.Cell):
    def __init__(self):
        super(SummaryNet, self).__init__()
        self.s = P.ScalarSummary()
        self.add = P.Add()

    def construct(self, x, y):
        self.s("x1", x)
        return self.add(x, y)


class HistogramSummaryNet(nn.Cell):
    def __init__(self):
        super(HistogramSummaryNet, self).__init__()
        self.summary = P.HistogramSummary()
        self.add = P.Add()

    def construct(self, x, y):
        out = self.add(x, y)
        string_in = "out"
        self.summary(string_in, out)
        return out


class Moments(nn.Cell):
    """Moments net definition"""

    def __init__(self, axis=None, keep_dims=None):
        super(Moments, self).__init__()
        self.moments = nn.Moments(axis=axis, keep_dims=keep_dims)

    def construct(self, input_x):
        mean, variance = self.moments(input_x)
        return mean, variance


class BatchNorm3d(nn.Cell):
    """BatchNorm3d net definition"""

    def __init__(self, num_features):
        super(BatchNorm3d, self).__init__()
        self.bn3d = normalization.BatchNorm3d(num_features=num_features)

    def construct(self, input_x):
        bn3d_out = self.bn3d(input_x)
        return bn3d_out


class NLLLoss(nn.Cell):
    """NLLLoss net definition"""

    def __init__(self, reduction):
        super(NLLLoss, self).__init__()
        self.nll_loss = P.NLLLoss(reduction=reduction)

    def construct(self, input_x, target, weight):
        loss = self.nll_loss(input_x, target, weight)
        return loss


class ClipByNorm(nn.Cell):
    """ClipByNorm net definition"""

    def __init__(self, axis=None):
        super(ClipByNorm, self).__init__()
        self.clip_by_norm = nn.ClipByNorm(axis=axis)

    def construct(self, input_x, max_norm):
        norm = self.clip_by_norm(input_x, max_norm)
        return norm


class ClipByGlobalNorm(nn.Cell):
    """ClipByGlobalNorm net definition"""

    def __init__(self, x, clip_norm=1.0, use_norm=None):
        super(ClipByGlobalNorm, self).__init__()
        self.x = x
        self.clip_norm = clip_norm
        self.use_norm = use_norm

    def construct(self):
        norm = C.clip_by_global_norm(self.x, self.clip_norm, self.use_norm)
        return norm


class Embedding(nn.Cell):
    """Embedding net definition"""

    def __init__(self, vocab_size, embedding_size, padding_idx=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size=vocab_size, embedding_size=embedding_size,
                                      padding_idx=padding_idx)

    def construct(self, index):
        res = self.embedding(index)
        return res


class EmbeddingLookup(nn.Cell):
    """EmbeddingLookup net definition"""

    def __init__(self, vocab_size, embedding_size, max_norm=None):
        super(EmbeddingLookup, self).__init__()
        self.embedding_lookup = nn.EmbeddingLookup(vocab_size=vocab_size, embedding_size=embedding_size,
                                                   max_norm=max_norm)

    def construct(self, index):
        res = self.embedding_lookup(index)
        return res


class CountNonZero(nn.Cell):
    """CountNonZero net definition"""

    def __init__(self, axis, keep_dims, dtype):
        super(CountNonZero, self).__init__()
        self.axis = axis
        self.keep_dims = keep_dims
        self.dtype = dtype

    def construct(self, input_x):
        nonzero_num = C.count_nonzero(input_x, self.axis, self.keep_dims, self.dtype)
        return nonzero_num


class Mish(nn.Cell):
    """Mish net definition"""

    def __init__(self):
        super(Mish, self).__init__()
        self.mish = P.Mish()

    def construct(self, input_x):
        out = self.mish(input_x)
        return out


class SeLU(nn.Cell):
    """Selu net definition"""

    def __init__(self):
        super(SeLU, self).__init__()
        self.selu = P.SeLU()

    def construct(self, input_x):
        out = self.selu(input_x)
        return out


class MulNoNan(nn.Cell):
    """MulNoNan net definition"""

    def __init__(self):
        super(MulNoNan, self).__init__()
        self.mul_no_nan = P.MulNoNan()

    def construct(self, input_x, input_y):
        out = self.mul_no_nan(input_x, input_y)
        return out


class ScatterUpdate(nn.Cell):
    """ScatterUpdate net definition"""

    def __init__(self, ref_shape, dtype=np.float32, use_locking=False):
        super(ScatterUpdate, self).__init__()
        self.scatter_update = P.ScatterUpdate(use_locking)
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_update(self.ref, indices, updates)
        return out


class ScatterMax(nn.Cell):
    """ScatterMax net definition"""

    def __init__(self, dtype=np.float32, use_locking=False):
        super(ScatterMax, self).__init__()
        self.scatter_max = P.ScatterMax(use_locking)
        self.ref = Parameter(Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_max(self.ref, indices, updates)
        return out


class ScatterMin(nn.Cell):
    """ScatterMin net definition"""

    def __init__(self, dtype=np.float32, use_locking=False):
        super(ScatterMin, self).__init__()
        self.scatter_min = P.ScatterMin(use_locking)
        self.ref = Parameter(Tensor(np.array([[-1.0, 2.0, 3.0], [-4.0, 1.0, 6.0]], dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_min(self.ref, indices, updates)
        return out


class ScatterAdd(nn.Cell):
    """ScatterAdd net definition"""

    def __init__(self, ref_shape, dtype=np.float32, use_locking=False):
        super(ScatterAdd, self).__init__()
        self.scatter_add = P.ScatterAdd(use_locking)
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_add(self.ref, indices, updates)
        return out


class ScatterNonAliasingAdd(nn.Cell):
    """ScatterNonAliasingAdd net definition"""

    def __init__(self, ref_shape, dtype=np.float32):
        super(ScatterNonAliasingAdd, self).__init__()
        self.scatter_no_aliasing_add = P.ScatterNonAliasingAdd()
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_no_aliasing_add(self.ref, indices, updates)
        return out


class ScatterNdSub(nn.Cell):
    """ScatterNdSub net definition"""

    def __init__(self, ref_shape, dtype=np.float32):
        super(ScatterNdSub, self).__init__()
        self.scatter_nd_sub = P.ScatterNdSub()
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_nd_sub(self.ref, indices, updates)
        return out


class ScatterNdAdd(nn.Cell):
    """ScatterNdAdd net definition"""

    def __init__(self, ref_shape, dtype=np.float32):
        super(ScatterNdAdd, self).__init__()
        self.scatter_nd_add = P.ScatterNdAdd()
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_nd_add(self.ref, indices, updates)
        return out


class ScatterSub(nn.Cell):
    """ScatterSub net definition"""

    def __init__(self, ref_shape, dtype=np.float32, use_locking=False):
        super(ScatterSub, self).__init__()
        self.scatter_sub = P.ScatterSub(use_locking)
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_sub(self.ref, indices, updates)
        return out


class ScatterMul(nn.Cell):
    """ScatterMul net definition"""

    def __init__(self, ref_shape, dtype=np.float32, use_locking=False):
        super(ScatterMul, self).__init__()
        self.scatter_mul = P.ScatterMul(use_locking)
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype)), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_mul(self.ref, indices, updates)
        return out


class ScatterDiv(nn.Cell):
    """ScatterDiv net definition"""

    def __init__(self, ref_shape, dtype=np.float32, use_locking=False):
        super(ScatterDiv, self).__init__()
        self.scatter_div = P.ScatterDiv(use_locking)
        self.ref = Parameter(Tensor(np.ones(ref_shape, dtype) * 10), name="ref")

    def construct(self, indices, updates):
        out = self.scatter_div(self.ref, indices, updates)
        return out


class Conv3D(nn.Cell):
    """Conv3D net definition"""

    def __init__(self, out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, data_format):
        super(Conv3D, self).__init__()
        self.conv = nps.Conv3D(out_channel=out_channel, kernel_size=kernel_size, mode=mode, pad_mode=pad_mode,
                               pad=pad, stride=stride, dilation=dilation, group=group, data_format=data_format)

    def construct(self, x, w):
        out = self.conv(x, w)
        return out


class Conv3DBackpropInput(nn.Cell):
    """Conv3DBackpropInput net definition"""

    def __init__(self, input_shape, out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group,
                 data_format):
        super(Conv3DBackpropInput, self).__init__()
        self.conv = nps.Conv3DBackpropInput(out_channel=out_channel, kernel_size=kernel_size, mode=mode,
                                            pad_mode=pad_mode, pad=pad, stride=stride, dilation=dilation,
                                            group=group, data_format=data_format)
        self.x_size = input_shape

    def construct(self, w, doutput):
        ms_out = self.conv(w, doutput, self.x_size)
        return ms_out


class Conv3DBackpropFilter(nn.Cell):
    """Conv3DBackpropFilter net definition"""

    def __init__(self, w_shape, out_channel, kernel_size, mode, pad_mode, pad, stride, dilation, group, data_format):
        super(Conv3DBackpropFilter, self).__init__()
        self.conv = G.Conv3DBackpropFilter(out_channel=out_channel, kernel_size=kernel_size, mode=mode,
                                           pad_mode=pad_mode, pad=pad, stride=stride, dilation=dilation,
                                           group=group, data_format=data_format)
        self.w_size = w_shape

    def construct(self, x, doutput):
        ms_out = self.conv(x, doutput, self.w_size)
        return ms_out


class Conv3DTranspose(nn.Cell):
    """Conv3DTranspose net definition"""

    def __init__(self, in_channel, out_channel, kernel_size, mode, pad, stride, dilation, group, data_format):
        super(Conv3DTranspose, self).__init__()
        self.conv = nps.Conv3DTranspose(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                        mode=mode, pad=pad, stride=stride, dilation=dilation, group=group,
                                        data_format=data_format)

    def construct(self, x, w):
        ms_out = self.conv(x, w)
        return ms_out


class ApplyFtrlNet(nn.Cell):
    def __init__(self):
        super(ApplyFtrlNet, self).__init__()
        self.apply_ftrl = P.ApplyFtrl()
        self.lr = 0.001
        self.l1 = 0.0
        self.l2 = 0.0
        self.lr_power = -0.5
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="linear")

    def construct(self, grad):
        out = self.apply_ftrl(self.var, self.accum, self.linear, grad, self.lr, self.l1, self.l2, self.lr_power)
        return out


class SparseApplyFtrlNet(nn.Cell):
    def __init__(self):
        super(SparseApplyFtrlNet, self).__init__()
        self.sparse_apply_ftrl = P.SparseApplyFtrl(lr=0.001, l1=0.0, l2=0.0, lr_power=-0.5)
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl(self.var, self.accum, self.linear, grad, indices)
        return out


class SparseApplyFtrlV2Net(nn.Cell):
    def __init__(self):
        super(SparseApplyFtrlV2Net, self).__init__()
        self.sparse_apply_ftrl_v2 = P.SparseApplyFtrlV2(lr=0.001, l1=0.0, l2=0.0, l2_shrinkage=0.0, lr_power=-0.5)
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.linear = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="linear")

    def construct(self, grad, indices):
        out = self.sparse_apply_ftrl_v2(self.var, self.accum, self.linear, grad, indices)
        return out


class SparseApplyProximalAdagradNet(nn.Cell):
    def __init__(self):
        super(SparseApplyProximalAdagradNet, self).__init__()
        self.sparse_apply_proximal_adagrad = P.SparseApplyProximalAdagrad()
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.lr = 0.01
        self.l1 = 0.0
        self.l2 = 0.0

    def construct(self, grad, indices):
        out = self.sparse_apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2, grad, indices)
        return out


class ApplyProximalAdagradNet(nn.Cell):
    def __init__(self):
        super(ApplyProximalAdagradNet, self).__init__()
        self.apply_proximal_adagrad = P.ApplyProximalAdagrad()
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.lr = 0.01
        self.l1 = 0.0
        self.l2 = 0.0

    def construct(self, grad):
        out = self.apply_proximal_adagrad(self.var, self.accum, self.lr, self.l1, self.l2, grad)
        return out


class ApplyAdaMaxNet(nn.Cell):
    def __init__(self):
        super(ApplyAdaMaxNet, self).__init__()
        self.apply_ada_max = P.ApplyAdaMax()
        self.beta1_power = 0.9
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-10
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="m")
        self.v = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="v")

    def construct(self, grad):
        out = self.apply_ada_max(self.var, self.m, self.v, self.beta1_power, self.lr,
                                 self.beta1, self.beta2, self.epsilon, grad)
        return out


class ApplyAdadeltaNet(nn.Cell):
    def __init__(self):
        super(ApplyAdadeltaNet, self).__init__()
        self.apply_adadelta = P.ApplyAdadelta()
        self.lr = 0.001
        self.rho = 0.0
        self.epsilon = 1e-6
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")
        self.accum_update = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum_update")

    def construct(self, grad):
        out = self.apply_adadelta(self.var, self.accum, self.accum_update, self.lr, self.rho, self.epsilon, grad)
        return out


class ApplyAdagradNet(nn.Cell):
    def __init__(self):
        super(ApplyAdagradNet, self).__init__()
        self.apply_adagrad = P.ApplyAdagrad()
        self.lr = 0.001
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")

    def construct(self, grad):
        out = self.apply_adagrad(self.var, self.accum, self.lr, grad)
        return out


class ApplyAdagradV2Net(nn.Cell):
    def __init__(self):
        super(ApplyAdagradV2Net, self).__init__()
        self.apply_adagrad_v2 = P.ApplyAdagradV2(epsilon=1e-6)
        self.lr = 0.001
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")

    def construct(self, grad):
        out = self.apply_adagrad_v2(self.var, self.accum, self.lr, grad)
        return out


class ApplyAddSignNet(nn.Cell):
    def __init__(self):
        super(ApplyAddSignNet, self).__init__()
        self.apply_add_sign = P.ApplyAddSign()
        self.lr = 0.001
        self.alpha = 1.0
        self.sign_decay = 0.99
        self.beta = 0.99
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="m")

    def construct(self, grad):
        out = self.apply_add_sign(self.var, self.m, self.lr, self.alpha, self.sign_decay, self.beta, grad)
        return out


class ApplyPowerSignNet(nn.Cell):
    def __init__(self):
        super(ApplyPowerSignNet, self).__init__()
        self.apply_power_sign = P.ApplyPowerSign()
        self.lr = 0.001
        self.logbase = np.e
        self.sign_decay = 0.99
        self.beta = 0.99
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="m")

    def construct(self, grad):
        out = self.apply_power_sign(self.var, self.m, self.lr, self.logbase, self.sign_decay, self.beta, grad)
        return out


class ApplyGradientDescentNet(nn.Cell):
    def __init__(self):
        super(ApplyGradientDescentNet, self).__init__()
        self.apply_gradient_descent = P.ApplyGradientDescent()
        self.alpha = 0.001
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")

    def construct(self, delta):
        out = self.apply_gradient_descent(self.var, self.alpha, delta)
        return out


class ApplyProximalGradientDescentNet(nn.Cell):
    def __init__(self):
        super(ApplyProximalGradientDescentNet, self).__init__()
        self.apply_proximal_gradient_descent = P.ApplyProximalGradientDescent()
        self.alpha = 0.001
        self.l1 = 0.0
        self.l2 = 0.0
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")

    def construct(self, delta):
        out = self.apply_proximal_gradient_descent(self.var, self.alpha, self.l1, self.l2, delta)
        return out


class SparseApplyAdagradNet(nn.Cell):
    def __init__(self):
        super(SparseApplyAdagradNet, self).__init__()
        self.sparse_apply_adagrad = P.SparseApplyAdagrad(lr=0.01)
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")

    def construct(self, grad, indices):
        out = self.sparse_apply_adagrad(self.var, self.accum, grad, indices)
        return out


class SparseApplyAdagradV2Net(nn.Cell):
    def __init__(self):
        super(SparseApplyAdagradV2Net, self).__init__()
        self.sparse_apply_adagrad_v2 = P.SparseApplyAdagradV2(lr=0.01, epsilon=0.001)
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="accum")

    def construct(self, grad, indices):
        out = self.sparse_apply_adagrad_v2(self.var, self.accum, grad, indices)
        return out


class ApplyRMSNet(nn.Cell):
    def __init__(self):
        super(ApplyRMSNet, self).__init__()
        self.apply_rms = P.ApplyRMSProp()
        self.lr = 0.001
        self.rho = 0.0
        self.momentum = 0.0
        self.epsilon = 1e-10
        self.var = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="var")
        self.ms = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="ms")
        self.moment = Parameter(Tensor(np.random.rand(3, 3).astype(np.float32)), name="moment")

    def construct(self, grad):
        out = self.apply_rms(self.var, self.ms, self.moment, self.lr, grad, self.rho, self.momentum, self.epsilon)
        return out


class InplaceAddNet(nn.Cell):
    def __init__(self):
        super(InplaceAddNet, self).__init__()
        self.inplace_add = P.InplaceAdd(indices=(0, 1))

    def construct(self, x, v):
        out = self.inplace_add(x, v)
        return out


class InplaceSubNet(nn.Cell):
    def __init__(self):
        super(InplaceSubNet, self).__init__()
        self.inplace_sub = P.InplaceSub(indices=(0, 1))

    def construct(self, x, v):
        out = self.inplace_sub(x, v)
        return out


class NormalNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(NormalNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean, stddev):
        out = C.normal(self.shape, mean, stddev, self.seed)
        return out


class LaplaceNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(LaplaceNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean, lambda_param):
        out = C.laplace(self.shape, mean, lambda_param, self.seed)
        return out


class GammaNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(GammaNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, alpha, beta):
        out = C.gamma(self.shape, alpha, beta, self.seed)
        return out


class PoissonNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(PoissonNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, mean):
        out = C.poisson(self.shape, mean, self.seed)
        return out


class UniformNet(nn.Cell):
    def __init__(self, shape=None, seed=0):
        super(UniformNet, self).__init__()
        self.shape = shape
        self.seed = seed

    def construct(self, a, b):
        out = C.uniform(self.shape, a, b, self.seed)
        return out


class CTCGreedyDecoderNet(nn.Cell):
    def __init__(self):
        super(CTCGreedyDecoderNet, self).__init__()
        self.ctc_greedy_decoder = P.CTCGreedyDecoder()
        self.assert_op = P.Assert(300)

    def construct(self, inputs, sequence_length):
        out = self.ctc_greedy_decoder(inputs, sequence_length)
        self.assert_op(True, (out[0], out[1], out[2], out[3]))
        return out[2]


class StridedSliceNet(nn.Cell):
    def __init__(self):
        super(StridedSliceNet, self).__init__()
        self.begins = (1, 2, 3, 2, 1)
        self.ends = (5, 6, 7, 8, 9)
        self.strides = (1, 2, 3, 2, 1)
        self.strided_slice_0 = P.StridedSlice(begin_mask=3, end_mask=5, ellipsis_mask=4,
                                              shrink_axis_mask=2, new_axis_mask=8)
        self.strided_slice_1 = P.StridedSlice(begin_mask=5, end_mask=2, ellipsis_mask=2,
                                              shrink_axis_mask=6, new_axis_mask=10)
        self.strided_slice_2 = P.StridedSlice(begin_mask=3, end_mask=3, ellipsis_mask=4,
                                              shrink_axis_mask=5, new_axis_mask=13)
        self.strided_slice_3 = P.StridedSlice(begin_mask=0, end_mask=0, ellipsis_mask=4,
                                              shrink_axis_mask=12, new_axis_mask=15)
        self.const_0 = Tensor(np.ones([6, 8, 9, 1, 8], np.float32))
        self.const_1 = Tensor(np.ones([5, 7, 8, 1, 8], np.float32))
        self.const_2 = Tensor(np.ones([1, 3, 7, 8, 9, 1, 8], np.float32))
        self.const_3 = Tensor(np.ones([1, 1, 6, 7, 8, 9, 1, 8], np.float32))

    def construct(self, x):
        out_0 = self.strided_slice_0(x, self.begins, self.ends, self.strides) + self.const_0
        out_1 = self.strided_slice_1(x, self.begins, self.ends, self.strides) + self.const_1
        out_2 = self.strided_slice_2(x, self.begins, self.ends, self.strides) + self.const_2
        out_3 = self.strided_slice_3(x, self.begins, self.ends, self.strides) + self.const_3
        return out_0, out_1, out_2, out_3

@pytest.mark.skip(reason='0 in shape is not support')
def test_strided_slice_const():
    class StridedSLiceConstNet(nn.Cell):
        """StridedSLiceConstNet net definition"""

        def __init__(self):
            super(StridedSLiceConstNet, self).__init__()
            self.begins = (0, 2, -5, 2, 1)
            self.ends = (0, 6, 9, 8, 9)
            self.strides = (1, 2, 1, 2, 1)
            self.strided_slice = P.StridedSlice(begin_mask=2,
                                                end_mask=6,
                                                ellipsis_mask=4,
                                                shrink_axis_mask=6,
                                                new_axis_mask=18)

        def construct(self, x):
            out = self.strided_slice(x, self.begins, self.ends, self.strides)
            return out

    net = StridedSLiceConstNet()
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
    x = Tensor(np.ones([6, 7, 8, 9, 10]), mstype.float32)
    ret = net(x)
    assert ret.shape == (0, 1, 7, 8, 9, 3, 1)
    assert (ret.asnumpy() == np.array([], np.float32).reshape([0, 1, 7, 8, 9, 3, 1])).all()


class ParallelConcatNet(nn.Cell):
    def __init__(self):
        super(ParallelConcatNet, self).__init__()
        self.parallel_concat = P.ParallelConcat()

    def construct(self, x1, x2):
        return self.parallel_concat((x1, x2))


class BasicLSTMCellNet(nn.Cell):
    """ BasicLSTMCellNet definition """

    def __init__(self):
        super(BasicLSTMCellNet, self).__init__()
        self.lstm = P.BasicLSTMCell()

    def construct(self, x, h, c, w, b):
        return self.lstm(x, h, c, w, b)


class DynamicGRUV2Net(nn.Cell):
    """ DynamicGRUV2Net definition """

    def __init__(self):
        super(DynamicGRUV2Net, self).__init__()
        self.dynamic_gru = P.DynamicGRUV2()

    def construct(self, x, w_i, w_h, b_i, b_h, init_h):
        return self.dynamic_gru(x, w_i, w_h, b_i, b_h, None, init_h)


class EditDistance(nn.Cell):
    def __init__(self, hypothesis_shape, truth_shape, normalize=True):
        super(EditDistance, self).__init__()
        self.edit_distance = P.EditDistance(normalize)
        self.hypothesis_shape = hypothesis_shape
        self.truth_shape = truth_shape

    def construct(self, hypothesis_indices, hypothesis_values, truth_indices, truth_values):
        return self.edit_distance(hypothesis_indices, hypothesis_values, self.hypothesis_shape,
                                  truth_indices, truth_values, self.truth_shape)


test_case_math_ops = [
    ('BitwiseAnd', {
        'block': P.BitwiseAnd(),
        'desc_inputs': [Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mstype.int16),
                        Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mstype.int16)],
        'skip': ['backward']}),
    ('BitwiseAnd_1', {
        'block': P.BitwiseAnd(),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [-1, -2, -3]]), mstype.int16),
                        Tensor(np.array([1, 1, 1]), mstype.int16)],
        'skip': ['backward']}),
    ('BitwiseOr', {
        'block': P.BitwiseOr(),
        'desc_inputs': [Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mstype.int16),
                        Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mstype.int16)],
        'skip': ['backward']}),
    ('BitwiseOr_1', {
        'block': P.BitwiseOr(),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [-1, -2, -3]]), mstype.int16),
                        Tensor(np.array([1, 1, 1]), mstype.int16)],
        'skip': ['backward']}),
    ('BitwiseXor', {
        'block': P.BitwiseXor(),
        'desc_inputs': [Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mstype.int16),
                        Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mstype.int16)],
        'skip': ['backward']}),
    ('BitwiseXor_1', {
        'block': P.BitwiseXor(),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [-1, -2, -3]]), mstype.int16),
                        Tensor(np.array([1, 1, 1]), mstype.int16)],
        'skip': ['backward']}),
    ('Neg', {
        'block': P.Neg(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('Sub', {
        'block': P.Sub(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Add', {
        'block': P.Add(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Mul0', {
        'block': P.Mul(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Mul1', {
        'block': P.Mul(),
        'desc_inputs': [[2, 3, 1, 1], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Mul2', {
        'block': P.Mul(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 1, 1]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Mul3', {
        'block': P.Mul(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Mul4', {
        'block': P.Mul(),
        'desc_inputs': [[2, 3, 3, 5], [3, 5]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Add0', {
        'block': P.Add(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Add1', {
        'block': P.Add(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Add2', {
        'block': P.Add(),
        'desc_inputs': [[2, 3, 3, 5], [3, 5]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Add3', {
        'block': P.Add(),
        'desc_inputs': [[2, 3, 1, 1], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Add4', {
        'block': P.Add(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 1, 1]],
        'desc_bprop': [[2, 3, 3, 5]],
        'skip': ['backward']}),
    ('Minimum', {
        'block': P.Minimum(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Pow_0', {
        'block': P.Pow(),
        'desc_const': [2.0],
        'desc_inputs': [[2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Pow_1', {
        'block': P.Pow(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Exp', {
        'block': P.Exp(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Expm1', {
        'block': P.Expm1(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Erf', {
        'block': P.Erf(),
        'desc_inputs': [Tensor(np.array([-2, -1, 0, 1, 2]).astype(np.float16))],
        'desc_bprop': [Tensor(np.array([-2, -1, 0, 1, 2]).astype(np.float16))]}),
    ('Floor', {
        'block': P.Floor(),
        'desc_inputs': [[2, 512, 56, 56]],
        'desc_bprop': [[2, 512, 56, 56]],
        'skip': ['backward']}),
    ('Ceil', {
        'block': P.Ceil(),
        'desc_inputs': [[2, 512, 56, 56]],
        'desc_bprop': [[2, 512, 56, 56]],
        'skip': ['backward']}),
    ('InplaceAdd', {
        'block': InplaceAddNet(),
        'desc_inputs': [Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)),
                        Tensor(np.array([[0.5, 1], [1, 1.5]]).astype(np.float32))],
        'skip': ['backward']}),
    ('InplaceSub', {
        'block': InplaceSubNet(),
        'desc_inputs': [Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)),
                        Tensor(np.array([[0.5, 1], [1, 1.5]]).astype(np.float32))],
        'skip': ['backward']}),
    ('ACos', {
        'block': P.ACos(),
        'desc_inputs': [Tensor(np.array([2., 3.]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([2., 3.]).astype(np.float32))]}),
    ('ACosGrad', {
        'block': G.ACosGrad(),
        'desc_inputs': [[2, 3], [2, 3]],
        'skip': ['backward']}),
    ('Acosh', {
        'block': P.Acosh(),
        'desc_inputs': [Tensor(np.array([2., 3.]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([2., 3.]).astype(np.float32))]}),
    ('AcoshGrad', {
        'block': G.AcoshGrad(),
        'desc_inputs': [[2, 3], [2, 3]],
        'skip': ['backward']}),
    ('Sin', {
        'block': P.Sin(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Asin', {
        'block': P.Asin(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Asinh', {
        'block': P.Asinh(),
        'desc_inputs': [[3, 4, 5]],
        'desc_bprop': [[3, 4, 5]]}),
    ('Tan', {
        'block': P.Tan(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Reciprocal', {
        'block': P.Reciprocal(),
        'desc_inputs': [[2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Minimum_0', {
        'block': P.Minimum(),
        'desc_inputs': [[2, 3, 3, 5], [3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Maximum', {
        'block': P.Maximum(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Maximum_0', {
        'block': P.Maximum(),
        'desc_inputs': [[3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('MaximumGrad', {
        'block': G.MaximumGrad(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5], [2, 3, 3, 5]],
        'skip': ['backward']}),
    ('MinimumGrad', {
        'block': G.MinimumGrad(),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5], [2, 3, 3, 5]],
        'skip': ['backward']}),
    ('StridedSlice_00', {
        'block': P.StridedSlice(shrink_axis_mask=0),
        'desc_const': [(0, 1, 2, 1),
                       (2, 3, 3, 4),
                       (1, 1, 1, 2)],
        'desc_inputs': [[2, 3, 3, 5]],
        'desc_bprop': [[2, 2, 1, 3]],
        'skip': ['backward']}),
    ('Slice_1', {
        'block': P.Slice(),
        'desc_const': [(0, 1, 2, 1),
                       (1, 1, 1, 2)],
        'desc_inputs': [[2, 3, 3, 5]],
        'desc_bprop': [[1, 1, 1, 2]]}),
    ('StridedSliceGrad', {
        'block': G.StridedSliceGrad(),
        'desc_const': [(64, 1, 1024),
                       (0, 1, 0),
                       (64, 2, 1024),
                       (1, 1, 1)],
        'desc_inputs': [[64, 128, 1024]],
        'skip': ['backward']}),
    ('Normal', {
        'block': NormalNet((3, 2, 4), 0),
        'desc_inputs': [Tensor(0.0, mstype.float32), Tensor(1.0, mstype.float32)],
        'skip': ['backward']}),
    ('Laplace', {
        'block': LaplaceNet((3, 2, 4), 0),
        'desc_inputs': [Tensor(1.0, mstype.float32), Tensor(1.0, mstype.float32)],
        'skip': ['backward']}),
    ('Gamma', {
        'block': GammaNet((3, 2, 4), 0),
        'desc_inputs': [Tensor(1.0, mstype.float32), Tensor(1.0, mstype.float32)],
        'skip': ['backward']}),
    ('Poisson', {
        'block': PoissonNet((3, 2, 4), 0),
        'desc_inputs': [Tensor(2.0, mstype.float32)],
        'skip': ['backward']}),
    ('Uniform', {
        'block': UniformNet((3, 2, 4), 0),
        'desc_inputs': [Tensor(0.0, mstype.float32), Tensor(1.0, mstype.float32)],
        'skip': ['backward']}),
    ('RandomChoiceWithMask', {
        'block': P.RandomChoiceWithMask(256),
        'desc_inputs': [Tensor(np.random.rand(24000, 4).astype(np.bool_))],
        'desc_bprop': [[256, 4], [256, 4]],
        'skip': ['backward']}),
    ('LessEqual', {
        'block': P.LessEqual(),
        'desc_inputs': [Tensor(np.random.rand(4).astype(np.float16)),
                        Tensor(np.random.rand(4).astype(np.float16))],
        'skip': ['backward']}),
    ('Less', {
        'block': P.Less(),
        'desc_inputs': [[2, 1, 4, 5], [2, 1, 4, 5]],
        'desc_bprop': [Tensor(np.zeros((2, 1, 4, 5), np.bool_))],
        'skip': ['backward']}),
    ('RealDiv_0', {
        'block': P.RealDiv(),
        'desc_const': [Tensor(2048.0), Tensor(0.0)],
        'desc_inputs': [],
        'skip': ['backward']}),
    ('RealDiv', {
        'block': P.RealDiv(),
        'desc_inputs': [[4], Tensor(np.ones(4).astype(np.float32))],
        'desc_bprop': [[4]]}),
    ('RealDiv_1', {
        'block': P.RealDiv(),
        'desc_inputs': [[512, 1024], [512, 1024]],
        'desc_bprop': [[512, 1024]]}),
    ('FloorDiv', {
        'block': P.FloorDiv(),
        'desc_inputs': [Tensor(np.random.rand(4).astype(np.float16)),
                        Tensor(np.random.rand(4).astype(np.float16))],
        'skip': ['backward']}),
    ('FloorMod', {
        'block': P.FloorMod(),
        'desc_inputs': [[3, 4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('TruncateDiv', {
        'block': P.TruncateDiv(),
        'desc_inputs': [[3, 4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('TruncateMod', {
        'block': P.TruncateMod(),
        'desc_inputs': [[3, 4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('identity', {
        'block': ops.functional.identity,
        'desc_inputs': [[2, 2]],
        'skip': ['backward']}),
    ('MatMul_1', {
        'block': P.MatMul(transpose_a=False, transpose_b=False),
        'desc_inputs': [[1024, 160], [160, 1024]],
        'desc_bprop': [[1024, 1024]]}),
    ('MatMul_2', {
        'block': P.MatMul(transpose_a=True, transpose_b=True),
        'desc_inputs': [[160, 1024], [1024, 160]],
        'desc_bprop': [[1024, 1024]]}),
    ('Sub', {
        'block': P.Sub(),
        'desc_inputs': [[3], [3]],
        'desc_bprop': [[3]]}),
    ('TruncatedNormal', {
        'block': P.TruncatedNormal(),
        'desc_const': [(1, 2, 3)],
        'desc_inputs': [],
        'skip': ['backward'],
        'add_fake_input': True}),
    ('Select', {
        'block': P.Select(),
        'desc_inputs': [Tensor(np.array([[True, False, False], [False, True, True]])),
                        [2, 3], [2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('ClipByNorm_1', {
        'block': ClipByNorm(),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32)),
                        Tensor(np.array([0.01]).astype(np.float32))],
        'skip': ['backward']}),
    ('ClipByNorm_2', {
        'block': ClipByNorm(axis=0),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32)),
                        Tensor(np.array([0.01]).astype(np.float32))],
        'skip': ['backward']}),
    ('ClipByGlobalNorm', {
        'block': ClipByGlobalNorm(x=Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32)),
                                  clip_norm=1.0, use_norm=None),
        'desc_inputs': [],
        'skip': ['backward']}),
    ('Embedding_1', {
        'block': Embedding(vocab_size=10, embedding_size=3),
        'desc_inputs': [Tensor(np.array([0, 2, 2, 7]).astype(np.int32))],
        'skip': ['backward']}),
    ('Embedding_2', {
        'block': Embedding(vocab_size=10, embedding_size=3, padding_idx=2),
        'desc_inputs': [Tensor(np.array([0, 2, 2, 7]).astype(np.int32))],
        'skip': ['backward']}),
    ('EmbeddingLookup_1', {
        'block': EmbeddingLookup(vocab_size=10, embedding_size=3),
        'desc_inputs': [Tensor(np.array([0, 2, 2, 7]).astype(np.int32))],
        'skip': ['backward']}),
    ('EmbeddingLookup_2', {
        'block': EmbeddingLookup(vocab_size=10, embedding_size=3, max_norm=0.01),
        'desc_inputs': [Tensor(np.array([0, 2, 2, 7]).astype(np.int32))],
        'skip': ['backward']}),
    ('Moments', {
        'block': Moments(axis=(), keep_dims=False),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32))],
        'skip': ['backward']}),
    ('NLLLoss', {
        'block': NLLLoss(reduction="mean"),
        'desc_inputs': [Tensor(np.random.rand(3, 16), mstype.float32),
                        Tensor(np.random.rand(3), mstype.int32),
                        Tensor(np.random.rand(16), mstype.float32)],
        'desc_bprop': [(Tensor(np.random.rand(1), mstype.float32), Tensor(np.random.rand(1), mstype.float32))]}),
    ('BatchNorm3d', {
        'block': BatchNorm3d(num_features=3),
        'desc_inputs': [Tensor(np.random.rand(3, 3, 3, 5, 4).astype(np.float32))],
        'skip': ['backward']}),
    ('Conv3D', {
        'block': Conv3D(out_channel=32, kernel_size=(4, 3, 3), mode=1, pad_mode='valid', pad=0,
                        stride=1, dilation=1, group=1, data_format="NCDHW"),
        'desc_inputs': [Tensor(np.random.random((16, 3, 10, 32, 32)).astype(np.float16)),
                        Tensor(np.random.random((32, 3, 4, 3, 3)).astype(np.float16))],
        'skip': ['backward']}),
    ('Conv3DBackpropInput', {
        'block': Conv3DBackpropInput(input_shape=(16, 32, 13, 37, 33), out_channel=32, kernel_size=(4, 6, 2), mode=1,
                                     pad_mode='valid', pad=0, stride=1, dilation=1, group=1, data_format="NCDHW"),
        'desc_inputs': [Tensor(np.random.random((32, 32, 4, 6, 2)).astype(np.float16)),
                        Tensor(np.random.random((16, 32, 10, 32, 32)).astype(np.float16))],
        'skip': ['backward']}),
    ('Conv3DBackpropFilter', {
        'block': Conv3DBackpropFilter(w_shape=(32, 32, 4, 6, 2), out_channel=32, kernel_size=(4, 6, 2), mode=1,
                                      pad_mode='valid', pad=0, stride=1, dilation=1, group=1, data_format="NCDHW"),
        'desc_inputs': [Tensor(np.random.random((16, 32, 13, 37, 33)).astype(np.float16)),
                        Tensor(np.random.random((16, 32, 10, 32, 32)).astype(np.float16))],
        'skip': ['backward']}),
    ('Conv3DTranspose', {
        'block': Conv3DTranspose(in_channel=32, out_channel=3, kernel_size=(4, 6, 2), mode=1,
                                 pad=0, stride=1, dilation=1, group=1, data_format="NCDHW"),
        'desc_inputs': [Tensor(np.random.random((32, 3, 10, 32, 32)).astype(np.float16)),
                        Tensor(np.random.random((3, 3, 4, 6, 2)).astype(np.float16))],
        'skip': ['backward']}),
    ('CountNonZero', {
        'block': CountNonZero(axis=(), keep_dims=False, dtype=mstype.int32),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 4).astype(np.float32))],
        'skip': ['backward']}),
    ('FakeQuantWithMinMaxVars', {
        'block': Q.FakeQuantWithMinMaxVars(num_bits=8, narrow_range=False),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 5), mstype.float32),
                        Tensor(np.array([-6]), mstype.float32),
                        Tensor(np.array([6]), mstype.float32)],
        'desc_bprop': [Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)]}),
    ('FakeQuantWithMinMaxVarsPerChannel', {
        'block': Q.FakeQuantWithMinMaxVarsPerChannel(num_bits=8, narrow_range=False),
        'desc_inputs': [Tensor(np.random.rand(3, 16, 5, 4), mstype.float32),
                        Tensor(np.array([-6, -1, -2, -3]), mstype.float32),
                        Tensor(np.array([6, 1, 2, 3]), mstype.float32)],
        'desc_bprop': [Tensor(np.random.rand(3, 16, 5, 4), mstype.float32)]}),
    ('Mish', {
        'block': Mish(),
        'desc_inputs': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)],
        'desc_bprop': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)]}),
    ('SeLU', {
        'block': SeLU(),
        'desc_inputs': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)],
        'desc_bprop': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)]}),
    ('MulNoNan', {
        'block': MulNoNan(),
        'desc_inputs': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32),
                        Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)],
        'desc_bprop': [Tensor(np.random.rand(3, 6, 16, 16), mstype.float32)]}),
    ('Rank', {
        'block': P.Rank(),
        'desc_inputs': [[2, 3]],
        'skip': ['backward']}),
    ('InvertPermutation', {
        'block': P.InvertPermutation(),
        'desc_const': [(0, 3, 1, 2)],
        'desc_inputs': [],
        'skip': ['backward']}),
    ('Xdivy', {
        'block': P.Xdivy(),
        'desc_inputs': [[4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('Xlogy', {
        'block': P.Xlogy(),
        'desc_inputs': [[4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('SquaredDifference', {
        'block': P.SquaredDifference(),
        'desc_inputs': [[4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('Square', {
        'block': P.Square(),
        'desc_inputs': [[4]],
        'desc_bprop': [[4]]}),
    ('Rsqrt', {
        'block': P.Rsqrt(),
        'desc_inputs': [[4]],
        'desc_bprop': [[4]]}),
    ('Sqrt', {
        'block': P.Sqrt(),
        'desc_inputs': [[4]],
        'desc_bprop': [[4]]}),
    ('RealDiv', {
        'block': P.RealDiv(),
        'desc_inputs': [[4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('IsFinite', {
        'block': P.IsFinite(),
        'desc_inputs': [Tensor(np.random.random((3, 4, 5)).astype(np.float32))],
        'desc_bprop': [Tensor(np.random.random((3, 4, 5)).astype(np.bool))]}),
    ('Div', {
        'block': P.Div(),
        'desc_inputs': [[4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('Equal', {
        'block': P.Equal(),
        'desc_inputs': [[3, 4, 5], [4, 5]],
        'desc_bprop': [Tensor(np.zeros((3, 4, 5), np.bool_))]}),
    ('NotEqual', {
        'block': P.NotEqual(),
        'desc_inputs': [[4, 1], [2, 3, 4, 5]],
        'desc_bprop': [Tensor(np.ones((2, 3, 4, 5), np.bool_))]}),
    ('NotEqual_0', {
        'block': P.NotEqual(),
        'desc_inputs': [Tensor(np.array(1).astype(np.int32)), [2, 3, 4, 5]],
        'desc_bprop': [Tensor(np.ones((2, 3, 4, 5), np.bool_))],
        'skip': ['backward']}),
    ('ApproximateEqual', {
        'block': P.ApproximateEqual(),
        'desc_inputs': [[3, 4, 5], [3, 4, 5]],
        'desc_bprop': [Tensor(np.zeros((3, 4, 5), np.bool_))]}),
    ('Greater', {
        'block': P.Greater(),
        'desc_inputs': [[2, 3, 4, 1], [4, 5]],
        'desc_bprop': [Tensor(np.ones((2, 3, 4, 5), np.bool_))]}),
    ('GreaterEqual', {
        'block': P.GreaterEqual(),
        'desc_inputs': [[2, 3, 4, 1], [4, 5]],
        'desc_bprop': [Tensor(np.ones((2, 3, 4, 5), np.bool_))]}),
    ('LogicalNot', {
        'block': P.LogicalNot(),
        'desc_inputs': [Tensor(np.zeros((3, 4, 5), np.bool_))],
        'desc_bprop': [Tensor(np.ones((3, 4, 5), np.bool_))]}),
    ('LogicalAnd', {
        'block': P.LogicalAnd(),
        'desc_inputs': [Tensor(np.zeros((2, 3, 4), np.bool_)), Tensor(np.ones((1), np.bool_))],
        'desc_bprop': [Tensor(np.zeros((2, 3, 4), np.bool_))]}),
    ('LogicalOr', {
        'block': P.LogicalOr(),
        'desc_inputs': [Tensor(np.zeros((3, 4, 5), np.bool_)), Tensor(np.ones((3, 1, 1), np.bool_))],
        'desc_bprop': [Tensor(np.zeros((3, 4, 5), np.bool_))]}),
    ('NpuAllocFloatStatus', {
        'block': P.NPUAllocFloatStatus(),
        'desc_inputs': [],
        'add_fack_input': True,
        'fack_input_type': np.float32,
        'desc_bprop': [Tensor(np.zeros([8]).astype(np.float32))],
        'skip': ['backward']}),
    ('NpuGetFloatStatus', {
        'block': P.NPUGetFloatStatus(),
        'desc_inputs': [Tensor(np.zeros([8]).astype(np.float32))],
        'desc_bprop': [Tensor(np.zeros([8]).astype(np.float32))],
        'skip': ['backward']}),
    ('NpuClearFloatStatus', {
        'block': P.NPUClearFloatStatus(),
        'desc_inputs': [Tensor(np.zeros([8]).astype(np.float32))],
        'desc_bprop': [Tensor(np.zeros([8]).astype(np.float32))],
        'skip': ['backward']}),
    ('CheckValid', {
        'block': P.CheckValid(),
        'desc_inputs': [[20000, 4], [3]],
        'desc_bprop': [[20000]],
        'skip': ['backward']}),
    ('NMSWithMask', {
        'block': P.NMSWithMask(0.5),
        'desc_inputs': [[128, 5]],
        'desc_bprop': [[128, 5], [128], [128]],
        'skip': ['backward']}),
    ('Abs', {
        'block': P.Abs(),
        'desc_inputs': [[4]],
        'desc_bprop': [[4]]}),
    ('CumSum', {
        'block': CumSumNet(),
        'desc_inputs': [Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7],
                                        [1, 3, 7, 9]]).astype(np.float32))]}),
    ('ReduceSum_3', {
        'block': P.ReduceSum(),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[2]]}),
    ('ReduceSum_4', {
        'block': P.ReduceSum(keep_dims=True),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[1, 2]]}),
    ('ReduceSum_5', {
        'block': P.ReduceSum(keep_dims=True),
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[1, 1, 1]]}),
    ('ReduceSum_6', {
        'block': P.ReduceSum(),
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[1]]}),
    ('Sum_0', {
        'block': P.ReduceSum(),
        'desc_const': [(1,)],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[3]]}),
    ('Sum_1', {
        'block': P.ReduceSum(keep_dims=True),
        'desc_const': [(1,)],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[3, 1]]}),
    ('Sum_2', {
        'block': P.ReduceSum(),
        'desc_const': [(0, 1)],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[1]]}),
    ('Sum_3', {
        'block': P.ReduceSum(),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[2]]}),
    ('Sum_4', {
        'block': P.ReduceSum(keep_dims=True),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[1, 2]]}),
    ('Sum_5', {
        'block': P.ReduceSum(keep_dims=True),
        'desc_const': [()],
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[1, 1, 1]]}),
    ('Sum_6', {
        'block': P.ReduceSum(),
        'desc_const': [()],
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[1]]}),
    ('Sign', {
        'block': P.Sign(),
        'desc_inputs': [[3]],
        'desc_bprop': [[3]]}),
    ('Round', {
        'block': P.Round(),
        'desc_inputs': [[3]],
        'desc_bprop': [[3]]}),
    ('Atan2', {
        'block': P.Atan2(),
        'desc_inputs': [Tensor(np.array([0, 1]).astype(np.float32)),
                        Tensor(np.array([1, 1]).astype(np.float32))],
        'desc_bprop': [[2]]}),
    ('SquareSumAll', {
        'block': P.SquareSumAll(),
        'desc_inputs': [Tensor(np.array([0, 1, 4, 5]).astype(np.float32)),
                        Tensor(np.array([1, 1, 3, 7]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array(0.1).astype(np.float32)),
                       Tensor(np.array(0.1).astype(np.float32))]}),
    ('Cos', {
        'block': P.Cos(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('ReduceAll', {
        'block': P.ReduceAll(),
        'desc_const': [1],
        'desc_inputs': [Tensor(np.array([[True, False], [True, True]]))],
        'desc_bprop': []}),
    ('ReduceAny', {
        'block': P.ReduceAny(),
        'desc_const': [1],
        'desc_inputs': [Tensor(np.array([[True, False], [True, True]]))],
        'desc_bprop': []}),
    ('BesselI0e', {
        'block': P.BesselI0e(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('BesselI1e', {
        'block': P.BesselI1e(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Atan', {
        'block': P.Atan(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('AtanGrad', {
        'block': G.AtanGrad(),
        'desc_inputs': [[2, 3], [2, 3]],
        'skip': ['backward']}),
    ('Atanh', {
        'block': P.Atanh(),
        'desc_inputs': [[2, 3]],
        'desc_bprop': [[2, 3]]}),
    ('Cosh', {
        'block': P.Cosh(),
        'desc_inputs': [[3, 4, 5]],
        'desc_bprop': [[3, 4, 5]]}),
    ('Sinh', {
        'block': P.Sinh(),
        'desc_inputs': [[3, 4, 5]],
        'desc_bprop': [[3, 4, 5]]}),
    ('Inv', {
        'block': P.Inv(),
        'desc_inputs': [[21, 9, 12, 5]],
        'desc_bprop': [[21, 9, 12, 5]]}),
    ('Invert', {
        'block': P.Invert(),
        'desc_inputs': [Tensor(np.array([[24, 4, 13, 9], [1, 5, 10, 8]]).astype(np.int16))],
        'desc_bprop': [],
        'skip': ['backward']}),
    ('HistogramFixedWidth', {
        'block': P.HistogramFixedWidth(5),
        'desc_inputs': [Tensor([-1.0, 0.0, 1.5, 2.0, 5.0, 15], mstype.float16), Tensor([0.0, 5.0], mstype.float16)],
        'desc_bprop': [],
        'skip': ['backward']}),
    ('Mod', {
        'block': P.Mod(),
        'desc_inputs': [[3, 4, 5], [2, 3, 4, 5]],
        'desc_bprop': [[2, 3, 4, 5]]}),
    ('IFMR', {
        'block': Q.IFMR(min_percentile=0.2, max_percentile=0.9, search_range=(1.0, 2.0),
                        search_step=1.0, with_offset=False),
        'desc_inputs': [[3, 4, 5], Tensor([0.1], mstype.float32), Tensor([0.9], mstype.float32),
                        Tensor(np.random.rand(4).astype(np.int32))],
        'desc_bprop': [],
        'skip': ['backward']}),
]

test_case_nn_ops = [
    ('BiasAdd', {
        'block': P.BiasAdd(),
        'desc_inputs': [[1, 3, 3, 3], [3]],
        'desc_bprop': [[1, 3, 3, 3]]}),
    ('BiasAddGrad', {
        'block': G.BiasAddGrad(),
        'desc_inputs': [[1, 3, 3, 3]],
        'skip': ['backward']}),
    ('GeLU', {
        'block': P.GeLU(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('GeLUGrad', {
        'block': G.GeLUGrad(),
        'desc_inputs': [[2, 2], [2, 2], [2, 2]],
        'desc_bprop': [[2, 2]],
        'skip': ['backward']}),
    ('Tanh', {
        'block': P.Tanh(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('TanhGrad', {
        'block': G.TanhGrad(),
        'desc_inputs': [[1, 3, 4, 4], [1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]],
        'skip': ['backward']}),
    ('ReLU', {
        'block': P.ReLU(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('ReLU6', {
        'block': P.ReLU6(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('ReLUV2', {
        'block': P.ReLUV2(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4], ([1, 1, 4, 4, 2], {'dtype': np.uint8})]}),
    ('ReLUGrad', {
        'block': G.ReluGrad(),
        'desc_inputs': [[1, 3, 4, 4], [1, 3, 4, 4]],
        'skip': ['backward']}),
    ('Softplus', {
        'block': P.Softplus(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('SoftplusGrad', {
        'block': G.SoftplusGrad(),
        'desc_inputs': [[1, 3, 4, 4], [1, 3, 4, 4]],
        'skip': ['backward']}),
    ('Elu', {
        'block': P.Elu(),
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[2, 3, 4]]}),
    ('EluGrad', {
        'block': G.EluGrad(),
        'desc_inputs': [[2, 3, 4], [2, 3, 4]],
        'desc_bprop': [[2, 3, 4]],
        'skip': ['backward']}),
    ('Sigmoid', {
        'block': P.Sigmoid(),
        'desc_inputs': [[1, 3, 4, 4]],
        'desc_bprop': [[1, 3, 4, 4]]}),
    ('MaxPool', {
        'block': P.MaxPool(kernel_size=(2, 2), strides=(2, 2), pad_mode="VALID"),
        'desc_inputs': [[100, 3, 28, 28]],
        'desc_bprop': [[100, 3, 14, 14]]}),
    ('MaxPoolGrad', {
        'block': G.MaxPoolGrad(kernel_size=(2, 2), strides=(2, 2), pad_mode="VALID"),
        'desc_inputs': [[3, 4, 6, 6], [3, 4, 3, 3], [3, 4, 3, 3]],
        'desc_bprop': [[3, 4, 6, 6]],
        'skip': ['backward']}),
    ('MaxPool3D', {
        'block': P.MaxPool3D(kernel_size=2, strides=2, pad_mode="VALID"),
        'desc_inputs': [[100, 3, 28, 28, 28]],
        'desc_bprop': [[100, 3, 14, 14, 14]]}),
    ('MaxPool3DGrad', {
        'block': G.MaxPool3DGrad(kernel_size=2, strides=2, pad_mode="VALID"),
        'desc_inputs': [[3, 4, 6, 6, 6], [3, 4, 3, 3, 3], [3, 4, 3, 3, 3]],
        'desc_bprop': [[3, 4, 6, 6, 6]]}),
    ('AvgPool', {
        'block': P.AvgPool(kernel_size=(2, 2), strides=(2, 2), pad_mode="VALID"),
        'desc_inputs': [[100, 3, 28, 28]],
        'desc_bprop': [[100, 3, 14, 14]]}),
    ('MaxPoolWithArgmax', {
        'block': P.MaxPoolWithArgmax(kernel_size=2, strides=2),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[128, 32, 16, 32], ([128, 32, 16, 32], {'dtype': np.int32})]}),
    ('SoftmaxCrossEntropyWithLogits', {
        'block': P.SoftmaxCrossEntropyWithLogits(),
        'desc_inputs': [[1, 10], [1, 10]],
        'desc_bprop': [[1], [1, 10]],
        'skip': ['backward_exec']}),
    ('Flatten', {
        'block': P.Flatten(),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[128, 65536]]}),
    ('LogSoftmax', {
        'block': P.LogSoftmax(),
        'desc_inputs': [[64, 2]],
        'desc_bprop': [[64, 2]]}),
    ('LogSoftmaxGrad', {
        'block': G.LogSoftmaxGrad(),
        'desc_inputs': [[16, 1234], [16, 1234]],
        'desc_bprop': [[64, 2]],
        'skip': ['backward']}),
    ('L2Normalize', {
        'block': P.L2Normalize(),
        'desc_inputs': [[2, 2]],
        'desc_bprop': [[2, 2]]}),
    ('L2NormalizeGrad', {
        'block': G.L2NormalizeGrad(),
        'desc_inputs': [[2, 2], [2, 2], [2, 2]],
        'desc_bprop': [[2, 2]],
        'skip': ['backward']}),
    ('LayerNorm', {
        'block': P.LayerNorm(),
        'desc_inputs': [[2, 16], [16], [16]],
        'desc_bprop': [[2, 16], [2, 1], [2, 1]]}),
    ('LayerNormGrad', {
        'block': G.LayerNormGrad(),
        'desc_inputs': [[2, 16], [2, 16], [2, 16], [2, 16], [16]],
        'desc_bprop': [[2, 16], [16], [16]],
        'skip': ['backward']}),
    ('BatchNorm', {
        'block': P.BatchNorm(),
        'desc_inputs': [[128, 64, 32, 32], [64], [64], [64], [64]],
        'desc_bprop': [[128, 64, 32, 32], [64], [64], [64], [64]],
        'skip': []}),
    ('BatchNormGrad', {
        'block': G.BatchNormGrad(),
        'desc_inputs': [[128, 64, 32, 32], [128, 64, 32, 32], [64], [64], [64], [64]],
        'desc_bprop': [[128, 64, 32, 32], [64], [64]],
        'skip': ['backward']}),
    ('SyncBatchNorm', {
        'block': inner.SyncBatchNorm(),
        'desc_inputs': [[128, 64, 32, 32], [64], [64], [64], [64]],
        'desc_bprop': [[128, 64, 32, 32], [64], [64], [64], [64]],
        'skip': []}),
    ('SyncBatchNormGrad', {
        'block': G.SyncBatchNormGrad(),
        'desc_inputs': [[128, 64, 32, 32], [128, 64, 32, 32], [64], [64], [64]],
        'desc_bprop': [[128, 64, 32, 32], [64], [64], [64], [64]],
        'skip': ['backward']}),
    ('TopK', {
        'block': P.TopK(),
        'desc_const': [5],
        'desc_inputs': [[20, 20, 10]],
        'desc_bprop': [[20, 20, 5]],
        'skip': ['backward']}),
    ('Sort', {
        'block': P.Sort(),
        'desc_inputs': [[2, 3, 4]],
        'desc_bprop': [[2, 3, 4], ([2, 3, 4], {'dtype': np.int32})]}),
    ('GatherV2_0', {
        'block': P.Gather(),
        'desc_const': [0],
        'desc_inputs': [[3, 1, 2], Tensor(np.array([0, 1]).astype(np.int32))],
        'desc_bprop': [[2, 1, 2]]}),
    ('GatherV2_1', {
        'block': P.Gather(),
        'desc_const': [2],
        'desc_inputs': [[3, 1, 3], Tensor(np.array([0, 1]).astype(np.int32))],
        'desc_bprop': [[3, 1, 2]]}),
    ('GatherV2_2', {
        'block': P.Gather(),
        'desc_const': [0],
        'desc_inputs': [[3, 1, 3], Tensor(np.array([[0, 1], [0, 1], [0, 1]]).astype(np.int32))],
        'desc_bprop': [[3, 2, 1, 3]]}),
    ('GatherV2_3', {
        'block': P.Gather(),
        'desc_const': [2],
        'desc_inputs': [[3, 1, 3], Tensor(np.array([[0, 1], [0, 1], [0, 1]]).astype(np.int32))],
        'desc_bprop': [[3, 1, 3, 2]]}),
    ('GatherV2_4', {
        'block': P.Gather(),
        'desc_const': [1],
        'desc_inputs': [[32, 5, 1024], Tensor(np.array([3]).astype(np.int32))],
        'desc_bprop': [[32, 1, 1024]]}),
    ('GatherV2_5', {
        'block': P.Gather(),
        'desc_const': [-1],
        'desc_inputs': [[3, 1, 3], Tensor(np.array([0, 1]).astype(np.int32))],
        'desc_bprop': [[3, 1, 2]]}),
    ('GatherV2_6', {
        'block': P.Gather(),
        'desc_const': [0],
        'desc_inputs': [[1152], Tensor(np.array(10).astype(np.int32))],
        'desc_bprop': [Tensor(np.array(10).astype(np.float32))]}),
    ('SparseGatherV2_0', {
        'block': P.SparseGatherV2(),
        'desc_const': [0],
        'desc_inputs': [[3, 1, 2], Tensor(np.array([0, 1]).astype(np.int32))],
        'desc_bprop': [[2, 1, 2]]}),
    ('Range', {
        'block': inner.Range(1.0, 5.0),
        'desc_inputs': [Tensor(np.ones([10]).astype(np.float32))],
        'desc_bprop': [[10]]}),
    ('UnsortedSegmentSum', {
        'block': P.UnsortedSegmentSum(),
        'desc_const': [1280],
        'desc_inputs': [[1280, 1024], Tensor(np.ones(1280).astype(np.int32))],
        'desc_bprop': [[1280, 1024]]}),
    ('UnsortedSegmentSum_1', {
        'block': P.UnsortedSegmentSum(),
        'desc_const': [4],
        'desc_inputs': [[3, 2, 1, 3], Tensor(np.array([[0, 1], [0, 1], [0, 1]]).astype(np.int32))],
        'desc_bprop': [[4, 1, 3]]}),
    ('UnsortedSegmentMin', {
        'block': P.UnsortedSegmentMin(),
        'desc_const': [4],
        'desc_inputs': [[3, 2, 1, 3], Tensor(np.array([1, 2, 3]).astype(np.int32))],
        'desc_bprop': [[4, 2, 1, 3]]}),
    ('UnsortedSegmentMax', {
        'block': P.UnsortedSegmentMax(),
        'desc_const': [4],
        'desc_inputs': [[3, 2, 1, 3], Tensor(np.array([1, 2, 3]).astype(np.int32))],
        'desc_bprop': [[4, 2, 1, 3]]}),
    ('UnsortedSegmentProd', {
        'block': P.UnsortedSegmentProd(),
        'desc_const': [4],
        'desc_inputs': [[3, 2, 1, 3], Tensor(np.array([0, 1, 0]).astype(np.int32))],
        'desc_bprop': [[4, 2, 1, 3]]}),
    ('DropoutGenMask', {
        'block': P.DropoutGenMask(),
        'desc_const': [(2, 2), Tensor(0.5, mstype.float32)],
        'desc_inputs': [],
        'desc_bprop': [Tensor(np.ones(1).astype(np.int8))],
        'skip': ['backward']}),
    ('DropoutDoMask', {
        'block': P.DropoutDoMask(),
        'desc_const': [Tensor(0.5)],
        'desc_inputs': [[64, 12, 128, 128], Tensor(np.ones(1572864).astype(np.uint8))],
        'desc_bprop': [[64, 12, 128, 128]]}),
    ('Dropout', {
        'block': nn.Dropout(0.5),
        'desc_inputs': [[64, 12, 128, 128]],
        'desc_bprop': [[64, 12, 128, 128]]}),
    ('ReduceMean0', {
        'block': P.ReduceMean(),
        'desc_const': [(2,)],
        'desc_inputs': [[3, 2, 2]],
        'desc_bprop': [[3, 2]]}),
    ('ReduceMean1', {
        'block': P.ReduceMean(),
        'desc_const': [2],
        'desc_inputs': [[3, 2, 2]],
        'desc_bprop': [[3, 2]]}),
    ('All', {
        'block': P.ReduceAll(),
        'desc_const': [(1,)],
        'desc_inputs': [Tensor(np.ones([3, 2]).astype(np.bool_))],
        'desc_bprop': [[3]],
        'skip': ['backward']}),
    ('DescConst', {
        'block': Tensor(np.array([2], np.float32)),
        'desc_inputs': [],
        'desc_bprop': [[1]],
        'skip': ['backward'],
        'add_fake_input': True}),
    ('Fill', {
        'block': P.Fill(),
        'desc_const': [mstype.float32, (2, 3), 1.0],
        'desc_inputs': [],
        'desc_bprop': [[2, 3]],
        'skip': ['backward'],
        'add_fake_input': True}),
    ('OnesLike', {
        'block': P.OnesLike(),
        'desc_inputs': [Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))],
        'desc_bprop': [Tensor(np.array([[1, 1], [1, 1]]).astype(np.int32))]
    }),
    ('ZerosLike', {
        'block': P.ZerosLike(),
        'desc_inputs': [Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))],
        'desc_bprop': [Tensor(np.array([[1, 1], [1, 1]]).astype(np.int32))]
    }),
    ('Softmax', {
        'block': P.Softmax(),
        'desc_inputs': [[5, 5]],
        'desc_bprop': [[5, 5]]}),
    ('Softsign', {
        'block': P.Softsign(),
        'desc_inputs': [[5, 5]],
        'desc_bprop': [[5, 5]]}),
    ('DepthwiseConv2dNative_1', {
        'block': P.DepthwiseConv2dNative(3, (3, 3), pad_mode="pad", pad=1, stride=2),
        'desc_inputs': [[10, 32, 32, 32], [1, 32, 3, 3]],
        'desc_bprop': [[10, 32, 16, 16]]}),
    ('DepthwiseConv2dNative_2', {
        'block': P.DepthwiseConv2dNative(1, (3, 3), pad_mode="same", pad=0, stride=1),
        'desc_inputs': [[2592, 2048, 4, 4], [1, 2048, 3, 3]],
        'desc_bprop': [[2592, 2048, 4, 4]]}),
    ('SigmoidCrossEntropyWithLogits', {
        'block': P.SigmoidCrossEntropyWithLogits(),
        'desc_inputs': [[128, 10], [128, 10]],
        'desc_bprop': [[128, 10]]}),
    ('Pad', {
        'block': P.Pad(((1, 2), (2, 3))),
        'desc_inputs': [[7, 7]],
        'desc_bprop': [[10, 12]]}),
    ('BinaryCrossEntropy', {
        'block': P.BinaryCrossEntropy(),
        'desc_inputs': [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        'desc_bprop': []}),
    ('SparseApplyAdagrad', {
        'block': SparseApplyAdagradNet(),
        'desc_inputs': [[3, 3], Tensor(np.ones((3,), np.int32))],
        'desc_bprop': [[3, 3], [3, 3]],
        'skip': ['backward']}),
    ('SparseApplyAdagradV2', {
        'block': SparseApplyAdagradV2Net(),
        'desc_inputs': [[3, 3], Tensor(np.ones((3,), np.int32))],
        'skip': ['backward']}),
    ('SparseApplyFtrl', {
        'block': SparseApplyFtrlNet(),
        'desc_inputs': [[3, 3], Tensor(np.ones((3,), np.int32))],
        'skip': ['backward']}),
    ('SparseApplyFtrlV2', {
        'block': SparseApplyFtrlV2Net(),
        'desc_inputs': [[3, 3], Tensor(np.ones((3,), np.int32))],
        'skip': ['backward']}),
    ('ApplyProximalAdagrad', {
        'block': ApplyProximalAdagradNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('SparseApplyProximalAdagrad', {
        'block': SparseApplyProximalAdagradNet(),
        'desc_inputs': [[3, 3], Tensor(np.ones((3,), np.int32))],
        'skip': ['backward']}),
    ('ApplyAdaMax', {
        'block': ApplyAdaMaxNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyAdadelta', {
        'block': ApplyAdadeltaNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyAdagrad', {
        'block': ApplyAdagradNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyAdagradV2', {
        'block': ApplyAdagradV2Net(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyAddSign', {
        'block': ApplyAddSignNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyPowerSign', {
        'block': ApplyPowerSignNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyGradientDescent', {
        'block': ApplyGradientDescentNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('ApplyProximalGradientDescent', {
        'block': ApplyProximalGradientDescentNet(),
        'desc_inputs': [[3, 3]],
        'skip': ['backward']}),
    ('Flatten_1', {
        'block': NetForFlatten(),
        'desc_inputs': [Tensor(np.ones([2, 3, 4]).astype(np.int32)), Tensor(np.ones([2, 12]).astype(np.int32))],
        'desc_bprop': [Tensor(np.ones([2, 12]).astype(np.int32))],
        'skip': ['backward']}),
    ('Flatten_2', {
        'block': NetForFlatten(),
        'desc_inputs': [Tensor(np.ones([8]).astype(np.int32)), Tensor(np.ones([8, 3]).astype(np.int32))],
        'desc_bprop': [Tensor(np.ones([8, 3]).astype(np.int32))],
        'skip': ['backward']}),
    ('Flatten_3', {
        'block': NetForFlattenComposed(),
        'desc_inputs': [Tensor(np.ones([2, 3, 4]).astype(np.int32)), Tensor(np.ones([2, 12]).astype(np.int32))],
        'desc_bprop': [Tensor(np.ones([2, 12]).astype(np.int32))],
        'skip': []}),
    ('ArgmaxNet', {
        'block': ArgmaxNet(),
        'desc_inputs': [Tensor(np.array([[128, 32, 32, 64], [128, 32, 32, 64]]).astype(np.float16))],
        'desc_bprop': [Tensor(np.array([[128, 32, 32, 64], [128, 32, 32, 64]]).astype(np.float16))],
        'skip': ['backward']}),
    ('ArgminNet', {
        'block': ArgminNet(),
        'desc_inputs': [Tensor(np.array([[128, 32, 32, 64], [128, 32, 32, 64]]).astype(np.float16))],
        'desc_bprop': [Tensor(np.array([[128, 32, 32, 64], [128, 32, 32, 64]]).astype(np.float16))],
        'skip': ['backward']}),
    ('StridedSliceNet', {
        'block': StridedSliceNet(),
        'desc_inputs': [[6, 7, 8, 9, 10]],
        'skip': ['backward']}),
    ('OneHot', {
        'block': P.OneHot(),
        'desc_const': [3, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)],
        'desc_inputs': [Tensor(np.array([64]).astype(np.int32))],
        'desc_bprop': [[1, 3]]}),
    ('ReduceProd_0', {
        'block': P.ReduceProd(),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[2]]}),
    ('ReduceProd_1', {
        'block': P.ReduceProd(keep_dims=True),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[1, 2]]}),
    ('CumProd', {
        'block': P.CumProd(),
        'desc_const': [0],
        'desc_inputs': [[3, 2]],
        'desc_bprop': [[3, 2]]}),
    ('ApplyFtrl', {
        'block': ApplyFtrlNet(),
        'desc_inputs': [[3, 3]],
        'desc_bprop': [3, 3],
        'skip': ['backward']}),
    ('ApplyRMSProp', {
        'block': ApplyRMSNet(),
        'desc_inputs': [[3, 3]],
        'desc_bprop': [3, 3],
        'skip': ['backward']}),
    ('ApplyCenteredRMSProp', {
        'block': P.ApplyCenteredRMSProp(),
        'desc_const': [0.9, 0.0, 1e-10, 0.001],
        'desc_inputs': [Tensor(1., mstype.float32), Tensor(2., mstype.float32), Tensor(1., mstype.float32),
                        Tensor(2., mstype.float32), Tensor(1., mstype.float32)],
        'desc_bprop': [1],
        'skip': ['backward']}),
    ('CTCLoss', {
        'block': P.CTCLoss(),
        'desc_inputs': [Tensor(np.ones([6, 4, 6]).astype(np.float32)),
                        Tensor(np.array([[0, 1], [1, 0], [2, 3], [3, 2]]).astype(np.int64)),
                        Tensor(np.array([1, 2, 3, 4]).astype(np.int32)),
                        Tensor(np.array([6, 6, 6, 6]).astype(np.int32))],
        'desc_bprop': [[4], [6, 4, 6]]}),
    ('CTCGreedyDecoder', {
        'block': CTCGreedyDecoderNet(),
        'desc_inputs': [[2, 2, 3], Tensor(np.array([2, 2]).astype(np.int32))],
        'skip': ['backward']}),
    ('L2Loss_1', {
        'block': P.L2Loss(),
        'desc_inputs': [Tensor(np.array([1, 2, 3, 4]), mstype.float32)],
        'desc_bprop': []}),
    ('L2Loss_2', {
        'block': P.L2Loss(),
        'desc_inputs': [Tensor(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), mstype.float16)],
        'desc_bprop': []}),
    ('BCEWithLogitsLoss', {
        'block': P.BCEWithLogitsLoss(),
        'desc_inputs': [[3, 3], [3, 3], [3, 3], [3, 3]],
        'desc_bprop': []}),
    ('ResizeBilinear', {
        'block': P.ResizeBilinear((5, 5)),
        'desc_inputs': [Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mstype.float16)],
        'desc_bprop': [Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mstype.float32)]}),
    ('ResizeBilinearGrad', {
        'block': G.ResizeBilinearGrad(),
        'desc_inputs': [Tensor([[[[1, 2, 3, 4, 5]]]], mstype.float32), Tensor([[[[1, 2, 3, 4, 5]]]], mstype.float32)],
        'desc_bprop': [Tensor([[[[1, 2, 3, 4, 5]]]], mstype.float32)],
        'skip': ['backward']}),
    ('ROIAlign', {
        'block': P.ROIAlign(7, 7, 0.03125, 2),
        'desc_inputs': [[2, 256, 192, 320], [1024, 5]],
        'desc_bprop': [[1024, 256, 7, 7]]}),
    ('ROIAlignGrad', {
        'block': G.ROIAlignGrad((1, 1, 1, 1), 2, 2, 0.5, 2),
        'desc_inputs': [[1, 1, 2, 2], [1, 5]],
        'desc_bprop': [[1, 1, 2, 2]],
        'skip': ['backward']}),
    ('LARSUpdate', {
        'block': P.LARSUpdate(1e-05, 0.001, False),
        'desc_const': [0.0, 0.001],
        'desc_inputs': [[3, 3], [3, 3], [3, 3], [3, 3]],
        'desc_bprop': [3, 3],
        'skip': ['backward']}),
    ('SGD', {
        'block': P.SGD(0.0, 0.0, False),
        'desc_inputs': [[3, 3], [3, 3], Tensor(0.001, mstype.float32), [3, 3], Tensor(0.1, mstype.float32), [3, 3]],
        'desc_bprop': [3, 3],
        'skip': ['backward']}),
    ('BinaryCrossEntropy', {
        'block': P.BinaryCrossEntropy(),
        'desc_inputs': [Tensor([[0.3, 0.8], [0.4, 0.3]], mstype.float16),
                        Tensor([[0.4, 1.2], [-0.4, -0.9]], mstype.float16),
                        Tensor([[-1.4, -0.7], [0.9, 0.7]], mstype.float16)],
        'desc_bprop': []}),
    ('BinaryCrossEntropyGrad', {
        'block': G.BinaryCrossEntropyGrad(),
        'desc_inputs': [Tensor([[0.3, 0.8], [0.4, 0.3]], mstype.float16),
                        Tensor([[0.4, 1.2], [-0.4, -0.9]], mstype.float16), Tensor(0.85, mstype.float16),
                        Tensor([[-1.4, -0.7], [0.9, 0.7]], mstype.float16)],
        'desc_bprop': [],
        'skip': ['backward']}),
    ('DataFormatDimMap', {
        'block': P.DataFormatDimMap(),
        'desc_inputs': [Tensor([0, 1, 2, 3], mstype.int32)],
        'desc_bprop': [],
        'skip': ['backward']}),
    ('MaxPoolGradGrad', {
        'block': G.MaxPoolGradGrad(),
        'desc_inputs': [Tensor(np.random.rand(1, 1, 2, 2), mstype.float16),
                        Tensor(np.random.rand(1, 1, 2, 2), mstype.float16),
                        Tensor(np.random.rand(1, 1, 2, 2), mstype.float16)],
        'desc_bprop': [],
        'skip': ['backward']}),
    ('MaxPoolGradGradWithArgmax', {
        'block': G.MaxPoolGradGradWithArgmax(),
        'desc_inputs': [Tensor(np.random.rand(1, 1, 2, 2), mstype.float16),
                        Tensor(np.random.rand(1, 1, 2, 2), mstype.float16),
                        Tensor(np.zeros((1, 1, 2, 2)), mstype.uint16)],
        'desc_bprop': [],
        'skip': ['backward']}),
]

test_case_array_ops = [
    ('SpaceToDepth', {
        'block': P.SpaceToDepth(2),
        'desc_inputs': [[1, 3, 2, 2]],
        'desc_bprop': [[1, 12, 1, 1]]}),
    ('DepthToSpace', {
        'block': P.DepthToSpace(2),
        'desc_inputs': [[1, 12, 1, 1]],
        'desc_bprop': [[1, 3, 2, 2]]}),
    ('Split', {
        'block': P.Split(1, 2),
        'desc_inputs': [Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]))],
        'skip': ['backward']}),
    ('Argmax', {
        'block': P.Argmax(),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [0],
        'skip': ['backward']}),
    ('Argmin', {
        'block': P.Argmin(),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [1],
        'skip': ['backward']}),
    ('ArgMaxWithValue', {
        'block': P.ArgMaxWithValue(),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[1], [1]],
        'skip': ['backward']}),
    ('ArgMinWithValue', {
        'block': P.ArgMinWithValue(),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[1], [1]],
        'skip': ['backward']}),
    ('Transpose_dim3', {
        'block': P.Transpose(),
        'desc_const': [(0, 2, 1)],
        'desc_inputs': [[1, 2, 3]],
        'desc_bprop': [[1, 3, 2]]}),
    ('Transpose_dim4', {
        'block': P.Transpose(),
        'desc_const': [(0, 1, 2, 3)],
        'desc_inputs': [[1, 2, 3, 4]],
        'desc_bprop': [[1, 2, 4, 3]]}),
    ('AddN', {
        'block': NetForTupleInput(P.AddN()),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('AccumulateNV2', {
        'block': NetForTupleInput(P.AccumulateNV2()),
        'desc_inputs': [[2, 3, 3, 5], [2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Shape', {
        'block': P.Shape(),
        'desc_inputs': [[3, 3, 2, 2]],
        'skip': ['backward']}),
    ('Reshape', {
        'block': P.Reshape(),
        'desc_const': [(64,)],
        'desc_inputs': [[64, 1]],
        'desc_bprop': [[64]]}),
    ('Cast', {
        'block': P.Cast(),
        'desc_const': [mstype.int32],
        'desc_inputs': [[2, 3, 4, 5]],
        'desc_bprop': [Tensor(np.ones((2, 3, 4, 5)).astype(np.int32))]}),
    ('ExpandDims', {
        'block': P.ExpandDims(),
        'desc_const': [0],
        'desc_inputs': [[2, 2]],
        'desc_bprop': [[1, 2, 2]]}),
    ('ExpandDims_1', {
        'block': P.ExpandDims(),
        'desc_const': [-1],
        'desc_inputs': [[2, 2]],
        'desc_bprop': [[2, 2, 1]]}),
    ('Squeeze', {
        'block': P.Squeeze(2),
        'desc_inputs': [[3, 2, 1]],
        'desc_bprop': [[3, 2]]}),
    ('Squeeze_0', {
        'block': P.Squeeze(),
        'desc_inputs': [[3, 1, 2, 1]],
        'desc_bprop': [[3, 2]]}),
    ('Squeeze_1', {
        'block': P.Squeeze(),
        'desc_inputs': [[1, 1, 1, 1]],
        'desc_bprop': [1.0],
        'skip': ['backward']}),
    ('Squeeze_2', {
        'block': P.Squeeze((2, 3)),
        'desc_inputs': [[3, 2, 1, 1]],
        'desc_bprop': [[3, 2]]}),
    ('Size', {
        'block': P.Size(),
        'desc_inputs': [[2, 3, 5]],
        'skip': ['backward']}),
    ('Tile_0', {
        'block': P.Tile(),
        'desc_const': [(1, 2)],
        'desc_inputs': [[64, 1]],
        'desc_bprop': [[64, 2]]}),
    ('Tile_1', {
        'block': P.Tile(),
        'desc_const': [(1, 1)],
        'desc_inputs': [[64, 1]],
        'desc_bprop': [[64, 1]]}),
    ('Tile_2', {
        'block': P.Tile(),
        'desc_const': [(2, 1, 1, 2)],
        'desc_inputs': [[2, 2, 2]],
        'desc_bprop': [[2, 2, 2, 4]]}),
    ('ReverseV2', {
        'block': P.ReverseV2(axis=[1]),
        'desc_inputs': [(Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)))],
        'desc_bprop': [(Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)))]}),
    ('Rint', {
        'block': P.Rint(),
        'desc_inputs': [(Tensor(np.array([-1.6, -0.1, 1.5, 2.0]).astype(np.float32)))],
        'skip': ['backward']}),
    ('ConcatV2_0', {
        'block': NetForConcat1(),
        'desc_inputs': [
            Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32)),
            Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))],
        'desc_bprop': [([4, 2], {'dtype': np.int32})]}),
    ('ConcatV2_1', {
        'block': NetForConcat2(),
        'desc_inputs': [Tensor(np.array([[[0, 1, 2]], [[2, 1, 2]]]).astype(np.int32)),
                        Tensor(np.array([[[0, 1]], [[2, 1]]]).astype(np.int32))],
        'desc_bprop': [([2, 1, 5], {'dtype': np.int32})]}),
    ('ConcatV2_2', {
        'block': NetForConcat(),
        'desc_inputs': [[2, 2]],
        'desc_bprop': [[4, 2]]}),
    ('ConcatV2_3', {
        'block': NetForConcat1(),
        'desc_inputs': [[2, 2], [2, 2]],
        'desc_bprop': [[4, 2]]}),
    ('ConcatV2_4', {
        'block': NetForConcat3(),
        'desc_inputs': [
            Tensor(np.ones((3, 2, 3), np.float32)),
            Tensor(np.ones((5, 2, 3), np.float32)),
            Tensor(np.ones((6, 2, 3), np.float32))],
        'desc_bprop': [[14, 2, 3]]}),
    ('ConcatV2_5', {
        'block': NetForConcat4(),
        'desc_inputs': [Tensor(np.array([1], np.float32)),
                        Tensor(np.array([1], np.float32)),
                        Tensor(np.array([1], np.float32))],
        'desc_bprop': [[3,]]}),
    ('Stack_0', {
        'block': NetForStackInput(P.Stack()),
        'desc_inputs': [[2, 2], [2, 2], [2, 2]],
        'desc_bprop': [[3, 2, 2]],
    }),
    ('Stack_1', {
        'block': NetForStackInput(P.Stack(axis=-2)),
        'desc_inputs': [[3, 2, 3], [3, 2, 3], [3, 2, 3]],
        'desc_bprop': [[3, 2, 3, 3]],
    }),
    ('Stack_2', {
        'block': NetForStackInput(P.Stack()),
        'desc_inputs': [[128, 128], [128, 128]],
        'desc_bprop': [[2, 128, 128]],
    }),
    ('Stack_3', {
        'block': NetForStackInput(P.Stack()),
        'desc_inputs': [[2, 2]],
        'desc_bprop': [[1, 2, 2]]}),
    ('Unpack_0', {
        'block': NetForUnpackInput(P.Unstack(axis=0)),
        'desc_inputs': [[2, 4]],
        'desc_bprop': [[4], [4]],
    }),
    ('Unpack_1', {
        'block': NetForUnpackInput(P.Unstack(axis=-1)),
        'desc_inputs': [Tensor(np.array([[1, 1, 1]], np.float32))],
        'desc_bprop': [[1], [1], [1]],
    }),
    ('Diag_1', {
        'block': P.Diag(),
        'desc_inputs': [[4]],
        'desc_bprop': [[4, 4]],
    }),
    ('Diag_2', {
        'block': P.Diag(),
        'desc_inputs': [[4, 4]],
        'desc_bprop': [[4, 4, 4, 4]],
    }),
    ('DiagPart_1', {
        'block': P.DiagPart(),
        'desc_inputs': [[4, 4]],
        'desc_bprop': [[4]],
    }),
    ('DiagPart_2', {
        'block': P.DiagPart(),
        'desc_inputs': [[4, 4, 4, 4]],
        'desc_bprop': [[4, 4]],
    }),
    ('SpaceToBatch_1', {
        'block': P.SpaceToBatch(2, [[0, 0], [0, 0]]),
        'desc_inputs': [[1, 3, 2, 2]],
        'desc_bprop': [[4, 3, 1, 1]],
    }),
    ('SpaceToBatch_2', {
        'block': P.SpaceToBatch(2, [[1, 1], [0, 4]]),
        'desc_inputs': [[1, 3, 2, 2]],
        'desc_bprop': [[4, 3, 2, 3]],
    }),
    ('BatchToSpace_1', {
        'block': P.BatchToSpace(2, [[0, 0], [0, 0]]),
        'desc_inputs': [[4, 3, 1, 1]],
        'desc_bprop': [[1, 3, 2, 2]],
    }),
    ('BatchToSpace_2', {
        'block': P.BatchToSpace(2, [[0, 0], [0, 1]]),
        'desc_inputs': [[4, 3, 1, 1]],
        'desc_bprop': [[1, 3, 2, 1]],
    }),
    ('UnsortedSegmentMin_1', {
        'block': P.UnsortedSegmentMin(),
        'desc_const': [2],
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [4, 5, 6], [4, 2, 1]]).astype(np.float32)),
                        Tensor(np.array([0, 1, 1]).astype(np.int32))],
        'desc_bprop': [Tensor(np.array([[1, 2, 3], [4, 2, 1]]).astype(np.float32))]}),
    ('BroadcastTo', {
        'block': P.BroadcastTo((2, 3)),
        'desc_inputs': [Tensor(np.array([1, 2, 3]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32))]}),
    ('InTopK', {
        'block': P.InTopK(2),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [2, 3, 6], [4, 2, 1]]).astype(np.float32)),
                        Tensor(np.array([2, 1, 2]).astype(np.int32))],
        'skip': ['backward'],
    }),
    ('InplaceUpdate', {
        'block': P.InplaceUpdate((0, 2)),
        'desc_inputs': [Tensor(np.arange(24).reshape(3, 4, 2).astype(np.float32)),
                        Tensor(np.arange(16).reshape(2, 4, 2).astype(np.float32))],
        'skip': ['backward'],
    }),
    ('ReverseSequence', {
        'block': P.ReverseSequence(1, 0),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)),
                        Tensor(np.array([1, 2, 3]).astype(np.int32))],
        'desc_bprop': [[3, 3]]}),
    ('EditDistance', {
        'block': EditDistance(Tensor(np.array([1, 1, 2]).astype(np.int64)),
                              Tensor(np.array([2, 2, 2]).astype(np.int64))),
        'desc_inputs': [Tensor(np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]).astype(np.int64)),
                        Tensor(np.array([1, 2, 3]).astype(np.float32)),
                        Tensor(np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]).astype(np.int64)),
                        Tensor(np.array([1, 3, 2, 1]).astype(np.float32))],
        'skip': ['backward'],
    }),
    ('LinSpace', {
        'block': P.LinSpace(),
        'desc_const': [5],
        'desc_inputs': [Tensor(1, mstype.float32),
                        Tensor(10, mstype.float32)],
        'skip': ['backward'],
    }),
    ('MatrixDiag', {
        'block': inner.MatrixDiag(),
        'desc_inputs': [Tensor(np.array([1, -1]), mstype.float32),
                        Tensor(np.arange(-12, 0).reshape(3, 2, 2), mstype.float32)],
        'skip': ['backward'],
    }),
    ('MatrixDiagPart', {
        'block': inner.MatrixDiagPart(),
        'desc_inputs': [Tensor(np.arange(12).reshape(3, 2, 2), mstype.float32),
                        Tensor(np.arange(-12, 0).reshape(3, 2, 2), mstype.float32)],
        'skip': ['backward'],
    }),
    ('MatrixSetDiag', {
        'block': inner.MatrixSetDiag(),
        'desc_inputs': [Tensor(np.arange(12).reshape(3, 2, 2), mstype.float32),
                        Tensor(np.arange(6).reshape(3, 2), mstype.float32),
                        Tensor(np.arange(-12, 0).reshape(3, 2, 2), mstype.float32)],
        'skip': ['backward'],
    }),
    ('TransShape', {
        'block': P.TransShape(),
        'desc_const': [(1, 12, 24, 24)],
        'desc_inputs': [[1, 3, 24, 24]],
        'desc_bprop': [[1, 12, 24, 24]],
    }),
    ('ParallelConcat', {
        'block': ParallelConcatNet(),
        'desc_inputs': [Tensor([[1, 2]], mstype.float32),
                        Tensor([[5, 6]], mstype.float32)],
        'skip': ['backward'],
    }),
]

test_case_other_ops = [
    ('ScalarLog', {
        'block': F.scalar_log,
        'desc_const': [0.0],
        'desc_inputs': [],
        'desc_bprop': [1],
        'skip': ['backward']}),
    ('BoundingBoxEncode', {
        'block': P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)),
        'desc_inputs': [[256, 4], [256, 4]],
        'desc_bprop': [[256, 4]],
        'skip': ['backward']}),
    ('BoundingBoxDecode', {
        'block': P.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), max_shape=(768, 1280)),
        'desc_inputs': [[256, 4], [256, 4]],
        'desc_bprop': [[256, 4]],
        'skip': ['backward']}),
    ('GatherNd', {
        'block': P.GatherNd(),
        'desc_inputs': (Tensor(np.ones((1, 3, 6, 6), np.float32)),
                        Tensor(np.ones((2, 4), np.int32))),
        'desc_bprop': [[2]]}),
    ('ScatterNd', {
        'block': P.ScatterNd(),
        'desc_const': [(3, 3)],
        'desc_inputs': (Tensor(np.ones((2, 2), np.int32)),
                        Tensor(np.ones((2,), np.int32))),
        'desc_bprop': [([3, 3], {'dtype': np.int32})]}),
    ('TensorScatterUpdate', {
        'block': P.TensorScatterUpdate(),
        'desc_inputs': (Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.float32),
                        Tensor(np.array([[0, 1], [1, 2]], np.int32)),
                        Tensor(np.ones([2, 5], np.float32) * 99)),
        'desc_bprop': [([3, 4, 5], {'dtype': np.float32})]}),
    ('ScatterMaxUseLocking', {
        'block': ScatterMax(use_locking=True),
        'desc_inputs': (Tensor(np.array([1, 0], np.int32)),
                        Tensor(np.array([[5.0, 5.0, 5.0], [4.0, 4.0, 4.0]], np.float32))),
        'skip': ['backward']}),
    ('ScatterMax1d', {
        'block': ScatterMax(),
        'desc_inputs': (Tensor(np.array([1, 0], np.int32)),
                        Tensor(np.array([[5.0, 5.0, 5.0], [4.0, 4.0, 4.0]], np.float32))),
        'skip': ['backward']}),
    ('ScatterMaxF32', {
        'block': ScatterMax(),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.float32) * 99)),
        'skip': ['backward']}),
    ('ScatterMaxF16', {
        'block': ScatterMax(np.float16),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.float16) * 99)),
        'skip': ['backward']}),
    ('ScatterMaxI32', {
        'block': ScatterMax(np.int32),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.int32) * 99)),
        'skip': ['backward']}),
    ('ScatterMinUseLocking', {
        'block': ScatterMin(use_locking=True),
        'desc_inputs': (Tensor(np.array([1, 0], np.int32)),
                        Tensor(np.ones([2, 3], np.float32))),
        'skip': ['backward']}),
    ('ScatterMin1d', {
        'block': ScatterMin(),
        'desc_inputs': (Tensor(np.array([1, 0], np.int32)),
                        Tensor(np.ones([2, 3], np.float32))),
        'skip': ['backward']}),
    ('ScatterMinF32', {
        'block': ScatterMin(),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.float32))),
        'skip': ['backward']}),
    ('ScatterMinF16', {
        'block': ScatterMin(np.float16),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.float16))),
        'skip': ['backward']}),
    ('ScatterMinI32', {
        'block': ScatterMin(np.int32),
        'desc_inputs': (Tensor(np.array([[0, 0], [1, 1]], np.int32)),
                        Tensor(np.ones([2, 2, 3], np.int32))),
        'skip': ['backward']}),
    ('ScatterUpdate', {
        'block': ScatterUpdate((6,)),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterAddUseLocking', {
        'block': ScatterAdd((6,), use_locking=True),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterNonAliasingAdd_1d', {
        'block': ScatterNonAliasingAdd((8,)),
        'desc_inputs': (Tensor(np.array([[2], [3], [4], [5]], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0, 8.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterNdAdd', {
        'block': ScatterNdAdd((8,)),
        'desc_inputs': (Tensor(np.array([[2], [3], [4], [5]], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0, 8.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterNdSub', {
        'block': ScatterNdAdd((8,)),
        'desc_inputs': (Tensor(np.array([[2], [3], [4], [5]], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0, 8.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterAdd', {
        'block': ScatterAdd((6,)),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterAddScalar', {
        'block': ScatterAdd((6,)),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterAdd2d', {
        'block': ScatterAdd((3, 4)),
        'desc_inputs': (Tensor(np.array([[0, 1], [1, 2]], np.int32)),
                        Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2]],
                                         [[3, 3, 3, 3], [4, 4, 4, 4]]], np.float32))),
        'skip': ['backward']}),
    ('ScatterAddF16', {
        'block': ScatterAdd((6,), np.float16),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float16))),
        'skip': ['backward']}),
    ('ScatterAddI8', {
        'block': ScatterAdd((6,), np.int8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int8))),
        'skip': ['backward']}),
    ('ScatterAddI32', {
        'block': ScatterAdd((6,), np.int32),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int32))),
        'skip': ['backward']}),
    ('ScatterAddU8', {
        'block': ScatterAdd((6,), np.uint8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.uint8))),
        'skip': ['backward']}),
    ('ScatterMulUseLocking', {
        'block': ScatterMul((6,), use_locking=True),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterMulScalar', {
        'block': ScatterMul((6,)),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterMul2d', {
        'block': ScatterMul((3, 4)),
        'desc_inputs': (Tensor(np.array([[0, 1], [1, 2]], np.int32)),
                        Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2]],
                                         [[3, 3, 3, 3], [4, 4, 4, 4]]], np.float32))),
        'skip': ['backward']}),
    ('ScatterMulF16', {
        'block': ScatterMul((6,), np.float16),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float16))),
        'skip': ['backward']}),
    ('ScatterMulI8', {
        'block': ScatterMul((6,), np.int8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int8))),
        'skip': ['backward']}),
    ('ScatterMulI32', {
        'block': ScatterMul((6,), np.int32),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int32))),
        'skip': ['backward']}),
    ('ScatterMulU8', {
        'block': ScatterMul((6,), np.uint8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.uint8))),
        'skip': ['backward']}),
    ('ScatterDivUseLocking', {
        'block': ScatterDiv((6,), use_locking=True),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterDivScalar', {
        'block': ScatterDiv((6,)),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterDiv2d', {
        'block': ScatterDiv((3, 4)),
        'desc_inputs': (Tensor(np.array([[0, 1], [1, 2]], np.int32)),
                        Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2]],
                                         [[3, 3, 3, 3], [4, 4, 4, 4]]], np.float32))),
        'skip': ['backward']}),
    ('ScatterDivF16', {
        'block': ScatterDiv((6,), np.float16),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float16))),
        'skip': ['backward']}),
    ('ScatterDivI8', {
        'block': ScatterDiv((6,), np.int8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int8))),
        'skip': ['backward']}),
    ('ScatterDivU8', {
        'block': ScatterDiv((6,), np.uint8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.uint8))),
        'skip': ['backward']}),
    ('ScatterSubUseLocking', {
        'block': ScatterSub((6,), use_locking=True),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterSubScalar', {
        'block': ScatterSub((6,)),
        'desc_inputs': (Tensor(np.array([2], np.int32)),
                        Tensor(np.array([2.0], np.float32))),
        'skip': ['backward']}),
    ('ScatterSub2d', {
        'block': ScatterSub((3, 4)),
        'desc_inputs': (Tensor(np.array([[0, 1], [1, 2]], np.int32)),
                        Tensor(np.array([[[1, 1, 1, 1], [2, 2, 2, 2]],
                                         [[3, 3, 3, 3], [4, 4, 4, 4]]], np.float32))),
        'skip': ['backward']}),
    ('ScatterSubF16', {
        'block': ScatterSub((6,), np.float16),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2.0, 3.0, 4.0], np.float16))),
        'skip': ['backward']}),
    ('ScatterSubI32', {
        'block': ScatterSub((6,), np.int32),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int32))),
        'skip': ['backward']}),
    ('ScatterSubI8', {
        'block': ScatterSub((6,), np.int8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([2, 3, 4], np.int8))),
        'skip': ['backward']}),
    ('ScatterSubU8', {
        'block': ScatterSub((6,), np.uint8),
        'desc_inputs': (Tensor(np.array([2, 0, 5], np.int32)),
                        Tensor(np.array([1, 1, 0], np.uint8))),
        'skip': ['backward']}),
    ('SmoothL1Loss', {
        'block': P.SmoothL1Loss(),
        'desc_inputs': [[256, 4], [256, 4]],
        'desc_bprop': [[256, 4]]}),
    ('IOU', {
        'block': P.IOU(),
        'desc_inputs': [Tensor(np.ones((256, 4), np.float16)), Tensor(np.ones((128, 4), np.float16))],
        'desc_bprop': [convert([128, 256], np.float16)]}),
    ('Summary', {
        'block': SummaryNet(),
        'desc_inputs': [Tensor(np.array([1.1]).astype(np.float32)),
                        Tensor(np.array([1.2]).astype(np.float32))],
        'skip': ['backward']}),
    ('HistogramSummary', {
        'block': HistogramSummaryNet(),
        'desc_inputs': [Tensor(np.array([1.1]).astype(np.float32)),
                        Tensor(np.array([1.2]).astype(np.float32))],
        'skip': ['backward']}),
    ('PopulationCount', {
        'block': P.PopulationCount(),
        'desc_inputs': [Tensor(np.array([1, 2, 3]).astype(np.int16))],
        'skip': ['backward']}),
    ('BasicLSTMCellNet', {
        'block': BasicLSTMCellNet(),
        'desc_inputs': [Tensor(np.random.rand(1, 32).astype(np.float16)),
                        Tensor(np.random.rand(1, 64).astype(np.float16)),
                        Tensor(np.random.rand(1, 64).astype(np.float16)),
                        Tensor(np.random.rand(96, 256).astype(np.float16)),
                        Tensor(np.random.rand(256,).astype(np.float16))],
        'desc_bprop': [Tensor(np.random.rand(1, 64).astype(np.float16)),
                       Tensor(np.random.rand(1, 64).astype(np.float16)),
                       Tensor(np.random.rand(1, 64).astype(np.float16)),
                       Tensor(np.random.rand(1, 64).astype(np.float16)),
                       Tensor(np.random.rand(1, 64).astype(np.float16))]}),
    ('DynamicGRUV2Net', {
        'block': DynamicGRUV2Net(),
        'desc_inputs': [Tensor(np.random.rand(2, 8, 64).astype(np.float16)),
                        Tensor(np.random.rand(64, 48).astype(np.float16)),
                        Tensor(np.random.rand(16, 48).astype(np.float16)),
                        Tensor(np.random.rand(48).astype(np.float16)),
                        Tensor(np.random.rand(48).astype(np.float16)),
                        Tensor(np.random.rand(8, 16).astype(np.float16))],
        'desc_bprop': [Tensor(np.random.rand(2, 8, 16).astype(np.float16)),
                       Tensor(np.random.rand(2, 8, 16).astype(np.float16)),
                       Tensor(np.random.rand(2, 8, 16).astype(np.float16)),
                       Tensor(np.random.rand(2, 8, 16).astype(np.float16)),
                       Tensor(np.random.rand(2, 8, 16).astype(np.float16))]}),
]

test_case_quant_ops = [
    ('Quant_1', {
        'block': inner.Quant(0.5, 0.0, False, "Round"),
        'desc_inputs': [Tensor(np.random.rand(1, 2, 4, 4), mstype.float32)],
        'skip': ['backward']}),
    ('Quant_2', {
        'block': inner.Quant(80.0, 10.0, True, "Round"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_3', {
        'block': inner.Quant(80.0, 0.0, False, "Floor"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_4', {
        'block': inner.Quant(80.0, 0.0, False, "Ceil"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_5', {
        'block': inner.Quant(80.0, 0.0, False, "Trunc"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_6', {
        'block': inner.Quant(-80.0, 10.0, False, "Round"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_7', {
        'block': inner.Quant(80.0, -10.0, False, "Round"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float32)],
        'skip': ['backward']}),
    ('Quant_8', {
        'block': inner.Quant(80.0, 10.0, False, "Round"),
        'desc_inputs': [Tensor([100.0, 200.0], mstype.float16)],
        'skip': ['backward']}),
]

test_case_quantum_ops = [
    ('PQC', {
        'block': P.PQC(n_qubits=3,
                       encoder_params_names=['e0', 'e1', 'e2'],
                       ansatz_params_names=['a', 'b', 'c'],
                       gate_names=['RX', 'RX', 'RX', 'npg', 'npg',
                                   'npg', 'RX', 'npg', 'npg', 'RZ',
                                   'npg', 'npg', 'RY'],
                       gate_matrix=[[[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.7071067811865475', '0.7071067811865475'],
                                      ['0.7071067811865475', '-0.7071067811865475']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.7071067811865475', '0.7071067811865475'],
                                      ['0.7071067811865475', '-0.7071067811865475']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.7071067811865475', '0.7071067811865475'],
                                      ['0.7071067811865475', '-0.7071067811865475']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '1.0'], ['1.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '-0.0'], ['0.0', '0.0']],
                                     [['0.0', '-1.0'], ['1.0', '0.0']]],
                                    [[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '1.0'], ['1.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['1.0', '0.0'], ['0.0', '-1.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]],
                                    [[['0.0', '0.0'], ['0.0', '0.0']],
                                     [['0.0', '0.0'], ['0.0', '0.0']]]],
                       gate_obj_qubits=[[0], [1], [2], [0], [1], [2],
                                        [0], [1], [2], [1], [1], [0], [2]],
                       gate_ctrl_qubits=[[], [], [], [], [], [], [], [], [], [], [2], [], []],
                       gate_params_names=[['e0'], ['e1'], ['e2'], [], [], [], ['a'], [], [],
                                          ['b'], [], [], ['c']],
                       gate_coeff=[[1.0], [1.0], [1.0], [], [], [], [1.0], [], [], [1.0], [],
                                   [], [1.0]],
                       gate_requires_grad=[[True], [True], [True], [], [], [], [True], [], [],
                                           [True], [], [], [True]],
                       hams_pauli_coeff=[[1.0]],
                       hams_pauli_word=[[['X', 'Y', 'Z']]],
                       hams_pauli_qubit=[[[0, 1, 2]]],
                       n_threads=1),
        'desc_inputs': [Tensor(np.array([[1.0, 2.0, 3.0]]).astype(np.float32)),
                        Tensor(np.array([2.0, 3.0, 4.0]).astype(np.float32))],
        'skip': ['backward']}),
    ('Evolution', {
        'block': P.Evolution(n_qubits=3,
                             param_names=['a'],
                             gate_names=['npg', 'npg', 'npg', 'RY'],
                             gate_matrix=[[[['0.7071067811865475', '0.7071067811865475'],
                                            ['0.7071067811865475', '-0.7071067811865475']],
                                           [['0.0', '0.0'], ['0.0', '0.0']]],
                                          [[['0.7071067811865475', '0.7071067811865475'],
                                            ['0.7071067811865475', '-0.7071067811865475']],
                                           [['0.0', '0.0'], ['0.0', '0.0']]],
                                          [[['0.0', '1.0'], ['1.0', '0.0']],
                                           [['0.0', '0.0'], ['0.0', '0.0']]],
                                          [[['0.0', '0.0'], ['0.0', '0.0']],
                                           [['0.0', '0.0'], ['0.0', '0.0']]]],
                             gate_obj_qubits=[[0], [1], [0], [0]],
                             gate_ctrl_qubits=[[], [], [1], []],
                             gate_params_names=[[], [], [], ['a']],
                             gate_coeff=[[], [], [], [1.0]],
                             gate_requires_grad=[[], [], [], [True]],
                             hams_pauli_coeff=[[1.0]],
                             hams_pauli_word=[[['Z']]],
                             hams_pauli_qubit=[[[0]]]),
        'desc_inputs': [Tensor(np.array([0.5]).astype(np.float32))],
        'skip': ['backward']}),
]

test_case_lists = [test_case_nn_ops, test_case_math_ops, test_case_array_ops,
                   test_case_other_ops, test_case_quant_ops, test_case_quantum_ops]
test_case = functools.reduce(lambda x, y: x + y, test_case_lists)
# use -k to select certain testcast
# pytest tests/python/ops/test_ops.py::test_backward -k LayerNorm


test_exec_case = test_case

test_backward_exec_case = filter(lambda x: 'skip' not in x[1] or 'backward' not in x[1]['skip'], test_case)


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_exec_case


@mindspore_test(pipeline_for_compile_grad_ge_graph_for_case_by_case_config)
def test_backward_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_backward_exec_case


raise_set = [
    ('Cast_Error', {
        'block': (P.Cast(), {'exception': TypeError}),
        'desc_const': [mstype.int32],
        'desc_inputs': ['wrong input'],
        'desc_bprop': [Tensor(np.ones((2, 3, 3, 5)).astype(np.int32))]}),
    ('Maximum_Error', {
        'block': (P.Maximum(), {'exception': TypeError}),
        'desc_const': [(1, 2, 3)],
        'desc_inputs': [[2, 3, 3, 5]],
        'desc_bprop': [[2, 3, 3, 5]]}),
    ('Shape_error', {
        'block': (P.Shape(), {'exception': TypeError}),
        'desc_inputs': [(64, 1)],
        'desc_bprop': [[64]]}),
    ('Flatten_Error', {
        'block': (NetForFlatten0D(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.array(0).astype(np.int32))],
        'desc_bprop': [Tensor(np.array(0).astype(np.int32))]}),
    ('ScatterNdUpdate', {
        'block': (P.ScatterNdUpdate(), {'exception': TypeError}),
        'desc_inputs': (Tensor(np.ones((2, 3), np.float32)),
                        Tensor(np.ones((2, 2), np.float32)),
                        Tensor(np.ones((2,), np.float32))),
        'desc_bprop': [[2, 3]]}),
    ('PReLU', {
        'block': (P.PReLU(), {'exception': ValueError}),
        'desc_inputs': [[2], [1]],
        'desc_bprop': [[1]]}),
    ('SSIM', {
        'block': (nn.SSIM(), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.ones((1, 3, 8, 8)), mstype.float32),
                        Tensor(np.ones((1, 3, 8, 8)), mstype.float32)]})
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config_exception)
def test_check_exception():
    return raise_set
