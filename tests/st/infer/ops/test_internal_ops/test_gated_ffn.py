# Copyright 2024 Huawei Technologies Co., Ltd
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

"""
Tests for gated feed-forward network operator.
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, Tensor, context, ops
from mindspore.profiler import ProfilerLevel, AicoreMetrics
from mindspore.ops.auto_generate import QuantBatchMatmul
from .st_utils import gen_ms_tensor, run_expect_single
from tests.mark_utils import arg_mark


class NetGatedFFN(nn.Cell):
    """Matmul with split and silu on one side, then mul."""

    def __init__(self, weight, outn, silu_index):
        super().__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.mul = ops.Mul()
        self.w = ms.Parameter(weight, requires_grad=False)
        # self.split = ms.ops.Split(axis=-1, output_num=2)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [outn, outn]
        self.n = outn + outn
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()
        self.silu = nn.SiLU()
        self.silu_index = silu_index

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.matmul0(x, self.w)
        res2 = self.reshape(res0, out_shape)
        # res3 = self.split(res2)
        res3 = self.split_with_size(res2, self.sizes, -1)
        if self.silu_index == 0:
            res = self.mul(self.silu(res3[0]), res3[1])
        elif self.silu_index == 1:
            res = self.mul(res3[0], self.silu(res3[1]))
        else:
            res = self.mul(self.silu(res3[0]), res3[1])
        return res


class NetGatedFFN2(nn.Cell):
    """Matmul with split and silu on the first output, fastgelu on the second,
       add the two and then mul with the third output of the split"""

    def __init__(self, weight, outn, silu_index):
        super().__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.mul = ops.Mul()
        self.w = ms.Parameter(weight, requires_grad=False)
        # self.split = ms.ops.Split(axis=-1, output_num=3)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [outn, outn, outn]
        self.n = 3 * outn
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()
        self.silu = nn.SiLU()
        self.silu_index = silu_index
        self.fastgelu = ms.ops.FastGeLU()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.matmul0(x, self.w)
        res2 = self.reshape(res0, out_shape)
        # res3 = self.split(res2)
        res3 = self.split_with_size(res2, self.sizes, -1)
        res4 = self.silu(res3[0])
        res4 += self.fastgelu(res3[1])
        return self.mul(res4, res3[2])

#use for QuantBatchMatMul
class NetQGatedFFN(nn.Cell):
    def __init__(self, split_size, weight, scale, bias, outn, ta, tb, dst_dtype):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=ta,
                                     transpose_x2=tb,
                                     dtype=dst_dtype)
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)
        self.split_size = split_size
        if split_size == 2:
            self.sizes = [outn, outn]
            self.n = outn + outn
        else:
            self.sizes = [outn, outn, outn]
            self.n = outn + outn + outn

        self.mul = ops.Mul()
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()
        self.silu = nn.SiLU()
        self.fastgelu = ms.ops.FastGeLU()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.dbmm(x, self.weight, self.scale, None, self.bias)
        res2 = self.reshape(res0, out_shape)
        res3 = self.split_with_size(res2, self.sizes, -1)
        if self.split_size == 2:
            return self.mul(self.silu(res3[0]), res3[1])
        res4 = self.silu(res3[0])
        res4 += self.fastgelu(res3[1])
        return self.mul(res4, res3[2])


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


def np_qbmm_compute(a, b, tmp, bias=None):
    c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp
    c = c.astype(np.float16)
    return c


def run_silu_np(input_np):
    expected = input_np / (1 + np.exp(-input_np))
    return expected


def run_fastgelu_np(x):
    abs_x = np.absolute(x)
    sub1 = np.subtract(x, abs_x)
    mul1 = np.multiply(sub1, 0.851)
    exp1 = np.exp(mul1)
    tmp1 = np.multiply(x, exp1)
    mul2 = np.multiply(abs_x, -1.702)
    exp2 = np.exp(mul2)
    tmp2 = np.add(exp2, 1.0)
    res = np.divide(tmp1, tmp2)
    return res


def run_expect_split_mul(x_np, wq_np, wk_np, wv_np, silu_index=None):
    res = []
    res.append(run_expect_single(x_np, wq_np, trans_a=False, trans_b=True))
    res.append(run_expect_single(x_np, wk_np, trans_a=False, trans_b=True))
    if silu_index == 0:
        res = [run_silu_np(res[0]) * res[1]]
    elif silu_index == 1:
        res = [res[0] * run_silu_np(res[1])]
    return res


def run_expect_split3_mul(x_np, wq_np, wk_np, wv_np, silu_index=None):
    w1res = run_expect_single(x_np, wq_np, trans_a=False, trans_b=True)
    w11res = run_expect_single(x_np, wk_np, trans_a=False, trans_b=True)
    w3res = run_expect_single(x_np, wv_np, trans_a=False, trans_b=True)
    res = [(run_silu_np(w1res) + run_fastgelu_np(w11res)) * w3res]
    return res


def run_expect_qsplit_mul(x_np, w1_np, w3_np, w11_np, scale, bias=None, silu_index=None, split_size=2):
    size = int(scale.shape[0] / split_size)
    bias1 = 0
    bias2 = 0
    bias3 = 0
    scale1 = None
    scale2 = None
    scale3 = None
    if split_size == 2:
        scale1 = scale[0:size]
        scale2 = scale[size:]
        if bias is not None:
            bias1 = bias[0:size]
            bias2 = bias[size:]
    elif split_size == 3:
        scale1 = scale[0:size]
        scale2 = scale[size:size*2]
        scale3 = scale[size*2:]
        if bias is not None:
            bias1 = bias[0:size]
            bias2 = bias[size:size*2]
            bias3 = bias[size*2:]

    w1res = (np.dot(x_np.astype(np.float32),
                    np.transpose(w1_np.astype(np.float32), (1, 0))) + bias1) * scale1
    w3res = (np.dot(x_np.astype(np.float32),
                    np.transpose(w3_np.astype(np.float32), (1, 0))) + bias2) * scale2

    if w11_np is None:
        if silu_index == 0:
            res = run_silu_np(w1res) * w3res
            return res.astype(np.float16)
        if silu_index == 1:
            res = w1res * run_silu_np(w3res)
            return res.astype(np.float16)
        return None

    w11res = (np.dot(x_np.astype(np.float32),
                     np.transpose(w11_np.astype(np.float32), (1, 0))) + bias3) * scale3
    res = (run_silu_np(w1res) + run_fastgelu_np(w3res)) * w11res
    return res.astype(np.float16)


def _test_gated_ffn(m=0, k=0, outn=0, split_into_two=True, silu_index=0, is_dyn=False, profiling=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    mstype = ms.float16
    np_type = np.float16
    np.random.seed(0)
    i0_host = np.random.normal(0., 0.1, size=[1, m, k]).astype(np_type)
    i1_host = np.random.normal(0., 0.1, size=[outn, k]).astype(np_type)
    i2_host = np.random.normal(0., 0.1, size=[outn, k]).astype(np_type)
    i3_host = np.random.normal(0., 0.1, size=[outn, k]).astype(np_type)
    # i0_host[:,:,:] = np.expand_dims(i0_host[:,1,:],axis=1)
    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    i2_host_fp32 = i2_host.astype(np.float32)
    i3_host_fp32 = i3_host.astype(np.float32)

    if split_into_two:
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32))
        input_np_list = [i0_host_fp32, w_host_fp32]
        input_tensor_list = gen_ms_tensor(input_np_list, mstype)
        net = NetGatedFFN(input_tensor_list[1], outn, silu_index)
        output_split = run_expect_split_mul(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, None, silu_index)
    else:
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32, i3_host_fp32))
        input_np_list = [i0_host_fp32, w_host_fp32]
        input_tensor_list = gen_ms_tensor(input_np_list, mstype)
        net = NetGatedFFN2(input_tensor_list[1], outn, silu_index)
        output_split = run_expect_split3_mul(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, i3_host_fp32, silu_index)

    if is_dyn:
        input_dyn = Tensor(shape=(None, None, None), dtype=mstype)
        net.set_inputs(input_dyn)
    # warm up
    for _ in range(10):
        out = net(input_tensor_list[0])
    test_loops = 1
    if profiling:
        test_loops = 50
        profiler = ms.Profiler(
            start_profile=True, profiler_level=ProfilerLevel.Level1,
            aicore_metrics=AicoreMetrics.ArithmeticUtilization)
    # start = time.time()  # Not used in this function
    for _ in range(test_loops):
        out = net(input_tensor_list[0])
    if profiling:
        profiler.stop()
        profiler.analyse()
    assert len(out) == len(output_split)
    for out_qkv_i, out_split_i in zip(out, output_split):
        output_fp32 = out_qkv_i.astype(ms.float32)
        output_np = output_fp32.asnumpy()
        acc = 0.01
        assert np.allclose(output_np, out_split_i, acc, acc)


def _test_qgated_ffn(m=0, k=0, outn=0, split_into_two=True, silu_index=0, is_dyn=False, profiling=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    np.random.seed(0)
    ms_type = ms.float16

    i0_host = np.random.randint(-5, 5, size=[1, m, k]).astype(np.int8)
    i1_host = np.random.randint(-5, 5, size=[outn, k]).astype(np.int8)

    i2_host = np.random.randint(-5, 5, size=[outn, k]).astype(np.int8)
    i3_host = np.random.randint(-5, 5, size=[outn, k]).astype(np.int8)
    bias = np.random.randint(-5, 5, size=[outn * 3]).astype(np.int32)
    tmp = np.random.rand(outn*3).astype(np.float32) / 1000

    w_host_i8 = np.vstack((i1_host, i2_host, i3_host))
    if split_into_two:
        i3_host = None
        w_host_i8 = np.vstack((i1_host, i2_host))
        bias = np.random.randint(-5, 5, size=[outn * 2]).astype(np.int32)
        tmp = np.random.rand(outn*2).astype(np.float32) / 1000

    scale = process_deq_scale(tmp)
    w_ms = Tensor(w_host_i8, ms.int8)
    net = NetQGatedFFN(
        2 if split_into_two else 3, w_ms, scale, bias, outn, False, True, ms_type)
    output_split = run_expect_qsplit_mul(
        i0_host, i1_host, i2_host, i3_host, tmp, bias, silu_index,
        2 if split_into_two else 3)
    if is_dyn:
        input_dyn = Tensor(shape=(None, None, None), dtype=ms.int8)
        net.set_inputs(input_dyn)

    test_loops = 1
    if profiling:
        test_loops = 50

    for _ in range(test_loops):
        out = net(Tensor(i0_host, ms.int8))
    # print("testing results")
    assert np.allclose(out, output_split, 0.01, 0.01)
    # print("test end")


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('m', [1])
@pytest.mark.parametrize('k', [4096])
def test_gated_ffn_quant2_0(m, k):
    """
    Feature: Test GatedFFN.
    Description: Test GatedFFN internal op.
    Expectation: Success.
    """
    _test_qgated_ffn(m=m, k=k, outn=5504, split_into_two=True, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('m', [4096])
@pytest.mark.parametrize('k', [4096])
def test_gated_ffn_quant2_1(m, k):
    """
    Feature: Test GatedFFN.
    Description: Test GatedFFN internal op.
    Expectation: Success.
    """
    _test_qgated_ffn(m=m, k=k, outn=5504, split_into_two=True, profiling=False)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('m', [1, 4096])
@pytest.mark.parametrize('k', [4096])
def test_gated_ffn_split2_float16_310P_0(m, k):
    """
    Feature: Test GatedFFN.
    Description: Test GatedFFN internal op.
    Expectation: Success.
    """
    _test_gated_ffn(m=m, k=k, outn=3584, split_into_two=True, profiling=False)
