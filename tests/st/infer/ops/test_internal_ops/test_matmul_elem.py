# Copyright 2025 Huawei Technologies Co., Ltd
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
Test matmul elementwise operator.
"""
from tests.mark_utils import arg_mark
from mindspore.common.np_dtype import bfloat16
from mindspore import context
import mindspore as ms
import pytest
import numpy as np
import os
os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "MatMulElemwise"


class MatMulCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)

    def construct(self, i0, i1):
        return self.net(i0, i1)


class MatMulReluCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.relu = ms.ops.ReLU()

    def construct(self, i0, i1):
        return self.relu(self.net(i0, i1))


class MatMulBiasaddCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.bias = ms.ops.Add()

    def construct(self, i0, i1, i2):
        return self.bias(self.net(i0, i1), i2)


class MatMulGeluCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.gelu = ms.ops.GeLU()

    def construct(self, i0, i1):
        return self.gelu(self.net(i0, i1))


class MatMulFastGeluCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.fast_gelu = ms.ops.FastGeLU()

    def construct(self, i0, i1):
        return self.fast_gelu(self.net(i0, i1))


class MatMulSigmoidCastBiasAddCustom(ms.nn.Cell):
    def __init__(self, ta, tb, mstype):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.sigmoid = ms.ops.sigmoid
        self.bias = ms.ops.Add()
        self.cast = ms.ops.Cast()
        self.mstype = mstype

    def construct(self, i0, i1, i2):
        return self.cast(self.bias(self.sigmoid(self.net(i0, i1)), i2), self.mstype)


def compare(out, expect, dtype):
    limit = 0.0
    if dtype == ms.float16:
        limit = 0.004
    elif dtype == ms.bfloat16:
        limit = 0.03

    out_flatten = out.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum((np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32))
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        # print("[FAILED] err_cnt = ", err_cnt, "/", limit_cnt)
        return False
    # print("[SUCCESS] err_cnt = ", err_cnt, "/", limit_cnt)
    return True


def matmul_int8(i0_host, i1_host, scale, m, k, n, trans_a=False, trans_b=False, profiling=False):
    i1_host = np.transpose(i1_host.astype(np.float32), (1, 0))

    expect = None
    if not trans_a and not trans_b:
        expect = np.dot(i0_host.astype(np.float32),
                        i1_host.astype(np.float32)) * scale
    elif not trans_a and trans_b:
        expect = np.dot(i0_host.astype(np.float32),
                        np.transpose(i1_host.astype(np.float32), (1, 0))) * scale
    elif trans_a and not trans_b:
        expect = np.dot(np.transpose(i0_host.astype(np.float32), (1, 0)),
                        i1_host.astype(np.float32)) * scale
    elif trans_a and trans_b:
        expect = np.dot(np.transpose(i0_host.astype(np.float32), (1, 0)),
                        np.transpose(i1_host.astype(np.float32), (1, 0))) * scale
    return expect


def matmul(m, k, n, trans_a=False, trans_b=False, mstype=ms.float16, profiling=False):
    np_type = None
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16
    if trans_a:
        i0_host = np.random.normal(0.0, 0.5, size=[k, m]).astype(np_type)
    else:
        i0_host = np.random.normal(0.0, 0.5, size=[m, k]).astype(np_type)

    if trans_b:
        i1_host = np.random.normal(0.0, 0.5, size=[n, k]).astype(np_type)
    else:
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)

    expect = None
    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    elif trans_a and trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32.T)

    return i0_host_fp32, i1_host_fp32, expect, np_type


def gelu_np(x, dtype=np.float32):
    x = x.astype(np.float32)
    y = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi)
                   * (x + 0.044715 * x * x * x)))
    return y.astype(dtype)


def relu_np(x):
    return np.maximum(0, x)


def fast_gelu_np(x):
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


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def matmul_biasadd(m, k, n, trans_a=False, trans_b=False, mstype=ms.float16, bias_type=ms.float16,
                   is_dyn=False, with_sigmoid=False, profiling=False, net_phase="prefill"):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    i0_host_fp32, i1_host_fp32, expect, np_type = matmul(
        m, k, n, trans_a, trans_b, mstype, profiling)
    i2_host = np.ones([n,]).astype(np_type) * 100
    i2_host_fp32 = i2_host.astype(np.float32)
    if with_sigmoid:
        expect = sigmoid_np(expect)
    expect = expect + i2_host_fp32

    if bias_type != ms.float32:
        bias_type = mstype

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)
    input3 = ms.Tensor(i2_host_fp32, bias_type)

    if with_sigmoid:
        net = MatMulSigmoidCastBiasAddCustom(trans_a, trans_b, mstype)
        net.phase = net_phase  # "prefill" or "increment"
    else:
        net = MatMulBiasaddCustom(trans_a, trans_b)
    if is_dyn:
        input_dyn = ms.Tensor(shape=(None, None), dtype=mstype)
        input_1d_dyn = ms.Tensor(shape=(None), dtype=bias_type)
        net.set_inputs(input_dyn, input_dyn, input_1d_dyn)

    if profiling:
        for _ in range(50):
            output = net(input1, input2, input3)
        return

    output = net(input1, input2, input3)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = compare(expect, output_np, mstype)
    assert res, "matmul_biasadd compare fail."


def matmul_unary(m, k, n, trans_a=False, trans_b=False, mstype=ms.float16,
                 elemtype="gelu", is_dyn=False, profiling=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    i0_host_fp32, i1_host_fp32, expect, _ = matmul(
        m, k, n, trans_a, trans_b, mstype, profiling)

    if elemtype == "gelu":
        expect = gelu_np(expect)
    elif elemtype == "relu":
        expect = relu_np(expect)
    elif elemtype == "fastgelu":
        expect = fast_gelu_np(expect)
    else:
        raise ValueError("unknown elemtype = {}".format(elemtype))

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = None
    if elemtype == "gelu":
        net = MatMulGeluCustom(trans_a, trans_b)
    elif elemtype == "relu":
        net = MatMulReluCustom(trans_a, trans_b)
    elif elemtype == "fastgelu":
        net = MatMulFastGeluCustom(trans_a, trans_b)
    else:
        raise ValueError("unknown elemtype = {}".format(elemtype))

    if is_dyn:
        input_dyn = ms.Tensor(shape=(None, None), dtype=mstype)
        net.set_inputs(input_dyn, input_dyn)

    if profiling:
        for _ in range(50):
            output = net(input1, input2)
        return

    output = net(input1, input2)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    assert np.allclose(output_np, expect, 0.01, 0.01)


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_matmul_biasadd_1024_1024_1024_False_False_float16():
    """
    Feature: test matmul_biasadd operator in graph mode
    Description: test matmul_biasadd.
    Expectation: the result is correct
    """
    matmul_biasadd(1024, 1024, 1024, trans_a=False,
                   trans_b=False, mstype=ms.float16)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('m', [256, 1024])
@pytest.mark.parametrize('n', [64, 256, 512, 1000])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dyn', [False, True])
def test_matmul_sigmoid_biasadd_m_7168_n_False_True(m, n, mstype, is_dyn):
    """
    Feature: test matmul_biasadd operator in graph mode
    Description: test matmul_biasadd.
    Expectation: the result is correct
    """
    matmul_biasadd(m, 7168, n, trans_a=False, trans_b=True, mstype=mstype,
                   is_dyn=is_dyn, with_sigmoid=True, net_phase="prefill")
