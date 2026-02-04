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
Test matmul operator.
"""
import os
import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore.common.np_dtype import bfloat16
from .st_utils import custom_compare
from tests.mark_utils import arg_mark

np.set_printoptions(precision=2, suppress=True, linewidth=200)


class MatMulCustom(ms.nn.Cell):
    def __init__(self, weight, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)
        self.weight = ms.Parameter(weight, requires_grad=False)

    def construct(self, i0):
        return self.net(i0, self.weight)


def matmul(m, k, n, trans_a=False, trans_b=False, with_bias=False,
           mstype=ms.float16, profiling=False, net_phase="increment"):
    os.environ['USE_LLM_CUSTOM_MATMUL'] = "off"
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

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
    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    elif trans_a and trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32.T)
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = MatMulCustom(input2, trans_a, trans_b)
    net.phase = net_phase  # 设置成 "prefill" 或者 "increment"

    if profiling:
        for _ in range(50):
            output = net(input1)
        return

    output = net(input1)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(expect, output_np, mstype)
    assert res, "matmul compare fail."


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard', essential_mark='unessential', mem_peak=110.00)
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
def test_matmul_1024_1024_1024_nz_input_fp16(trans_b, mstype):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    matmul(1024, 1024, 1024, trans_a=False, trans_b=trans_b,
           mstype=mstype, net_phase="prefill")


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level0', card_mark='onecard', essential_mark='unessential', mem_peak=110.00)
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
def test_matmul_1024_1234_1234_nz_input_unaligned_k_n_fp16(trans_b, mstype):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    using 1234 just test for unalign 32 scene.
    Expectation: the result is correct
    """
    matmul(1024, 1234, 1234, trans_a=False, trans_b=trans_b,
           mstype=mstype, net_phase="prefill")


class MatMulASD(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)

    def construct(self, i0, i1):
        return self.net(i0, i1)


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


def matmul_asd(m, k, n, trans_a=False, trans_b=False, with_bias=False, mstype=ms.float16, profiling=False):
    os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"
    os.environ['USE_LLM_CUSTOM_MATMUL'] = "off"
    os.environ['INTERNAL_PRINT_TILING'] = "on"
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

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
    if not trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32)
    elif not trans_a and trans_b:
        expect = np.matmul(i0_host_fp32, i1_host_fp32.T)
    elif trans_a and not trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32)
    elif trans_a and trans_b:
        expect = np.matmul(i0_host_fp32.T, i1_host_fp32.T)
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host
    if with_bias:
        bias_host = i1_host.flatten()[:n]
        expect = expect + bias_host

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = MatMulASD(trans_a, trans_b)

    if profiling:
        for _ in range(50):
            output = net(input1, input2)
        return

    output = net(input1, input2)
    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = compare(expect, output_np, mstype)
    assert res, "matmul compare fail."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=110.00)
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
def test_matmul_1024_1024_1024(enable_asd_env_fixture, trans_b, mstype):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    matmul(1024, 1024, 1024, trans_a=False, trans_b=trans_b, mstype=mstype)


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential', mem_peak=110.00)
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('mstype', [ms.float16])
def test_matmul_1024_1024_1024_310p(enable_asd_env_fixture, trans_b, mstype):
    """
    Feature: test matmul operator in graph mode
    Description: test matmul.
    Expectation: the result is correct
    """
    matmul(1024, 1024, 1024, trans_a=False, trans_b=trans_b, mstype=mstype)
