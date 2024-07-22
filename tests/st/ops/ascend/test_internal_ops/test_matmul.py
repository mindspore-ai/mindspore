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

import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore.common.np_dtype import bfloat16

class MatMulCustom(ms.nn.Cell):
    def __init__(self, ta, tb):
        super().__init__()
        self.net = ms.ops.MatMul(ta, tb)

    def construct(self, i0, i1):
        return self.net(i0, i1)


def compare(out, expect, dtype):
    if dtype == ms.float16:
        limit = 0.004
    elif dtype == ms.bfloat16:
        limit = 0.03

    out_flatten = out.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum((np.abs(out_flatten - expect_flatten) / np.abs(expect_flatten) > limit).astype(np.int32))
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        print("[FAILED] err_cnt = ", err_cnt, "/", limit_cnt)
        return False
    print("[SUCCESS] err_cnt = ", err_cnt, "/", limit_cnt)
    return True


def matmul(m, k, n, trans_a=False, trans_b=False, with_bias=False, mstype=ms.float16, is_dyn=False, profiling=False):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

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
    print("numpy compute done")

    input1 = ms.Tensor(i0_host_fp32, mstype)
    input2 = ms.Tensor(i1_host_fp32, mstype)

    net = MatMulCustom(trans_a, trans_b)

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
    res = compare(expect, output_np, mstype)
    assert res, "matmul compare fail."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dynamic', [False, True])
@pytest.mark.parametrize('trans_a', [False, True])
@pytest.mark.parametrize('trans_b', [False, True])
@pytest.mark.parametrize('dim_m', [1, 32, 1024, 4096])
def test_matmul(ms_dtype, is_dynamic, trans_a, trans_b, dim_m):
    """
    Feature: test MatMul
    Description: test matmul.
    Expectation: the result is correct
    """
    matmul(dim_m, 4096, 4096, trans_a=trans_a, trans_b=trans_b, mstype=ms_dtype, is_dyn=is_dynamic)
