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

""" test MatmulQkv internal op """
import os
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, ops
from mindspore.common.np_dtype import bfloat16


class NetMatmulWithSplit3(nn.Cell):
    """Matmul with split."""

    def __init__(self, weight, n0, n1, n2):
        super(NetMatmulWithSplit3, self).__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [n0, n1, n2]
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        res0 = self.matmul0(x, self.w)
        new_shape = (1, self.shape(x)[0], self.w.shape[0])
        res2 = self.reshape(res0, new_shape)
        res = self.split_with_size(res2, self.sizes, -1)
        return res


class NetMatmulWithSplit2(nn.Cell):
    """Matmul with split."""

    def __init__(self, weight, n0, n1):
        super(NetMatmulWithSplit2, self).__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [n0, n1]
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        res0 = self.matmul0(x, self.w)
        new_shape = (1, self.shape(x)[0], self.w.shape[0])
        res2 = self.reshape(res0, new_shape)
        res = self.split_with_size(res2, self.sizes, -1)
        return res


def custom_compare(output, expect, mstype):
    if mstype == ms.float16:
        limit = 0.004
    elif mstype == ms.bfloat16:
        limit = 0.03

    print("limit = ", limit)
    out_flatten = output.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum(np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32)
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        print("[FAILED]", "err_cnt = ", err_cnt, "/", limit_cnt)
        return False

    print("[SUCCESS]", "err_cnt = ", err_cnt, "/", limit_cnt)
    return True


def gen_ms_tensor(input_np_list, mstype):
    input_tensor_list = []
    for input_np in input_np_list:
        input_tensor_list.append(Tensor(input_np, dtype=mstype))
    return input_tensor_list


def run_expect_single(x_np, y_np, trans_a=False, trans_b=True):
    if not trans_a and not trans_b:
        expect = np.matmul(x_np, y_np)
    elif not trans_a and trans_b:
        expect = np.matmul(x_np, y_np.T)
    elif trans_a and not trans_b:
        expect = np.matmul(x_np.T, y_np)
    elif trans_a and trans_b:
        expect = np.matmul(x_np.T, y_np.T)
    return expect


def run_expect_split(x_np, wq_np, wk_np, wv_np):
    res = list()
    res.append(run_expect_single(x_np, wq_np, False, True))
    res.append(run_expect_single(x_np, wk_np, False, True))
    if wv_np is not None:
        res.append(run_expect_single(x_np, wv_np, False, True))
    return res


def _test_matmul_qkv(m=0, k=0, n0=0, n1=0, n2=0, mstype=ms.float16, is_dyn=False, profiling=False):
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

    i0_host = np.random.normal(0.0, 0.5, size=[m, k]).astype(np_type)
    i1_host = np.random.normal(0.0, 0.5, size=[n0, k]).astype(np_type)
    i2_host = np.random.normal(0.0, 0.5, size=[n1, k]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    i2_host_fp32 = i2_host.astype(np.float32)

    if n2 != 0:
        i3_host = np.random.normal(0.0, 0.5, size=[n2, k]).astype(np_type)
        i3_host_fp32 = i3_host.astype(np.float32)
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32, i3_host_fp32))
    else:
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32))

    input_np_list = [i0_host_fp32, w_host_fp32]
    input_tensor_list = gen_ms_tensor(input_np_list, mstype)

    if n2 != 0:
        net = NetMatmulWithSplit3(input_tensor_list[1], n0, n1, n2)
        output_split = run_expect_split(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, i3_host_fp32)
    else:
        net = NetMatmulWithSplit2(input_tensor_list[1], n0, n1)
        output_split = run_expect_split(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, None)
    if is_dyn:
        input_dyn = Tensor(shape=(None, None), dtype=mstype)
        net.set_inputs(input_dyn)

    if profiling:
        for _ in range(50):
            out = net(input_tensor_list[0])
        return

    out = net(input_tensor_list[0])

    assert len(out) == len(output_split)
    result = True
    for _, (out_qkv_i, out_split_i) in enumerate(zip(out, output_split)):
        output_fp32 = out_qkv_i.astype(ms.float32)
        output_np = output_fp32.asnumpy()
        curr_res = custom_compare(output_np,
                                  out_split_i, mstype=mstype)
        result = result and curr_res

    assert result, "compare correct."


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dynamic', [False, True])
@pytest.mark.parametrize('dim_k', [128, 1024, 11264])
def test_matmul_qkv_out_num_3_with_diff_k(ms_dtype, is_dynamic, dim_k):
    """
    Feature: Test MatmulSplitOut3 with different dim_k
    Description: Test MatmulSplitOut3 with different dim_k enabling infer_boost in kbk.
    Expectation: Success.
    """
    _test_matmul_qkv(32, dim_k, 1408, 128, 128, mstype=ms_dtype, is_dyn=is_dynamic)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dynamic', [False, True])
@pytest.mark.parametrize('dim_m', [1, 32, 1024])
def test_matmul_qkv_out_num_3_with_diff_m(ms_dtype, is_dynamic, dim_m):
    """
    Feature: Test MatmulSplitOut3 with different dim_m
    Description: Test MatmulSplitOut3 with different dim_m enabling infer_boost in kbk.
    Expectation: Success.
    """
    _test_matmul_qkv(dim_m, 8192, 1024, 128, 128, mstype=ms_dtype, is_dyn=is_dynamic)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('ms_dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('is_dynamic', [False, True])
@pytest.mark.parametrize('dim_m', [1, 32, 1024])
def test_matmul_qkv_out_num_2_with_diff_m(ms_dtype, is_dynamic, dim_m):
    """
    Feature: Test MatmulSplitOut2 with different dim_m
    Description: Test MatmulSplitOut2 with different dim_m enabling infer_boost in kbk.
    Expectation: Success.
    """
    _test_matmul_qkv(dim_m, 4096, 3584, 3584, mstype=ms_dtype, is_dyn=is_dynamic)
