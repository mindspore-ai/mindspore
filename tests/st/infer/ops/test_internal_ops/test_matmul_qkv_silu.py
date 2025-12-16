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

""" test MatmulFfn SplitWithSize Silu Fusion internal op """
import os
import numpy as np
import pytest
from mindspore.common.np_dtype import bfloat16

import mindspore as ms
from mindspore import Tensor, context, ops, nn
from .st_utils import custom_compare, gen_ms_tensor, run_expect_single
from tests.mark_utils import arg_mark


class NetMatmulWithSplit2(nn.Cell):
    """Matmul with split and silu."""

    def __init__(self, weight, n0, n1, silu_index):
        super().__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [n0, n1]
        self.n = n0 + n1
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()
        self.silu = nn.SiLU()
        self.silu_index = silu_index

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.matmul0(x, self.w)
        res2 = self.reshape(res0, out_shape)
        res3 = self.split_with_size(res2, self.sizes, -1)
        res = None
        if self.silu_index == 0:
            res = [self.silu(res3[0]), res3[1]]
        elif self.silu_index == 1:
            res = [res3[0], self.silu(res3[1])]
        return res


def run_silu(input_np):
    # input_cast = input_np.astype(np.float32)
    expected = input_np / (1 + np.exp(-input_np))
    # expected = expected.astype(np_dtype)
    return expected


def run_expect_split(x_np, wq_np, wk_np, wv_np, silu_index=None):
    res = []
    res.append(run_expect_single(x_np, wq_np, trans_a=False, trans_b=True))
    res.append(run_expect_single(x_np, wk_np, trans_a=False, trans_b=True))
    if silu_index == 0:
        res = [run_silu(res[0]), res[1]]
    elif silu_index == 1:
        res = [res[0], run_silu(res[1])]
    return res


def _test_matmul_qkv(m=0, k=0, n0=0, n1=0, n2=0, silu_index=None, mstype=ms.float16, is_dyn=False, profiling=False):
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = "InferenceMatmulSplitSilu"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    np_type = None
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16

    i0_host = np.random.normal(0.0, 0.1, size=[1, m, k]).astype(np_type)
    i1_host = np.random.normal(0.0, 0.1, size=[n0, k]).astype(np_type)
    i2_host = np.random.normal(0.0, 0.1, size=[n1, k]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    i2_host_fp32 = i2_host.astype(np.float32)

    if n2 != 0:
        i3_host = np.random.normal(0.0, 0.1, size=[n2, k]).astype(np_type)
        i3_host_fp32 = i3_host.astype(np.float32)

        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32, i3_host_fp32))
    else:
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32))

    input_np_list = [i0_host_fp32, w_host_fp32]
    input_tensor_list = gen_ms_tensor(input_np_list, mstype)

    net = NetMatmulWithSplit2(input_tensor_list[1], n0, n1, silu_index)
    output_split = run_expect_split(
        i0_host_fp32, i1_host_fp32, i2_host_fp32, None, silu_index)
    if is_dyn:
        input_dyn = Tensor(shape=(None, None, None), dtype=mstype)
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
        curr_res = custom_compare(output_np, out_split_i, mstype=mstype)
        # print("=======output_np=========")
        # print(output_np)
        # print("=======output_split_i=========")
        # print(out_split_i)
        result = result and curr_res

    assert result, "compare correct."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('m', [1, 32])
@pytest.mark.parametrize('k', [1024])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('silu_index', [0])
@pytest.mark.parametrize('is_dyn', [False])
def test_matmul_ffn_3584_3584(m, k, mstype, silu_index, is_dyn):
    """
    Feature: Test MatmulFfn.
    Description: Test MatmulFfn internal op.
    Expectation: Success.
    """
    _test_matmul_qkv(m=m, k=k, n0=3584, n1=3584, n2=0,
                     silu_index=silu_index, mstype=mstype, is_dyn=is_dyn)
