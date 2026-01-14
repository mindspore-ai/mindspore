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

""" test MatmulQkv BiasAdd SplitWithSize fusion """
import os
import numpy as np
import pytest
from mindspore.common.np_dtype import bfloat16
import mindspore as ms
from mindspore import Tensor, context, ops, nn
from .st_utils import custom_compare, gen_ms_tensor, run_expect_single
from tests.mark_utils import arg_mark


class NetMatmulWithSplit3(nn.Cell):
    """Matmul with split."""

    def __init__(self, weight, bias, n0, n1, n2):
        super().__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.bias_add = ms.ops.Add()
        self.sizes = [n0, n1, n2]
        self.n = n0 + n1 + n2
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.matmul0(x, self.w)
        res1 = self.bias_add(res0, self.bias)
        res2 = self.reshape(res1, out_shape)
        res = self.split_with_size(res2, self.sizes, -1)
        return res


class NetMatmulWithSplit2(nn.Cell):
    """Matmul with split."""

    def __init__(self, weight, bias, n0, n1):
        super().__init__()
        self.matmul0 = ops.MatMul(False, True)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.bias_add = ms.ops.Add()
        self.sizes = [n0, n1]
        self.n = n0 + n1
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.matmul0(x, self.w)
        res1 = self.bias_add(res0, self.bias)
        res2 = self.reshape(res1, out_shape)
        res = self.split_with_size(res2, self.sizes, -1)
        return res


def run_expect_split(x_np, wq_np, wk_np, wv_np, b0_np, b1_np, b2_np):
    res = []
    res.append(run_expect_single(x_np, wq_np, b0_np, False, True))
    res.append(run_expect_single(x_np, wk_np, b1_np, False, True))
    if wv_np is not None:
        res.append(run_expect_single(x_np, wv_np, b2_np, False, True))
    return res


def _test_matmul_qkv(m=0, k=0, n0=0, n1=0, n2=0, mstype=ms.float16, is_dyn=False, profiling=False):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    np_type = None
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.float32 == mstype:
        np_type = np.float32
    elif ms.bfloat16 == mstype:
        np_type = bfloat16

    i0_host = np.random.normal(0.0, 0.5, size=[1, m, k]).astype(np_type)
    i1_host = np.random.normal(0.0, 0.5, size=[n0, k]).astype(np_type)
    i2_host = np.random.normal(0.0, 0.5, size=[n1, k]).astype(np_type)

    b1_host = np.random.normal(0.0, 0.5, size=[n0]).astype(np_type)
    b2_host = np.random.normal(0.0, 0.5, size=[n1]).astype(np_type)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    i2_host_fp32 = i2_host.astype(np.float32)

    b1_host_fp32 = b1_host.astype(np.float32)
    b2_host_fp32 = b2_host.astype(np.float32)

    if n2 != 0:
        i3_host = np.random.normal(0.0, 0.5, size=[n2, k]).astype(np_type)
        i3_host_fp32 = i3_host.astype(np.float32)

        b3_host = np.random.normal(0.0, 0.5, size=[n2]).astype(np_type)
        b3_host_fp32 = b3_host.astype(np.float32)
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32, i3_host_fp32))
        b_host_fp32 = np.concatenate(
            (b1_host_fp32, b2_host_fp32, b3_host_fp32))
    else:
        w_host_fp32 = np.vstack((i1_host_fp32, i2_host_fp32))
        b_host_fp32 = np.concatenate((b1_host_fp32, b2_host_fp32))

    input_np_list = [i0_host_fp32, w_host_fp32, b_host_fp32]
    input_tensor_list = gen_ms_tensor(input_np_list, mstype)

    if n2 != 0:
        net = NetMatmulWithSplit3(
            input_tensor_list[1], input_tensor_list[2], n0, n1, n2)
        output_split = run_expect_split(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, i3_host_fp32, b1_host_fp32, b2_host_fp32, b3_host_fp32)
    else:
        net = NetMatmulWithSplit2(
            input_tensor_list[1], input_tensor_list[2], n0, n1)
        output_split = run_expect_split(
            i0_host_fp32, i1_host_fp32, i2_host_fp32, None, b1_host_fp32, b2_host_fp32, None)
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
        result = result and curr_res

    assert result, "compare correct."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=106.00)
@pytest.mark.parametrize('m', [32])
@pytest.mark.parametrize('k', [1024, 2048, 128])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('is_dyn', [False, True])
def test_matmul_qkv_1408_128_128(m, k, mstype, is_dyn):
    """
    Feature: Test MatmulQkv.
    Description: Test MatmulQkv internal op.
    Expectation: Success.
    """
    _test_matmul_qkv(m, k, 1408, 128, 128, mstype, is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=408.00)
@pytest.mark.parametrize('m', [16, 256, 1024])
@pytest.mark.parametrize('k_n_shape', [(4096, 11008, 11008), (8192, 2752, 2752)])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('is_dyn', [True])
def test_matmul_ffn_0(m, k_n_shape, mstype, is_dyn):
    """
    Feature: Test MatmulBiasSplitOut2.
    Description: Test MatmulBiasSplitOut2 internal op.
    Expectation: Success.
    """
    k, n0, n1 = k_n_shape
    _test_matmul_qkv(m, k, n0, n1, mstype=mstype, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=438.00)
@pytest.mark.parametrize('m', [16, 256, 1024])
@pytest.mark.parametrize('k_n_shape', [(12288, 5376, 5376)])
@pytest.mark.parametrize('mstype', [ms.float16])
@pytest.mark.parametrize('is_dyn', [False])
def test_matmul_ffn_1(m, k_n_shape, mstype, is_dyn):
    """
    Feature: Test MatmulBiasSplitOut2.
    Description: Test MatmulBiasSplitOut2 internal op.
    Expectation: Success.
    """
    k, n0, n1 = k_n_shape
    _test_matmul_qkv(m, k, n0, n1, mstype=mstype, is_dyn=is_dyn)
