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
Test transpose batch matmul transpose operator.
"""
import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from .st_utils import custom_compare

import mindspore as ms
from mindspore.ops import operations as P
from mindspore import Tensor, nn, context
from mindspore.common.np_dtype import bfloat16


class TransBMMTransNet(nn.Cell):
    '''TransBMMTransNet for fusion'''

    def __init__(self):
        super().__init__()
        self.trans = P.Transpose()
        self.bmm = P.BatchMatMul(transpose_a=False, transpose_b=False)

    def construct(self, x1, x2, perm1, perm2):
        transpose_in = self.trans(x1, perm1)
        bmm = self.bmm(transpose_in, x2)
        out = self.trans(bmm, perm2)
        return out


class TransBMMTransSWFTNet(nn.Cell):
    '''TransBMMTransNet for fusion'''

    def __init__(self, transpose_a, transpose_b):
        super().__init__()
        self.trans = P.Transpose()
        self.bmm = P.BatchMatMul(
            transpose_a=transpose_a, transpose_b=transpose_b)

    def construct(self, x1, x2, perm1, perm2):
        transpose_in = self.trans(x1, perm1)
        bmm = self.bmm(transpose_in, x2)
        out = self.trans(bmm, perm2)
        return out


class MintTransBMMTransNet(nn.Cell):
    '''MintTransBMMTransNet using mint.transpose and mint.matmul'''

    def __init__(self, transpose_b=False):
        super().__init__()
        self.transpose_b = transpose_b

    def construct(self, x1, x2, dim0, dim1):
        transpose_in = ms.mint.transpose(x1, dim0, dim1)
        if self.transpose_b:
            # For mint.matmul, transpose_b can be achieved by swapping last two dims of weight
            x2 = ms.mint.transpose(x2, -1, -2)
        bmm = ms.mint.matmul(transpose_in, x2)
        out = ms.mint.transpose(bmm, dim0, dim1)
        return out


def trans_bmm_trans_net(b0, b1, m, k, n, mstype=ms.float16, is_dyn=False):
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = "TransposeBatchMatmulTranspose"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    np_type = None
    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.bfloat16 == mstype:
        np_type = bfloat16

    net = TransBMMTransNet()

    if b0 == 0 and b1 != 0:
        perm = (1, 0, 2)
        i0_host = np.random.normal(0.0, 0.5, size=[m, b1, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[b1, k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    elif b0 != 0 and b1 == 0:
        perm = (0, 1, 2)
        i0_host = np.random.normal(0.0, 0.5, size=[b0, m, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    elif b0 == 0 and b1 == 0:
        perm = (0, 1)
        i0_host = np.random.normal(0.0, 0.5, size=[m, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    else:
        perm = (0, 2, 1, 3)
        i0_host = np.random.normal(
            0.0, 0.5, size=[b0, m, b1, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[b1, k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    trans_out = i0_host_fp32.transpose(perm)
    bmm = np.matmul(trans_out, i1_host_fp32)
    expect = bmm.transpose(perm)

    input1 = ms.Tensor(i0_host, mstype)
    input2 = ms.Tensor(i1_host, mstype)
    output = net(input1, input2, perm, perm)

    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(expect, output_np, mstype)
    assert res, "TransposeBatchMatmulTranspose compare fail."


def swft_trans_bmm_trans_net(b, m, k, n, transpose_b=False):
    mstype = ms.float16
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = "TransposeBatchMatmulTranspose"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = TransBMMTransSWFTNet(False, transpose_b)
    a = np.random.uniform(-1, 1, [m, b, k]).astype(np.float16)
    if transpose_b:
        w = np.random.uniform(-1, 1, [b, n, k]).astype(np.float16)
        expect = np.matmul(
            a.transpose(1, 0, 2), w.transpose(0, 2, 1)).transpose(1, 0, 2)
    else:
        w = np.random.uniform(-1, 1, [b, k, n]).astype(np.float16)
        expect = np.matmul(a.transpose(1, 0, 2), w).transpose(1, 0, 2)
    tensor_a = ms.Tensor(a, mstype)
    tensor_w = ms.Tensor(w, mstype)
    perm = (1, 0, 2)
    output = net(tensor_a, tensor_w, perm, perm)
    res = custom_compare(output.astype(ms.float32).numpy(),
                         expect.astype(np.float32), mstype)
    assert res, "TransposeBatchMatmulTranspose compare fail."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=114.0)
@pytest.mark.parametrize('b0', [1, 2])
@pytest.mark.parametrize('b1', [1, 4])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
def test_transpose_batch_matmul_transpose_with_b0_b1(b0, b1, mstype):
    """
    Feature: test transpose operator in graph mode
    Description: test transpose.
    Expectation: the result is correct
    """
    trans_bmm_trans_net(b0, b1, 64, 128, 256, mstype)
    trans_bmm_trans_net(b0, b1, 510, 510, 510, mstype)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=100.0)
@pytest.mark.parametrize('b1', [2, 3])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
def test_transpose_batch_matmul_transpose_with_b1(b1, mstype):
    """
    Feature: test transpose operator in graph mode
    Description: test transpose.
    Expectation: the result is correct
    """
    trans_bmm_trans_net(0, b1, 64, 128, 256, mstype)
    trans_bmm_trans_net(0, b1, 70, 70, 70, mstype)


def mint_trans_bmm_trans_net(b0, b1, m, k, n, transpose_b=False):
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = "TransposeBatchMatmulTranspose"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    net = MintTransBMMTransNet(transpose_b)

    # Choose dims and shapes to match supported patterns: (1,0,2) for 3D and (0,2,1,3) for 4D
    if b0 == 0 and b1 != 0:
        # 3D: (m, b1, k) x (b1, k, n)
        dim0, dim1 = 0, 1
        a = np.random.uniform(-1, 1, [m, b1, k]).astype(np.float16)
        if transpose_b:
            w = np.random.uniform(-1, 1, [b1, n, k]).astype(np.float16)
            expect = np.matmul(a.transpose(1, 0, 2),
                               w.transpose(0, 2, 1)).transpose(1, 0, 2)
        else:
            w = np.random.uniform(-1, 1, [b1, k, n]).astype(np.float16)
            expect = np.matmul(a.transpose(1, 0, 2), w).transpose(1, 0, 2)
    else:
        # 4D: (b0, m, b1, k) x (b1, k, n)
        dim0, dim1 = 1, 2
        a = np.random.uniform(-1, 1, [b0, m, b1, k]).astype(np.float16)
        if transpose_b:
            w = np.random.uniform(-1, 1, [b1, n, k]).astype(np.float16)
            expect = np.matmul(a.transpose(0, 2, 1, 3),
                               w.transpose(0, 2, 1)).transpose(0, 2, 1, 3)
        else:
            w = np.random.uniform(-1, 1, [b1, k, n]).astype(np.float16)
            expect = np.matmul(a.transpose(0, 2, 1, 3),
                               w).transpose(0, 2, 1, 3)

    tensor_a = ms.Tensor(a, ms.float16)
    tensor_w = ms.Tensor(w, ms.float16)
    output = net(tensor_a, tensor_w, dim0, dim1)
    res = custom_compare(output.astype(ms.float32).asnumpy(),
                         expect.astype(np.float32), ms.float16)
    assert res, "TransposeBatchMatmulTranspose (mint) compare fail."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=108.00)
def test_transpose_batch_matmul_transpose_mint():
    """
    Feature: test mint transpose + matmul fusion into TransposeBatchMatmulTranspose
    Description: use mint.transpose and mint.matmul to form the fusion pattern.
    Expectation: the result is correct
    """
    mint_trans_bmm_trans_net(0, 16, 32, 128, 512)
    mint_trans_bmm_trans_net(2, 16, 32, 128, 512)
    mint_trans_bmm_trans_net(2, 32, 32, 512, 128, True)


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=106.00)
def test_transpose_batch_matmul_transpose_swft():
    """
    Feature: test transpose operator in with swft kernels
    Description: test transpose.
    Expectation: the result is correct
    """
    swft_trans_bmm_trans_net(16, 32, 128, 512)
    swft_trans_bmm_trans_net(16, 32, 512, 128, True)
    swft_trans_bmm_trans_net(32, 32, 128, 512)
    swft_trans_bmm_trans_net(32, 32, 512, 128, True)
