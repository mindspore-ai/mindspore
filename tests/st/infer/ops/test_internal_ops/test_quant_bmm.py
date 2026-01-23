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
Test quantized batch matmul operator.
"""
import os
import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore.ops.auto_generate import QuantBatchMatmul
from tests.mark_utils import arg_mark


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


def np_qbmm_compute(a, b, tmp_scale, bias=None, tmp_pertoken_scale=None):
    c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp_scale
    if tmp_pertoken_scale is not None:
        pertoken_scale = tmp_pertoken_scale[:, np.newaxis]
        c = c * pertoken_scale
    c = c.astype(np.float16)
    return c


class Qbmm(ms.nn.Cell):
    def __init__(self, weight, scale, bias, pertoken_scale, trans_a, trans_b, dst_dtype):
        super().__init__()
        self.qbmm = QuantBatchMatmul(
            transpose_x1=trans_a,
            transpose_x2=trans_b,
            dtype=dst_dtype
        )
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = bias
        self.pertoken_scale = pertoken_scale

    def construct(self, x):
        return self.qbmm(x, self.weight, self.scale, None, self.bias, self.pertoken_scale)


class QbmmAdd(ms.nn.Cell):
    def __init__(self, weight, scale, bias, pertoken_scale, trans_a, trans_b, dst_dtype):
        super().__init__()
        self.dbmm = QuantBatchMatmul(transpose_x1=trans_a,
                                     transpose_x2=trans_b,
                                     dtype=dst_dtype)
        self.add = ms.ops.Add()
        self.weight = ms.Parameter(weight, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.bias = bias
        self.pertoken_scale = pertoken_scale

    def construct(self, x):
        return self.add(self.dbmm(x, self.weight, self.scale, None, None, self.pertoken_scale), self.bias)


def qbmm(m, k, n, trans_a=False, trans_b=False, scale_dtype=ms.int64, with_outer_add=False,
         bias_none=False, pertoken_scale_none=True, is_dyn=False):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    # NOTE: 在多卡并行的场景里，要控制随机种子以防多卡numpy生成不一致
    seed = 0
    np.random.seed(seed)
    a = np.random.uniform(-20, 20, size=(m, k)).astype(np.int8)
    np.random.seed(seed)
    b = np.random.uniform(-20, 20, size=(k, n)).astype(np.int8)
    np.random.seed(seed)
    bias = np.random.randint(-10, 10, (n)).astype(np.int32)
    np.random.seed(seed)
    tmp_scale = np.random.rand(n).astype(np.float32) / 1000
    scale = process_deq_scale(tmp_scale)
    bias_fp16 = bias * tmp_scale
    if bias_none:
        bias = None

    np.random.seed(seed)
    tmp_pertoken_scale = np.random.rand(m).astype(np.float32)
    pertoken_scale_ms = ms.Tensor(tmp_pertoken_scale, ms.float32)
    if pertoken_scale_none:
        tmp_pertoken_scale = None
        pertoken_scale_ms = None

    expect_np = np_qbmm_compute(a, b, tmp_scale, bias, tmp_pertoken_scale)

    if trans_a:
        a = np.transpose(a, (1, 0))
    if trans_b:
        b = np.transpose(b, (1, 0))

    a_ms = ms.Tensor(a, ms.int8)
    b_ms = ms.Tensor(b, ms.int8)

    if scale_dtype == ms.float32:
        scale_ms = ms.Tensor(tmp_scale, ms.float32)
    else:
        scale_ms = ms.Tensor(scale, ms.int64)

    net = None
    if not with_outer_add:
        bias_ms = ms.Tensor(bias, ms.int32)
        if bias_none:
            bias_ms = None
        net = Qbmm(b_ms, scale_ms, bias_ms, pertoken_scale_ms, trans_a, trans_b, dst_dtype=ms.float16)
    else:
        bias_ms = ms.Tensor(bias_fp16, ms.float16)
        net = QbmmAdd(b_ms, scale_ms, bias_ms, pertoken_scale_ms, trans_a, trans_b, dst_dtype=ms.float16)

    if is_dyn:
        input_dyn = ms.Tensor(shape=(None, None), dtype=ms.int8)
        net.set_inputs(input_dyn)

    output = net(a_ms)
    output_np = output.asnumpy()
    np.testing.assert_allclose(output_np, expect_np, rtol=0.002, atol=0.002)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=34.00)
@pytest.mark.parametrize('input_shape', [(128, 2560, 5120),
                                         (1024, 1024, 1024)])
def test_qbmm_False_True(input_shape):
    """
    Feature: testqbmm operator in graph mode
    Description: testqbmm.
    Expectation: the result is correct
    """
    m, k, n = input_shape
    qbmm(m, k, n, trans_a=False, trans_b=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=24.00)
@pytest.mark.parametrize('input_shape', [(128, 512, 1024),
                                         (1024, 1024, 1024)])
@pytest.mark.parametrize('tb', [False, True])
@pytest.mark.parametrize('is_dyn', [False, True])
def test_qbmm_add(input_shape, tb, is_dyn):
    """
    Feature: test qbmm operator in graph mode
    Description: test qbmm.
    Expectation: the result is correct
    """
    m, k, n = input_shape
    qbmm(m, k, n, trans_a=False, trans_b=tb,
         with_outer_add=True, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.01)
def test_qbmm_add_continuous():
    """
    Feature: test qbmm operator in graph mode
    Description: test qbmm.
    Expectation: the result is correct
    """
    qbmm(128, 128, 256, trans_a=False, trans_b=True,
         with_outer_add=True, is_dyn=False)
    qbmm(128, 64, 256, trans_a=False, trans_b=True,
         with_outer_add=True, is_dyn=False)
    qbmm(1280, 128, 256, trans_a=False, trans_b=False,
         with_outer_add=True, is_dyn=True)


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=354.00)
@pytest.mark.parametrize('m', [1, 128, 1024])
def test_qbmm_with_pertoken_ds(m):
    """
    Feature: test qbmm operator in graph mode
    Description: test qbmm.
    Expectation: the result is correct
    """
    qbmm(m, 7168, 36864, trans_b=True, bias_none=True, scale_dtype=ms.float32,
         pertoken_scale_none=False)
    qbmm(m, 7168, 18432, trans_b=True, bias_none=True, scale_dtype=ms.float32,
         pertoken_scale_none=False)
