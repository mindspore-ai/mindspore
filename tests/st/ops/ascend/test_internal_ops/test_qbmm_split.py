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
import pytest
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.ops.auto_generate import QuantBatchMatmul
from st_utils import custom_compare


class QbmmWithSplit3(ms.nn.Cell):
    def __init__(self, weight, bias, scale, n0, n1, n2, dst_dtype):
        super(QbmmWithSplit3, self).__init__()
        self.qbmm = QuantBatchMatmul(transpose_x1=False,
                                     transpose_x2=True,
                                     dtype=dst_dtype)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [n0, n1, n2]
        self.n = n0 + n1 + n2
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.qbmm(x, self.w, self.scale, None, self.bias)
        res1 = self.reshape(res0, out_shape)
        res = self.split_with_size(res1, self.sizes, -1)
        return res


class QbmmWithSplit2(ms.nn.Cell):
    def __init__(self, weight, bias, scale, n0, n1, dst_dtype):
        super(QbmmWithSplit2, self).__init__()
        self.qbmm = QuantBatchMatmul(transpose_x1=False,
                                     transpose_x2=True,
                                     dtype=dst_dtype)
        self.w = ms.Parameter(weight, requires_grad=False)
        self.bias = ms.Parameter(bias, requires_grad=False)
        self.scale = ms.Parameter(scale, requires_grad=False)
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.sizes = [n0, n1]
        self.n = n0 + n1
        self.reshape = ms.ops.Reshape()
        self.shape = ms.ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.n,)
        x = self.reshape(x, (-1, self.shape(x)[2]))
        res0 = self.qbmm(x, self.w, self.scale, None, self.bias)
        res1 = self.reshape(res0, out_shape)
        res = self.split_with_size(res1, self.sizes, -1)
        return res


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)


def np_qbmm_compute(a, b, tmp, bias=None):
    b = np.transpose(b, (1, 0))
    c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
    if bias is not None:
        c = c + bias
    c = c.astype(np.float32) * tmp
    c = c.astype(np.float16)
    return c


def run_expect_split(x_np, wq_np, wk_np, wv_np, b0_np, b1_np, b2_np, s0_np, s1_np, s2_np):
    res = list()
    res.append(np_qbmm_compute(x_np, wq_np, s0_np, b0_np))
    res.append(np_qbmm_compute(x_np, wk_np, s1_np, b1_np))
    if wv_np is not None:
        res.append(np_qbmm_compute(x_np, wv_np, s2_np, b2_np))
    return res


def qbmm_split(m, k, n0=0, n1=0, n2=0, is_dyn=False, profiling=False):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    out_dtype = ms.float16
    x0 = np.random.uniform(-127, 127, size=(1, m, k)).astype(np.int8)
    w0 = np.random.uniform(-127, 127, size=(n0, k)).astype(np.int8)
    w1 = np.random.uniform(-127, 127, size=(n1, k)).astype(np.int8)

    bias0 = np.random.randint(-127, 127, (n0)).astype(np.int32)
    bias1 = np.random.randint(-127, 127, (n1)).astype(np.int32)

    ori_scale0 = np.random.rand(n0).astype(np.float32) / 100  # float32
    ori_scale1 = np.random.rand(n1).astype(np.float32) / 100  # float32

    scale0 = process_deq_scale(ori_scale0)  # int64
    scale1 = process_deq_scale(ori_scale1)  # int64

    if n2 != 0:
        w2 = np.random.uniform(-127, 127, size=(n2, k)).astype(np.int8)
        bias2 = np.random.randint(-127, 127, (n2)).astype(np.int32)
        ori_scale2 = np.random.rand(n2).astype(np.float32) / 100  # float32
        scale2 = process_deq_scale(ori_scale2)  # int64
        weight = np.vstack((w0, w1, w2))
        bias = np.concatenate((bias0, bias1, bias2))
        scale = np.concatenate((scale0, scale1, scale2))
    else:
        weight = np.vstack((w0, w1))
        bias = np.concatenate((bias0, bias1))
        scale = np.concatenate((scale0, scale1))

    x_ms = ms.Tensor(x0, ms.int8)
    weight_ms = ms.Tensor(weight, ms.int8)
    bias_ms = ms.Tensor(bias, ms.int32)
    scale_ms = ms.Tensor(scale, ms.int64)

    if n2 != 0:
        net = QbmmWithSplit3(weight_ms, bias_ms, scale_ms, n0, n1, n2, dst_dtype=out_dtype)
        expect_output = run_expect_split(x0, w0, w1, w2, bias0, bias1, bias2, ori_scale0, ori_scale1, ori_scale2)
    else:
        net = QbmmWithSplit2(weight_ms, bias_ms, scale_ms, n0, n1, dst_dtype=out_dtype)
        expect_output = run_expect_split(x0, w0, w1, None, bias0, bias1, None, ori_scale0, ori_scale1, None)

    if is_dyn:
        input_dyn = Tensor(shape=(None, None, None), dtype=ms.int8)
        net.set_inputs(input_dyn)

    if profiling:
        for _ in range(50):
            output = net(x_ms)
        return

    output = net(x_ms)
    result = True
    for _, (out_i, expect_out_i) in enumerate(zip(output, expect_output)):
        output_np = out_i.asnumpy()
        curr_res = custom_compare(output_np, expect_out_i, ms.float16)
        result = result and curr_res
    assert result, "qbmm split compare fail."


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('m', [32])
@pytest.mark.parametrize('k', [1024, 11264, 128])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_qbmm_qkv_1408_128_128(m, k, is_dyn):
    """
    Feature: Test QbmmQkv.
    Description: Test QbmmQkv internal op.
    Expectation: Success.
    """
    qbmm_split(m, k, 1408, 128, 128, is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('m', [16, 1024])
@pytest.mark.parametrize('k', [4096])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_qbmm_qkv_11008_4096_4096(m, k, is_dyn):
    """
    Feature: Test MatmulQkv.
    Description: Test MatmulQkv internal op.
    Expectation: Success.
    """
    qbmm_split(m, k, 11008, 4096, 4096, is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('m', [32, 256])
@pytest.mark.parametrize('k', [8192])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_qbmm_ffn_3584_3584(m, k, is_dyn):
    """
    Feature: Test QbmmFfn.
    Description: Test QbmmFfn internal op.
    Expectation: Success.
    """
    qbmm_split(m=m, k=k, n0=3584, n1=3584, is_dyn=is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('m', [1, 16, 256, 1024])
@pytest.mark.parametrize('k_n_shape', [(8192, 1024, 128, 128), (8192, 2048, 256, 256),
                                       (4096, 4096, 4096, 4096), (12288, 1536, 1536, 1536)])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_qbmm_qkv(m, k_n_shape, is_dyn):
    """
    Feature: Test MatmulQkv.
    Description: Test MatmulQkv internal op.
    Expectation: Success.
    """
    k, n0, n1, n2 = k_n_shape
    qbmm_split(m, k, n0, n1, n2, is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('m', [1, 16, 256, 1024])
@pytest.mark.parametrize('k_n_shape', [(4096, 11008, 11008), (8192, 2752, 2752),
                                       (12288, 5376, 5376)])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_qbmm_ffn(m, k_n_shape, is_dyn):
    """
    Feature: Test MatmulFfn.
    Description: Test MatmulFfn internal op.
    Expectation: Success.
    """
    k, n0, n1 = k_n_shape
    qbmm_split(m, k, n0, n1, is_dyn=is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('input_shape', [(1, 8192, 1024, 128, 128),
                                         (32, 4096, 4096, 4096, 4096),
                                         (16, 12288, 1536, 1536, 1536)])
@pytest.mark.env_onecard
def test_dynamic_shape(input_shape):
    """
    Feature: Test MatmulQkv.
    Description: Test MatmulQkv internal op.
    Expectation: Success.
    """
    m, k, n0, n1, n2 = input_shape
    qbmm_split(m, k, n0, n1, n2, is_dyn=True)
