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
Tests for add_rms_norm operator.
"""
import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, Tensor, context, Parameter
from mindspore import ops
from mindspore.ops.operations._infer_ops import QuantV2
from tests.mark_utils import arg_mark


class AddRmsNormNet(nn.Cell):
    def __init__(self, with_cast=False, with_quant=False, is_internal=True,
                 with_rms_norm_out=False, dtype=ms.float16, last_dim=1, is_aclnn_quant_mode=False):
        # why last_dim: the latest ATB does not support scale and offset with shape {1}
        super().__init__()
        self.with_cast = with_cast
        self.with_quant = with_quant
        self.is_internal = is_internal
        self.with_rms_norm_out = with_rms_norm_out

        self.rmsnorm = ops.RmsNorm(epsilon=1e-5)
        self.quant = QuantV2()

        if is_internal:
            if is_aclnn_quant_mode:
                self.t_zp = Parameter(
                    Tensor(np.ones((last_dim)).astype(np.float32), dtype=dtype) * 2)
                self.t_scale = Parameter(
                    1.0 / (Tensor(np.ones((last_dim)).astype(np.float32), dtype=dtype) * 2))
            else:
                self.t_zp = Parameter(
                    Tensor(np.ones((last_dim)).astype(np.int8), dtype=ms.int8) * 2)
                self.t_scale = Parameter(
                    Tensor(np.ones((last_dim)).astype(np.float32), dtype=dtype) * 2)
        else:
            self.t_zp = Parameter(
                Tensor(np.ones((last_dim)).astype(np.float32), dtype=dtype) * 2)
            self.t_scale = Parameter(
                1.0 / (Tensor(np.ones((last_dim)).astype(np.float32), dtype=dtype) * 2))

    def construct(self, x1, x2, gamma):
        res = x1 + x2
        if self.with_cast:
            res = ops.cast(res, ms.float32)
        hidden_states, _ = self.rmsnorm(res, gamma)
        if self.with_rms_norm_out:
            rms_norm_out = hidden_states + 0
        if self.with_cast:
            hidden_states = ops.cast(hidden_states, ms.float16)
        if self.with_quant:
            hidden_states = self.quant(
                hidden_states, self.t_scale, self.t_zp, False, "ROUND", ms.int8)

        if self.with_rms_norm_out:
            return hidden_states, res, rms_norm_out
        return hidden_states, res


def _test_add_rms_norm_fusion(
        shape, dtype, internal_kernel, with_cast=False, with_quant=False, is_dynamic=False, with_rms_norm_out=False,
        is_aclnn_quant_mode=False
):
    np.random.seed(0)
    infer_boost = "on" if internal_kernel else "off"
    context.set_context(mode=0, device_target="Ascend",
                        enable_graph_kernel=False)
    context.set_context(
        jit_config={"jit_level": "O0", "infer_boost": infer_boost})
    if internal_kernel is True:
        context.set_context(
            graph_kernel_flags="--enable_pass=add_rms_norm_quant_fusion")
    else:
        context.set_context(
            graph_kernel_flags="--disable_pass=add_rms_norm_quant_fusion")

    np_dtype_map = {"float16": np.float16,
                    "bfloat16": np.float32, "float32": np.float32}
    ms_dtype_map = {"float16": ms.float16,
                    "bfloat16": ms.bfloat16, "float32": ms.float32}
    np_dtype = np_dtype_map[dtype]
    tensor_dtype = ms_dtype_map[dtype]
    gamma_dtype = np.float32 if with_cast else np_dtype
    gamma_ms_dtype = ms.float32 if with_cast else tensor_dtype

    input_x = np.random.rand(*shape).astype(np_dtype)
    input_y = np.random.rand(*shape).astype(np_dtype)
    gamma = np.ones([shape[-1]]).astype(gamma_dtype)
    net = AddRmsNormNet(with_cast, with_quant, internal_kernel, with_rms_norm_out, tensor_dtype, shape[-1],
                        is_aclnn_quant_mode)
    if is_dynamic:
        input_dyn = Tensor(shape=[None, None, None], dtype=tensor_dtype)
        gamma_dyn = Tensor(shape=[None], dtype=gamma_ms_dtype)
        net.set_inputs(input_dyn, input_dyn, gamma_dyn)

    output = net(
        Tensor(input_x, dtype=tensor_dtype), Tensor(
            input_y, dtype=tensor_dtype), Tensor(gamma, dtype=gamma_ms_dtype)
    )

    if with_rms_norm_out:
        return (
            output[0].astype(ms.float32).asnumpy(),
            output[1].astype(ms.float32).asnumpy(),
            output[2].astype(ms.float32).asnumpy(),
        )
    return output[0].astype(ms.float32).asnumpy(), output[1].astype(
        ms.float32).asnumpy()


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("is_dynamic", [False])
def test_add_rms_norm_normal(dtype, is_dynamic):
    """
    Feature: test add_rms_norm fusion in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    result_internal = _test_add_rms_norm_fusion(
        shape, dtype, True, is_dynamic=is_dynamic)
    result_aclnn = _test_add_rms_norm_fusion(shape, dtype, False)

    assert np.allclose(result_internal[0], result_aclnn[0], 0.001, 0.001)
    assert np.allclose(result_internal[1], result_aclnn[1], 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_add_rms_norm_f16_with_cast():
    """
    Feature: test add_rms_norm fusion with cast in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    dtype = "float16"
    result_internal = _test_add_rms_norm_fusion(shape, dtype, True, True)
    result_aclnn = _test_add_rms_norm_fusion(shape, dtype, False)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 5e-3
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend310p'],
          level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_add_rms_norm_f16_quant_with_cast():
    """
    Feature: test add_rms_norm fusion with quant and cast in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    dtype = "float16"
    result_internal = _test_add_rms_norm_fusion(
        shape, dtype, True, with_quant=True, with_cast=True)
    result_aclnn = _test_add_rms_norm_fusion(
        shape, dtype, False, with_quant=True)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 2
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("shape", [(1, 1024, 4096), (1, 1024, 11264)])
def test_add_rms_norm_bf16_quant(shape):
    """
    Feature: test add_rms_norm fusion with quant for bf16 in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    dtype = "bfloat16"
    result_internal = _test_add_rms_norm_fusion(
        shape, dtype, True, with_quant=True)
    result_aclnn = _test_add_rms_norm_fusion(
        shape, dtype, False, with_quant=True)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 2
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("shape", [(1, 15, 7168)])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_add_rms_norm_quant_rms_norm_out(shape, dtype):
    """
    Feature: test add_rms_norm fusion with quant for bf16 in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """

    result_internal = _test_add_rms_norm_fusion(
        shape, dtype, True, with_quant=True, with_rms_norm_out=True)
    result_aclnn = _test_add_rms_norm_fusion(
        shape, dtype, False, with_quant=True, with_rms_norm_out=True)
    assert np.allclose(result_internal[0], result_aclnn[0], 0.001, 1)
    assert np.allclose(result_internal[1], result_aclnn[1], 0.001, 0.001)
    assert np.allclose(result_internal[2], result_aclnn[2], 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("shape", [(1, 15, 7168), (1, 1, 20000)])
def test_add_rms_norm_aclnn_quant(dtype, shape):
    """
    Feature: test add_rms_norm fusion with quant in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    result_internal = _test_add_rms_norm_fusion(
        shape, dtype, True, with_quant=True, is_aclnn_quant_mode=True)
    result_aclnn = _test_add_rms_norm_fusion(
        shape, dtype, False, with_quant=True)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 2
    assert np.allclose(result_internal[1], result_aclnn[1], 0.001, 0.001)
