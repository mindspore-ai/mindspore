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
from mindspore import nn, Tensor, context, Parameter
from mindspore import ops
from mindspore.ops.operations._infer_ops import QuantV2

from tests.mark_utils import arg_mark

class Add_RmsNorm(nn.Cell):
    def __init__(self, with_cast=False, with_quant=False, is_internal=True):
        super().__init__()
        self.RmsNorm = ops.RmsNorm(epsilon=1e-5)
        self.with_cast = with_cast
        self.quant = QuantV2()
        self.with_quant = with_quant

        self.t_scale = Parameter(Tensor(np.ones((1,)).astype(
            np.float16), dtype=ms.float16) * 2)
        self.t_zp = Parameter(Tensor(np.ones((1,)).astype(
            np.float16), dtype=ms.float16) * 2)

    def construct(self, x1, x2, gamma):
        res = x1 + x2
        if self.with_cast:
            res = ops.cast(res, ms.float32)
        hidden_states, _ = self.RmsNorm(res, gamma)
        if self.with_cast:
            hidden_states = ops.cast(hidden_states, ms.float16)
        if self.with_quant:
            hidden_states = self.quant(
                hidden_states, self.t_scale, self.t_zp, False, "ROUND", ms.int8)
        return hidden_states, res


def _test_add_rmsnorm_fusion(shape, dtype, internal_kernel, with_cast=False, with_quant=False, is_dynamic=False):
    np.random.seed(0)
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    infer_boost = "on" if internal_kernel else "off"
    context.set_context(mode=0, device_target="Ascend",
                        enable_graph_kernel=False)
    context.set_context(
        jit_config={"jit_level": "O0", "infer_boost": infer_boost})

    np_dtype_map = {"float16": np.float16,
                    "bfloat16": np.float32,
                    "float32": np.float32}
    ms_dtype_map = {"float16": ms.float16,
                    "bfloat16": ms.bfloat16,
                    "float32": ms.float32}
    np_dtype = np_dtype_map[dtype]
    tensor_dtype = ms_dtype_map[dtype]
    gamma_dtype = np.float32 if with_cast else np_dtype
    gamma_ms_dtype = ms.float32 if with_cast else tensor_dtype

    input_x = np.random.rand(*shape).astype(np_dtype)
    input_y = np.random.rand(*shape).astype(np_dtype)
    gamma = np.ones([shape[-1]]).astype(gamma_dtype)
    net = Add_RmsNorm(with_cast, with_quant, internal_kernel)
    if is_dynamic:
        input_dyn = Tensor(shape=[None, None, None], dtype=tensor_dtype)
        gamma_dyn = Tensor(shape=[None], dtype=gamma_ms_dtype)
        net.set_inputs(input_dyn, input_dyn, gamma_dyn)

    output = net(Tensor(input_x, dtype=tensor_dtype), Tensor(input_y, dtype=tensor_dtype),
                 Tensor(gamma, dtype=gamma_ms_dtype))

    return output[0].astype(ms.float32).asnumpy(), output[1].astype(ms.float32).asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype', ["float16", "float32", "bfloat16"])
@pytest.mark.parametrize('is_dynamic', [False])
def test_add_rms_norm_normal(dtype, is_dynamic):
    """
    Feature: test add_rmsnorm fusion in graph mode
    Description: test add_rmsnorm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    result_internal = _test_add_rmsnorm_fusion(
        shape, dtype, True, is_dynamic=is_dynamic)
    result_aclnn = _test_add_rmsnorm_fusion(shape, dtype, False)
    loss = 5e-3
    if dtype == "bfloat16":
        loss = 1e-2
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < loss
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_add_rms_norm_f16_with_cast():
    """
    Feature: test add_rmsnorm fusion with cast in graph mode
    Description: test add_rmsnorm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    dtype = "float16"
    result_internal = _test_add_rmsnorm_fusion(shape, dtype, True, True)
    result_aclnn = _test_add_rmsnorm_fusion(shape, dtype, False)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 5e-3
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_add_rms_norm_f16_quant():
    """
    Feature: test add_rms_norm fusion with quant in graph mode
    Description: test add_rms_norm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    dtype = "float16"
    result_internal = _test_add_rmsnorm_fusion(
        shape, dtype, True, with_quant=True)
    result_aclnn = _test_add_rmsnorm_fusion(
        shape, dtype, False, with_quant=True)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 2
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3
