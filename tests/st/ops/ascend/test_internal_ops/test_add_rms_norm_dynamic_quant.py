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

"""test add rms norm dynamic quant"""

import os
import numpy as np
import pytest

# should be inited before importing mindspore
from op_checker import InternalOpEnabledChecker
op_checker = InternalOpEnabledChecker({'MS_SUBMODULE_LOG_v': '{DEVICE:1}'}, True, "./add_rmsnorm_dynamic_quant_log")

import mindspore as ms
from mindspore import nn, Tensor, context, Parameter
from mindspore import ops
from tests.mark_utils import arg_mark


KEYWORD = "kernel opname:AddRmsNormDynamicQuant, kernel type:internal_kernel"


class AddRmsNormNet(nn.Cell):
    """AddRmsNormNet"""
    def __init__(self, is_internal, shape, np_dtype, ms_dtype, has_smooth_scale):
        super().__init__()
        self.is_internal = is_internal

        self.RmsNorm = ops.RmsNorm(epsilon=1e-5)
        self.quant = ops.auto_generate.DynamicQuantExt()
        self.t_scale = Parameter(Tensor(np.ones((shape[-1])).astype(np_dtype), dtype=ms_dtype) * 2)

        self.shape = shape
        self.has_smooth_scale = has_smooth_scale

    def construct(self, x1, x2, gamma):
        """construct"""
        res = x1 + x2
        hidden_states, _ = self.RmsNorm(res, gamma)
        out_shape = hidden_states.shape[:-1] + (self.shape[-1],)
        hidden_states = ops.reshape(hidden_states, (-1, self.shape[-1]))

        if self.has_smooth_scale:
            hidden_states, scale = self.quant(hidden_states, self.t_scale)
        else:
            hidden_states, scale = self.quant(hidden_states, None)
        hidden_states = ops.reshape(hidden_states, out_shape)
        return hidden_states, scale, res


def _test_add_rms_norm_dynamic_quant_fusion(shape, dtype, has_smooth_scale, internal_kernel):
    """test"""
    np.random.seed(0)
    infer_boost = "on" if internal_kernel else "off"
    context.set_context(mode=0, device_target="Ascend", enable_graph_kernel=False)
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": infer_boost})
    os.environ['MS_SYNC_RUN'] = "on"
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"

    np_dtype_map = {"float16": np.float16,
                    "bfloat16": np.float32,
                    "float32": np.float32}
    ms_dtype_map = {"float16": ms.float16,
                    "bfloat16": ms.bfloat16,
                    "float32": ms.float32}
    np_dtype = np_dtype_map[dtype]
    tensor_dtype = ms_dtype_map[dtype]

    input_x = np.ones(shape).astype(np_dtype)
    input_y = np.ones(shape).astype(np_dtype)
    gamma = np.ones([shape[-1]]).astype(np_dtype)
    net = AddRmsNormNet(internal_kernel, shape, np_dtype, tensor_dtype, has_smooth_scale)

    dyn_shape = [None] * len(shape)
    input_dyn = Tensor(shape=dyn_shape, dtype=tensor_dtype)
    gamma_dyn = Tensor(shape=[None], dtype=tensor_dtype)
    net.set_inputs(input_dyn, input_dyn, gamma_dyn)

    output = net(Tensor(input_x, dtype=tensor_dtype), Tensor(input_y, dtype=tensor_dtype),
                 Tensor(gamma, dtype=tensor_dtype))

    if internal_kernel:
        assert op_checker.CheckOpExistByKeyword(KEYWORD)
    else:
        assert op_checker.CheckOpNotExistByKeyword(KEYWORD)

    return output[0].astype(ms.float32).asnumpy(), output[1].astype(ms.float32).asnumpy(), \
        output[2].astype(ms.float32).asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("shape", [(1, 93, 6144), (2, 1, 3, 4, 5, 6, 2, 2)])
@pytest.mark.parametrize('dtype', ["bfloat16", "float16"])
@pytest.mark.parametrize('has_smooth_scale', [True, False])
def test_add_rms_norm_dynamic_quant(shape, dtype, has_smooth_scale):
    """
    Feature: test add_rms_norm_dynamic_quant fusion in graph mode
    Description: test add_rms_norm_dynamic_quant.
    Expectation: the result is the same with aclnn version of the ops
    """
    internal_quant, internal_scale, internal_add_res =\
        _test_add_rms_norm_dynamic_quant_fusion(shape, dtype, has_smooth_scale, True)

    expect_quant, expect_scale, expect_add_res =\
        _test_add_rms_norm_dynamic_quant_fusion(shape, dtype, has_smooth_scale, False)

    assert np.amax(np.abs(internal_quant - expect_quant)) <= 1
    assert np.amax(np.abs(internal_scale - expect_scale)) <= 5e-2
    assert np.amax(np.abs(internal_add_res - expect_add_res)) <= 5e-2
