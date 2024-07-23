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

"""test where"""
import numpy as np
import pytest
import os
import mindspore.common.dtype as mstype

from mindspore.ops import operations as P
from mindspore import nn, Tensor, context, JitConfig
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    np.random.seed(0)
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x1, x2, gamma, beta, eps=1e-5):
    res = x1 + x2
    meanOut = res.mean(1).reshape(2, 1)
    rstdOut = np.power((res.var(1).reshape(2, 1) + eps), -0.5)
    y = rstdOut * (res - meanOut) * gamma + beta
    return y, meanOut, rstdOut, res

class Add_LayerNorm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=-1,
                                     begin_params_axis=-1,
                                     epsilon=1e-5)

    def construct(self, x1, x2, gamma, beta):
        res = x1 + x2
        y, meanOut, rstdOut = self.layernorm(res, gamma, beta)
        return y, meanOut, rstdOut, res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('tensor_type', [mstype.float32, mstype.float16, mstype.bfloat16])
def test_add_layer_norm(tensor_type):
    """
    Feature: test add_layernorm fusion in kbk mode
    Description: test add_layernorm.
    Expectation: the result is the same with aclnn version of two ops
    """
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "AddLayerNorm"
    context.set_context(mode=0)

    x1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((2, 3), np.float32)
    gamma = np.ones([3]).astype(np.float32)
    beta = np.zeros([3]).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    beta_tensor = Tensor(beta, dtype=tensor_type)

    net = Add_LayerNorm()
    net.set_jit_config(JitConfig(jit_level="O0", infer_boost="on"))
    output = net(x1_tensor, x2_tensor, gamma_tensor, beta_tensor)

    expect = generate_expect_forward_output(x1, x2, gamma, beta)
    np.testing.assert_allclose(output[0].float().asnumpy(), expect[0], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[1].float().asnumpy(), expect[1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[2].float().asnumpy(), expect[2], rtol=5e-3, atol=5e-3)

    os.unsetenv("MS_DISABLE_INTERNAL_KERNELS_LIST")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('tensor_type', [mstype.float32, mstype.float16, mstype.bfloat16])
def test_add_layer_norm_dynamic_shape(tensor_type):
    """
    Feature: test add_layernorm fusion with dynamic shape inputs
    Description: test add_layernorm.
    Expectation: the result is the same with aclnn version of two ops
    """
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "AddLayerNorm"
    context.set_context(mode=0)

    x1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((2, 3), np.float32)
    gamma = np.ones([3]).astype(np.float32)
    beta = np.zeros([3]).astype(np.float32)
    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    beta_tensor = Tensor(beta, dtype=tensor_type)
    input_dyn = Tensor(shape=[None, None], dtype=tensor_type)
    gamma_dyn = Tensor(shape=[None], dtype=tensor_type)

    net = Add_LayerNorm()
    net.set_jit_config(JitConfig(jit_level="O0", infer_boost="on"))
    net.set_inputs(input_dyn, input_dyn, gamma_dyn, gamma_dyn)
    output = net(x1_tensor, x2_tensor, gamma_tensor, beta_tensor)

    expect = generate_expect_forward_output(x1, x2, gamma, beta)
    np.testing.assert_allclose(output[0].float().asnumpy(), expect[0], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[1].float().asnumpy(), expect[1], rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(output[2].float().asnumpy(), expect[2], rtol=5e-3, atol=5e-3)

    os.unsetenv("MS_DISABLE_INTERNAL_KERNELS_LIST")
