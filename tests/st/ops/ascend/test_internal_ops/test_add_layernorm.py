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

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P
import os


class Add_LayerNorm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=-1,
                                     begin_params_axis=-1,
                                     epsilon=1e-5)

    def construct(self, x1, x2, gamma, beta):
        res = x1 + x2
        hidden_states, _, _ = self.layernorm(res, gamma, beta)
        return hidden_states, res


def _layer_norm_np(begin_norm_axis, begin_params_axis, x, gamma, beta):
    begin_norm_axis = begin_norm_axis if begin_norm_axis >= 0 else begin_norm_axis + len(x.shape)
    begin_params_axis = begin_params_axis if begin_params_axis >= 0 else begin_params_axis + len(x.shape)

    axis = [i for i in range(begin_norm_axis, len(x.shape))]
    mean = np.mean(x, axis=tuple(axis), keepdims=True)
    var = np.var(x, axis=tuple(axis), keepdims=True)

    gamma = gamma.reshape((*((1,) * begin_params_axis), *x.shape[begin_params_axis:]))
    beta = beta.reshape((*((1,) * begin_params_axis), *x.shape[begin_params_axis:]))
    y = np.subtract(x, mean) / np.sqrt(var + 1e-5) * gamma + beta
    return y


def _test_add_layernorm_fusion(shape, dtype, internal_kernel):
    np.random.seed(0)
    os.environ["MS_ENABLE_INTERNAL_KERNELS"] = "on" if internal_kernel else "off"
    np_dtype_map = {"float16": np.float16,
                    "bfloat16": np.float32,
                    "float32": np.float32}
    ms_dtype_map = {"float16": ms.float16,
                    "bfloat16": ms.bfloat16,
                    "float32": ms.float32}
    suffix = "internal" if internal_kernel else "aclnn"
    ms.context.set_context(mode=0,
                           device_target="Ascend",
                           save_graphs=3, save_graphs_path="./graphs_" + suffix, enable_graph_kernel=False)
    np_dtype = np_dtype_map[dtype]
    tensor_dtype = ms_dtype_map[dtype]

    input_x = np.random.rand(*shape).astype(np_dtype)
    input_y = np.random.rand(*shape).astype(np_dtype)
    gamma = np.ones([shape[-1]]).astype(np_dtype)
    beta = np.zeros([shape[-1]]).astype(np_dtype)
    net = Add_LayerNorm()
    output = net(Tensor(input_x, dtype=tensor_dtype), Tensor(input_y, dtype=tensor_dtype),
                 Tensor(gamma, dtype=tensor_dtype), Tensor(beta, dtype=tensor_dtype))
    os.unsetenv("MS_ENABLE_INTERNAL_KERNELS")

    return output[0].asnumpy(), output[1].asnumpy()


def test_add_layernorm_f16(dtype=np.float32):
    """
    Feature: test add_layernorm fusion in graph mode
    Description: test add_layernorm.
    Expectation: the result is the same with aclnn version of two ops
    """
    shape = (1, 1024, 11264)
    dtype = "float16"
    result_internal = _test_add_layernorm_fusion(shape, dtype, True)
    result_aclnn = _test_add_layernorm_fusion(shape, dtype, False)
    assert np.amax(np.abs(result_internal[0] - result_aclnn[0])) < 5e-3
    assert np.amax(np.abs(result_internal[1] - result_aclnn[1])) < 5e-3
