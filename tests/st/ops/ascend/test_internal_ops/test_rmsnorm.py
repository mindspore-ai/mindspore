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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from bfloat16 import bfloat16


class RmsNormNet(nn.Cell):
    def __init__(self, eps=1e-6):
        super(RmsNormNet, self).__init__()
        self.rmsnorm = ops.RmsNorm(eps)

    def construct(self, input_x, gamma):
        return self.rmsnorm(input_x, gamma)


def compute_np_rmsnorm(x, gamma, eps=1e-6):
    ori_dtype = x.dtype
    x = x.astype(np.float32)
    gamma = gamma.astype(np.float32)
    output = x / np.sqrt(np.mean(np.power(x, 2), -1, keepdims=True) + eps) * gamma
    output = output.astype(ori_dtype)
    return output


def gen_ms_tensor(input_np):
    if input_np.dtype == bfloat16:
        output = Tensor(input_np.astype(np.float32), dtype=ms.bfloat16)
    else:
        output = Tensor(input_np)
    return output


def gen_np_output(output):
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    return output_np


def rmsnorm_net(input_params_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = RmsNormNet()

    input_np_x = np.random.randn(*input_params_shape).astype(dtype)
    input_np_gamma = np.random.randn(input_params_shape[1]).astype(dtype)

    input_x = gen_ms_tensor(input_np_x)
    input_gamma = gen_ms_tensor(input_np_gamma)
    output = net(input_x, input_gamma)[0]
    output_np = gen_np_output(output)

    expected = compute_np_rmsnorm(input_np_x, input_np_gamma)

    np.testing.assert_array_almost_equal(output_np, expected, decimal=5)


def test_rmsnorm_fp16():
    """
    Feature: test rmsnorm operator in graph mode
    Description: test rmsnorm fp16.
    Expectation: the result is correct
    """
    input_params_shape = (4096, 128)
    dtype = np.float16
    rmsnorm_net(input_params_shape, dtype)


def test_rmsnorm_fp32():
    """
    Feature: test rmsnorm operator in graph mode
    Description: test rmsnorm fp32.
    Expectation: the result is correct
    """
    input_params_shape = (4096, 128)
    dtype = np.float32
    rmsnorm_net(input_params_shape, dtype)


def test_rmsnorm_bf16():
    """
    Feature: test rmsnorm operator in graph mode
    Description: test rmsnorm bf16.
    Expectation: the result is correct
    """
    input_params_shape = (4, 1024)
    dtype = bfloat16
    rmsnorm_net(input_params_shape, dtype)
