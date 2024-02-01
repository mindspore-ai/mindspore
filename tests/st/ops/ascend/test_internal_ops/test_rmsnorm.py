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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import operations as P
from bfloat16 import bfloat16

class RmsNormNet(nn.Cell):
    def __init__(self, ):
        super(RmsNormNet, self).__init__()
        self.rmsnorm = P.RmsNorm()

    def construct(self, input_x, gamma):
        return self.rmsnorm(input_x, gamma)

class NpRmsNormNet():
    def __init__(self, eps=1e-6):
        self.eps = eps
    
    def norm(self, x, gamma):
        output = x / np.sqrt(np.mean(np.power(x, 2), -1, keepdims=True) + self.eps) * gamma
        return output

def rmsnorm_net(input_params_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    net = RmsNormNet()

    input_np_x = np.random.randn(*input_params_shape)
    input_np_gamma = np.random.randn(input_params_shape[1])
    if dtype == bfloat16:
        output = net(Tensor(input_np_x.astype(np.float16), dtype=mindspore.bfloat16), Tensor(input_np_gamma.astype(np.float16), dtype=mindspore.bfloat16))[0]
        output = ops.cast(output, mindspore.float16).astype(np.float32)
    else:
        output = net(Tensor(input_np_x.astype(dtype)), Tensor(input_np_gamma.astype(dtype)))[0]

    np_net = NpRmsNormNet()
    expected = np_net.norm(input_np_x.astype(dtype), input_np_gamma.astype(dtype)).astype(dtype).astype(np.float32)

    if dtype == np.float32:
        np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=5)
    else:
        np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=1)


def test_rmsnorm_fp16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4096, 128)
    dtype = np.float16
    rmsnorm_net(input_params_shape, dtype)

def test_rmsnorm_fp32():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4096, 128)
    dtype = np.float32
    rmsnorm_net(input_params_shape, dtype)


def test_rmsnorm_bf16():
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    input_params_shape = (4, 1024)
    dtype = bfloat16
    rmsnorm_net(input_params_shape, dtype)
