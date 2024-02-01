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
""" test Activations internal op """
import numpy as np
from bfloat16 import bfloat16

import mindspore.nn as nn
from mindspore import Tensor, context


class NetSiLU(nn.Cell):
    """SiLU."""
    def __init__(self):
        super(NetSiLU, self).__init__()
        self.silu = nn.SiLU()

    def construct(self, x):
        return self.silu(x)


def gen_ms_tensor(input_np):
    if input_np.dtype == bfloat16:
        output = Tensor(input_np.astype(np.float32), dtype=mindspore.bfloat16)
    else:
        output = Tensor(input_np)
    return output


def gen_np_output(output):
    if output.dtype == mindspore.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    return output_np


def _test_silu(np_dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",)
    net = NetSiLU()

    input_np = np.array([[1.2, 2.1], [2.2, 3.2]], dtype=np_dtype)
    if np_dtype != np.float32:
        input_cast = input_np.astype(np.float32)
        expected = input_cast / (1 + np.exp(-input_cast))
        expected = expected.astype(np_dtype)
    else:
        expected = input_np / (1 + np.exp(-input_np))

    input_data = gen_ms_tensor(input_np)
    output = net(input_data)
    output_np = gen_np_output(output)

    np.testing.assert_array_almost_equal(output_np, expected)


def test_silu_fp16():
    """
    Feature: Test SiLU fp16.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_silu(np.float16)


def test_silu_bf16():
    """
    Feature: Test SiLU bf16.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_silu(bfloat16)
