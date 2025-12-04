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
import os
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.np_dtype import bfloat16
from mindspore import Tensor, context


class NetSwiGlu(nn.Cell):
    """SwiGlu."""

    def __init__(self, shape, dim):
        super(NetSwiGlu, self).__init__()
        self.silu = nn.SiLU()
        self.mul = ms.ops.Mul()
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.dim = dim
        self.shape = shape

    def construct(self, x):
        shp = self.shape[self.dim] // 2
        sizes = [shp, shp]
        gate, hidden = self.split_with_size(x, sizes, self.dim)
        gate = self.silu(gate)
        res = self.mul(hidden, gate)
        return res


def gen_np_output(output):
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    return output_np


def _test_swiglu(shape, dim, np_dtype, is_dyn):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    input_np = np.random.normal(0.0, 0.5, size=shape).astype(np_dtype)
    in_np_0, in_np_1 = np.split(input_np, 2, axis=dim)

    input_0_cast = in_np_0.astype(np.float32)
    input_1_cast = in_np_1.astype(np.float32)
    expected = input_0_cast / (1 + np.exp(-input_0_cast))
    expected *= input_1_cast
    if np_dtype != bfloat16:
        expected = expected.astype(np_dtype)
    else:
        expected = expected.astype(np.float32)

    input_data = Tensor(input_np)
    net = NetSwiGlu(shape, dim)

    if is_dyn:
        input_dyn = Tensor(shape=[None] * len(shape), dtype=input_data.dtype)
        net.set_inputs(input_dyn)
    output = net(input_data)

    output_np = gen_np_output(output)
    np.testing.assert_array_almost_equal(output_np, expected, decimal=2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('shape', [(1, 6144), (16, 5504)])
@pytest.mark.parametrize('dim', [-1])
@pytest.mark.parametrize('dtype', [np.float32, np.float16, bfloat16])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.env_onecard
def test_swiglu(shape, dim, dtype, is_dyn):
    """
    Feature: Test SwiGlu.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_swiglu(shape, dim, dtype, is_dyn)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('shape', [(1, 3584), (256, 6144), (16, 5376)])
@pytest.mark.env_onecard
def test_swiglu_dyn_shape(shape):
    """
    Feature: Test SwiGlu.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_swiglu(shape, dim=-1, np_dtype=np.float16, is_dyn=True)
