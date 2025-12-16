# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test SwiGluReshapeDynQuant internal op """
import numpy as np
import pytest
import mindspore as ms
from mindspore.common.np_dtype import bfloat16
from mindspore import Tensor, context, ops, nn
from tests.mark_utils import arg_mark


class SwiGluReshapeDynQuantNet(nn.Cell):
    """SwiGlu."""

    def __init__(self, shape, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.mul = ms.ops.Mul()
        self.split_with_size = ms.ops.auto_generate.SplitWithSize()
        self.quant = ops.auto_generate.DynamicQuantExt()
        self.reshape = ms.ops.Reshape()
        self.dim = dim
        self.shape = shape

    def construct(self, x, smooth_scales=None):
        _, _, _ = self.shape
        shp = self.shape[self.dim] // 2
        sizes = [shp, shp]
        gate, hidden = self.split_with_size(x, sizes, self.dim)
        gate = self.silu(gate)
        res = self.mul(hidden, gate)
        bs = res.shape[0] * res.shape[1]
        res = self.reshape(res, (bs, shp))
        hidden_states, scale = self.quant(res, smooth_scales)
        return hidden_states, scale


def gen_np_output(output):
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    return output_np


def get_expect(x, np_dtype, dim, shape, smooth_scales=None):
    x0, x1 = np.split(x, 2, axis=dim)
    x0_cast = x0.astype(np.float32)
    x1_cast = x1.astype(np.float32)
    res = x0_cast / (1 + np.exp(-x0_cast))
    res *= x1_cast
    q = res.copy()
    q = q.reshape((shape[0] * shape[1], shape[2] // 2))
    if smooth_scales is not None:
        q = q * smooth_scales
    x_abs = np.abs(q)
    x_max = x_abs.max(axis=-1, keepdims=True).astype(np.float32)
    scale = x_max / 127.0
    q = q.astype(np.float32) / scale
    output = np.round(q).astype(np.int8)
    scale = np.squeeze(scale, axis=-1)
    return output, scale


def _test_swiglu_reshape_dynamic_quant(shape, dim, np_dtype, has_smooth_scale=False, is_dyn=False, prof=False):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    x_np = np.random.normal(0.0, 0.5, size=shape).astype(np_dtype)
    t_shape = shape[dim] // 2
    smooth_scales_np = np.random.randn(t_shape).astype(np_dtype)
    input_data = Tensor(x_np)
    input_scale = Tensor(smooth_scales_np)
    net = SwiGluReshapeDynQuantNet(shape, dim)

    if has_smooth_scale:
        if is_dyn:
            input_dyn = Tensor(
                shape=[None] * len(shape), dtype=input_data.dtype)
            scale_dyn = Tensor(shape=(None,), dtype=input_scale.dtype)
            net.set_inputs(input_dyn, scale_dyn)
        if prof:
            for _ in range(50):
                output, scale = net(input_data, input_scale)
            return
        output, scale = net(input_data, input_scale)
        output_expect, scale_expect = get_expect(
            x_np, np_dtype, dim, shape, smooth_scales_np)
    else:
        if is_dyn:
            input_dyn = Tensor(
                shape=[None] * len(shape), dtype=input_data.dtype)
            net.set_inputs(input_dyn)
        if prof:
            for _ in range(50):
                output, scale = net(input_data)
            return
        output, scale = net(input_data)
        output_expect, scale_expect = get_expect(x_np, np_dtype, dim, shape)
    output_np = gen_np_output(output)
    scale_np = gen_np_output(scale)

    np.testing.assert_allclose(output_np, output_expect, atol=1, rtol=1e-2)
    np.testing.assert_allclose(scale_np, scale_expect, atol=1e-2, rtol=1e-2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(4, 16, 6144), (8, 256, 5502), (3, 4, 32768), (7, 4, 32766)])
@pytest.mark.parametrize('dim', [-1])
@pytest.mark.parametrize('dtype', [bfloat16])
@pytest.mark.parametrize('is_dyn', [False, True])
@pytest.mark.parametrize('has_smooth_scale', [False, True])
def test_swiglu_dynamic_quant_bfloat16(shape, dim, dtype, has_smooth_scale, is_dyn):
    """
    Feature: Test SwiGlu.
    Description: Test SiLU internal op.
    Expectation: Success.
    """
    _test_swiglu_reshape_dynamic_quant(
        shape, dim, dtype, has_smooth_scale, is_dyn)
