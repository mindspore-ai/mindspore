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
from mindspore.ops.operations._infer_ops import QuantV2
from mindspore import context
from mindspore.common.np_dtype import bfloat16


def get_ms_dtype(np_dtype):
    if np_dtype == np.float32:
        ms_dtype = ms.float32
    elif np_dtype == np.float16:
        ms_dtype = ms.float16
    elif np_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


class QuantPerChannel(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.quant_per_channel = QuantV2()

    def construct(self, x, scale, offset):
        y = self.quant_per_channel(x, scale, offset)
        return y

    def golden_calc(self, x, scale, offset, np_dtype=np.float16):
        int8_lower_bound = -128
        input_x = x.asnumpy().astype(np.float32)
        input_scale = scale.asnumpy().astype(np.float32)
        input_offset = offset.asnumpy().astype(np.int8)
        out = np.clip(
            (np.round((input_x / input_scale)) + input_offset),
            int8_lower_bound,
            127,
        )
        return out.astype(np.int8)

    def golden_compare(self, out, golden_out):
        np.testing.assert_allclose(out, golden_out, atol=1)

    def create_inputs(self, shape, shape1, in_dtype=np.float16):
        input0 = np.random.uniform(low=-1000, high=1000, size=shape).astype(in_dtype)
        input1 = np.random.uniform(low=-100, high=100, size=shape1).astype(in_dtype)
        input2 = np.random.uniform(low=-10, high=10, size=shape1).astype(np.int8)
        ms_dtype = get_ms_dtype(in_dtype)
        return (
            ms.Tensor(input0, ms_dtype),
            ms.Tensor(input1, ms_dtype),
            ms.Tensor(input2, ms.int8),
        )


def run(shape, shape1, is_dyn=False, in_dtype=np.float16):
    if "ASCEND_HOME_PATH" not in os.environ:
        os.environ["ASCEND_HOME_PATH"] = "/usr/local/Ascend/cann"
    ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = QuantPerChannel()

    if not is_dyn:
        x, scale, offset = net.create_inputs(shape, shape1, in_dtype)
        res = net(x, scale, offset)
        golden = net.golden_calc(x, scale, offset, in_dtype)
        net.golden_compare(res.asnumpy(), golden)
    else:
        ms_dtype = get_ms_dtype(in_dtype)
        x_dyn = ms.Tensor(shape=[None] * len(shape), dtype=ms_dtype)
        scale_dyn = ms.Tensor(shape=[None] * len(shape1), dtype=ms_dtype)
        offset_dyn = ms.Tensor(shape=[None] * len(shape1), dtype=ms.int8)
        net.set_inputs(x_dyn, scale_dyn, offset_dyn)
        for item in range(1, 6):
            shape = [i + item for i in shape]
            shape1 = [i + item for i in shape1]
            x1, scale1, offset1 = net.create_inputs(shape, shape1, in_dtype)
            res = net(x1, scale1, offset1)
            golden = net.golden_calc(x1, scale1, offset1, in_dtype)
            net.golden_compare(res.asnumpy(), golden)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_dynamic", [False])
@pytest.mark.parametrize("in_dtype", [np.float16])
@pytest.mark.parametrize("shapes", [[(10, 32768), (32768,)]])
def test_quant_per_channel_large(shapes, in_dtype, is_dynamic):
    """
    Feature: test quant_v2 op in kbk enabling infer_boost.
    Description: large input for quant_v2.
    Expectation: the result is correct
    """
    run(shapes[0], shapes[1], is_dynamic, in_dtype)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_dynamic", [False])
@pytest.mark.parametrize("in_dtype", [np.float16])
@pytest.mark.parametrize("shapes", [[(2, 2), (2,)]])
def test_quant_per_channel_small(shapes, in_dtype, is_dynamic):
    """
    Feature: test quant_v2 op in kbk enabling infer_boost.
    Description: small input for quant_v2.
    Expectation: the result is correct
    """
    run(shapes[0], shapes[1], is_dynamic, in_dtype)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_dynamic", [True])
@pytest.mark.parametrize("in_dtype", [np.float16])
@pytest.mark.parametrize("shapes", [[(99, 32), (99, 32)], [(2, 2), (2,)]])
def test_quant_per_channel_dyn(shapes, in_dtype, is_dynamic):
    """
    Feature: test quant_v2 op in kbk enabling infer_boost.
    Description: dynamic shape for quant_v2.
    Expectation: the result is correct
    """
    run(shapes[0], shapes[1], is_dynamic, in_dtype)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("is_dynamic", [False])
@pytest.mark.parametrize("in_dtype", [np.float16])
@pytest.mark.parametrize("shapes", [[(4096, 40), (40,)]])
def test_quant_per_channel_special(shapes, in_dtype, is_dynamic):
    """
    Feature: test quant_v2 op in kbk enabling infer_boost.
    Description: special shape for quant_v2.
    Expectation: the result is correct
    """
    run(shapes[0], shapes[1], is_dynamic, in_dtype)
