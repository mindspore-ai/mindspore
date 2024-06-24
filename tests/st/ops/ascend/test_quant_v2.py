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
from tests.mark_utils import arg_mark


"""test quant_v2"""
import numpy as np
import pytest
import mindspore.common.dtype as mstype

from mindspore.ops.operations._infer_ops import QuantV2
from mindspore import Tensor, jit, JitConfig
from tests.st.utils import test_utils


def generate_random_input(shape, dtype, tensor_type):
    np.random.seed(0)
    return Tensor(np.random.randn(*shape).astype(dtype), dtype=tensor_type)


def generate_expect_output(data, scale, offset, round_mode):
    if round_mode == "ROUND":
        out = np.around(data * scale + offset)
    elif round_mode == "FLOOR":
        out = np.floor(data * scale + offset)
    elif round_mode == "CEIL":
        out = np.ceil(data * scale + offset)
    elif round_mode == "TRUNC":
        out = np.trunc(data * scale + offset)
    else:
        out = np.around(data * scale + offset)
    return out.astype(np.int8)


@test_utils.run_with_cell
def quant_forward_func(data, scale, offset, sqrt_mode, round_mode, out_type):
    net = QuantV2()
    return net(data, scale, offset, sqrt_mode, round_mode, out_type)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
@pytest.mark.parametrize('rounding', ['ROUND', 'FLOOR', 'CEIL', 'TRUNC'])
@pytest.mark.parametrize('support_type', [mstype.float32, mstype.float16, mstype.bfloat16])
def test_quant_static_shape(mode, rounding, support_type):
    """
    Feature: Test quant_v2 with static shape in graph and pynative mode.
    Description: call ops.quant_v2 with valid input and index.
    Expectation: return the correct value.
    """
    np.random.seed(1)
    x = generate_random_input((2, 3, 4, 5), np.float32, support_type)
    scale = generate_random_input((5,), np.float32, support_type)
    offset = generate_random_input((5,), np.float32, support_type)

    if mode == 'pynative':
        ms_out = quant_forward_func(x, scale, offset, False, rounding, mstype.int8)
    elif mode == 'KBK':
        ms_out = (jit(quant_forward_func, jit_config=JitConfig(jit_level="O0")))\
            (x, scale, offset, False, rounding, mstype.int8)
    else:
        ms_out = (jit(quant_forward_func, jit_config=JitConfig(jit_level="O2")))\
            (x, scale, offset, False, rounding, mstype.int8)

    if support_type == mstype.bfloat16:
        expect = \
            generate_expect_output(x.float().asnumpy(), scale.float().asnumpy(), offset.float().asnumpy(), rounding)
    else:
        expect = generate_expect_output(x.asnumpy(), scale.asnumpy(), offset.asnumpy(), rounding)
    np.testing.assert_allclose(ms_out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('rounding', ['ROUND', 'FLOOR', 'CEIL', 'TRUNC'])
def test_quant_dynamic_shape(rounding):
    """
    Feature: Test quant_v2 with dynamic shape in graph mode.
    Description: call ops.quant_v2 with valid input and index.
    Expectation: return the correct value.
    """
    np.random.seed(1)
    x = generate_random_input((2, 3, 4, 5), np.float32, mstype.float32)
    scale = generate_random_input((5,), np.float32, mstype.float32)
    offset = generate_random_input((5,), np.float32, mstype.float32)

    x_dyn = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    scale_dyn = Tensor(shape=[None], dtype=mstype.float32)
    offset_dyn = Tensor(shape=[None], dtype=mstype.float32)

    test_cell = test_utils.to_cell_obj(quant_forward_func)
    test_cell.set_inputs(x_dyn, scale_dyn, offset_dyn, False, rounding, mstype.int8)
    ms_out = test_cell(x, scale, offset, False, rounding, mstype.int8)

    expect = generate_expect_output(x.asnumpy(), scale.asnumpy(), offset.asnumpy(), rounding)
    np.testing.assert_allclose(ms_out.asnumpy(), expect)
