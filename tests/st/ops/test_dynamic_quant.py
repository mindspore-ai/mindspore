import numpy as np
import pytest

import mindspore
from mindspore import JitConfig

from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


class DynamicQuantCell(mindspore.nn.Cell):
    def __init__(self):
        super(DynamicQuantCell, self).__init__()
        self.quant = mindspore.ops.auto_generate.DynamicQuantExt()

    def construct(self, x, smooth_scales=None):
        return self.quant(x, smooth_scales)


def get_expect(x, smooth_scales):
    if smooth_scales is not None:
        x = x * smooth_scales
    x_abs = np.abs(x)
    x_max = x_abs.max(axis=-1, keepdims=True).astype(np.float32)
    scale = x_max / 127.0
    x = x.astype(np.float32) / scale
    output = np.round(x).astype(np.int8)
    scale = np.squeeze(scale, axis=-1)

    return output, scale


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK'])
def test_dynamic_quant_f16(mode):
    """
    Feature:ops.DynamicQuantExt
    Description: ops.DynamicQuantExt basic
    Expectation: Success
    """
    run_dynamic_quant(mindspore.float16, mode)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', ['GE', 'KBK'])
def test_dynamic_quant_bf16(mode):
    """
    Feature:ops.DynamicQuantExt
    Description: ops.DynamicQuantExt basic
    Expectation: Success
    """
    run_dynamic_quant(mindspore.bfloat16, mode)


def run_dynamic_quant(dtype, mode):
    quant = DynamicQuantCell()
    if mode == 'pynative':
        mindspore.context.set_context(mode=mindspore.PYNATIVE_MODE)
    elif mode == 'GE':
        mindspore.context.set_context(mode=mindspore.GRAPH_MODE)
    elif mode == 'KBK':
        mindspore.context.set_context(mode=mindspore.GRAPH_MODE)
        quant.set_jit_config(JitConfig(jit_level='O0'))
    x = random_input((2, 3, 4), dtype)
    smooth_scales = random_input((4,), dtype)
    x_np = x.float().asnumpy()
    smooth_scales_np = smooth_scales.float().asnumpy()
    output, scale = quant(x, smooth_scales)
    output_expect, scale_expect = get_expect(x_np, smooth_scales_np)
    np.testing.assert_allclose(output.asnumpy(), output_expect, atol=1)
    np.testing.assert_allclose(scale.asnumpy(), scale_expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
def test_op():
    """
    Feature: ops.DynamicQuantExt
    Description: ops.DynamicQuantExt TEST_OP
    Expectation: success
    """
    dynamic_quant_ext_cell = DynamicQuantCell()
    inputs_seq = []
    for shape_x, shape_smooth_scales in [[(4, 4), (4,)], [(8, 8, 8), (8,)]]:
        x = mindspore.Tensor(random_input(shape_x))
        smooth_scales = mindspore.Tensor(random_input(shape_smooth_scales))
        inputs_seq.append([x, smooth_scales])

    # smooth_scales.shape == [x.shape[-1]]
    TEST_OP(dynamic_quant_ext_cell, inputs_seq, 'dynamic_quant_ext',
            disable_grad=True, disable_input_check=True)


def random_input(shape, dtype=mindspore.float16):
    return mindspore.Tensor(np.random.randn(*shape), dtype=dtype)
