from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore
from mindspore import JitConfig


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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK'])
def test_dynamic_quant_f16(mode):
    """
    Feature:ops.DynamicQuantExt
    Description: ops.DynamicQuantExt basic
    Expectation: Success
    """
    run_dynamic_quant(mindspore.float16, mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


def random_input(shape, dtype=mindspore.float16):
    return mindspore.Tensor(np.random.randn(*shape), dtype=dtype)
