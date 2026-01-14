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

import pytest
import numpy as np
import mindspore as ms
from mindspore.ops import GradOperation
from mindspore import Tensor, nn
from mindspore.mint.nn import GLU
from mindspore.mint.nn.functional import glu
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


_BF16_TOL = dict(rtol=4e-3, atol=4e-3)
_FP32_TOL = {}


@test_utils.run_with_cell
def glu_func(x: Tensor, *args, **kwargs):
    return glu(x, *args, **kwargs)


@test_utils.run_with_cell
def glu_nn(x: Tensor, *args, **kwargs):
    return GLU(*args, **kwargs)(x)


class GluGrad(nn.Cell):
    def __init__(self, net: nn.Cell, sens: Tensor):
        super().__init__()
        self.net = net
        self.grad_op = GradOperation(sens_param=True)
        self.grad_wrt_output = sens

    def construct(self, *args):
        return self.grad_op(self.net)(*args, self.grad_wrt_output)


def _test_glu_forward_main(ms_type, f, tol):
    x0 = np.arange(24).reshape(4, 6) / 12
    expected0 = [
        [0., 0.04854752, 0.10044756],
        [0.33958936, 0.4066179, 0.47624165],
        [0.7772999, 0.8573408, 0.9389512],
        [1.2779291, 1.3650842, 1.4529437]
    ]
    cases = [
        ([], dict(dim=1)),  # named args
        ([1], {}),          # positional arg
    ]
    for args, kwargs in cases:
        x = Tensor(x0.reshape(4, 6, 1), dtype=ms_type)
        expected = Tensor(np.array(expected0).reshape((4, 3, 1)), dtype=ms_type).asnumpy().astype(np.float32)
        assert np.allclose(f(x, *args, **kwargs).asnumpy().astype(np.float32), expected, **tol)
    # default dim=-1
    x = Tensor(x0, dtype=ms_type)
    expected = Tensor(expected0, dtype=ms_type).asnumpy().astype(np.float32)
    assert np.allclose(f(x).asnumpy().astype(np.float32), expected, **tol)


def _test_glu_backward_main(ms_type, f, tol):
    x = Tensor(np.arange(24).reshape(4, 6, 1) / 12, dtype=ms_type)
    dim = 1
    grad = Tensor(np.ones((4, 3, 1)), dtype=ms_type)
    grad_x = GluGrad(f, grad)(x, dim)
    expected_np = np.array([
        [0.5621765, 0.5825702, 0.60268533, 0., 0.02026518, 0.03990929],
        [0.6791787, 0.6970593, 0.71436244, 0.10894749, 0.12318113, 0.13603249],
        [0.7772999, 0.7913915, 0.80481535, 0.17310478, 0.17884858, 0.18326886],
        [0.85195273, 0.8621584, 0.8717662, 0.1891939, 0.18816537, 0.18631646]
    ]).reshape((4, 6, 1))
    expected = Tensor(expected_np, dtype=ms_type).asnumpy().astype(np.float32)
    assert np.allclose(grad_x.asnumpy().astype(np.float32), expected, **tol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential',
        mem_peak=0.01)
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.GRAPH_MODE, 'O0'),  # if Ascend, using aclnn
        (ms.PYNATIVE_MODE, 'PYNATIVE'),  # if Ascend, using aclnn
        (ms.GRAPH_MODE, 'O2'),  # if Ascend, using aicpu
    ]
)
@pytest.mark.parametrize('device', ['CPU', 'Ascend'])
def test_glu(mode, level, device):
    """
    Feature: mint ops.
    Description: test mint.nn.GLU and mint.nn.functional.glu which have AICPU and CPU supports
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level=level)
    ms.context.set_context(mode=mode, device_target=device)

    _test_glu_forward_main(ms.float32, glu_func, _FP32_TOL)
    _test_glu_forward_main(ms.float32, glu_nn, _FP32_TOL)
    _test_glu_backward_main(ms.float32, glu_func, _FP32_TOL)
    _test_glu_backward_main(ms.float32, glu_nn, _FP32_TOL)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_glu_bfloat16(mode):
    """
    Feature: mint ops.
    Description: test mint.nn.GLU and mint.nn.functional.glu with bfloat16 support. (AICPU and CPU not supports)
    Expectation: expect correct result.
    """
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode, device_target="Ascend")

    _test_glu_forward_main(ms.bfloat16, glu_nn, _BF16_TOL)
    _test_glu_backward_main(ms.bfloat16, glu_func, _BF16_TOL)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=3072.0)
def test_glu_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test mint.nn.GLU and mint.nn.functional.glu dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = Tensor(np.random.rand(2, 2, 2, 4), dtype=ms.float32)
    dim1 = -2

    x2 = Tensor(np.zeros((6, 6)), dtype=ms.float32)
    dim2 = 0
    TEST_OP(
        glu_func,
        [
            [x1, dim1],
            [x2, dim2],
        ],
        disable_case=['ScalarTensor']
    )
    TEST_OP(
        GLU(dim=0),
        [[x1], [x2]],
        disable_case=['ScalarTensor']
    )
