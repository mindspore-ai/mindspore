# Copyright 2024 Huawei Technomulies Co., Ltd
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
"""ST tests for ops.meshgrid behavior and correctness."""
import pytest
import numpy as np
import mindspore as ms
from mindspore.mint import meshgrid
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = meshgrid

    def construct(self, tensors, indexing):
        return self.func(tensors, indexing=indexing)

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(*tensors, indexing='ij'):
    return np.meshgrid(*tensors, indexing=indexing)


def meshgrid_forward_func(tensors, indexing='ij'):
    return meshgrid(*tensors, indexing=indexing)


def meshgrid_forward_func2(tensors, indexing='ij'):
    return meshgrid(tensors, indexing=indexing)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_meshgrid_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function mul forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((128,), np.float32)
    y = generate_random_input((4096,), np.float32)
    output1 = meshgrid_forward_func((ms.Tensor(x), ms.Tensor(y)), indexing='ij')
    output2 = meshgrid_forward_func2((ms.Tensor(x), ms.Tensor(y)), indexing='ij')
    expect_out = generate_expect_forward_output(x, y, indexing='ij')
    for ms_out, np_out in zip(output1, expect_out):
        np.testing.assert_allclose(ms_out.asnumpy(), np_out, rtol=1e-4)
    for ms_out, np_out in zip(output2, expect_out):
        np.testing.assert_allclose(ms_out.asnumpy(), np_out, rtol=1e-4)

    x2 = generate_random_input((4,), np.float32)
    y2 = generate_random_input((2,), np.float32)
    output3 = meshgrid_forward_func([ms.Tensor(x2), ms.Tensor(y2)], indexing='xy')
    output4 = meshgrid_forward_func2([ms.Tensor(x2), ms.Tensor(y2)], indexing='xy')
    expect_out2 = generate_expect_forward_output(x2, y2, indexing='xy')
    for ms_out, np_out in zip(output3, expect_out2):
        np.testing.assert_allclose(ms_out.asnumpy(), np_out, rtol=1e-4)
    for ms_out, np_out in zip(output4, expect_out2):
        np.testing.assert_allclose(ms_out.asnumpy(), np_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.01)
def test_ops_mul_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2,), np.float32)
    y1 = generate_random_input((3,), np.float32)
    x2 = generate_random_input((4,), np.float32)
    y2 = generate_random_input((5,), np.float32)

    TEST_OP(meshgrid_forward_func, [[[ms.Tensor(x1), ms.Tensor(y1)]], [[ms.Tensor(x2), ms.Tensor(y2)]]],
            case_config={'disable_input_check': True,
                         'disable_grad': True})
