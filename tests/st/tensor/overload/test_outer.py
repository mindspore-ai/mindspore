# Copyright 2023 Huawei Technologies Co., Ltd
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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(nn.Cell):

    def construct(self, x, vec2):
        return x.outer(vec2)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def outer_forward_func_dynamic(x, vec2):
    return x.outer(vec2)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_outer(mode):
    """
    Feature: tensor.outer
    Description: Verify the result of tensor.outer
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([7, 8, 9]), ms.int32)
    vec2 = Tensor(np.array([7, 10, 11]), ms.int32)
    net = Net()
    output_x = net(x, vec2)
    expect_x = Tensor(np.array([[49, 70, 77],
                                [56, 80, 88],
                                [63, 90, 99]]), ms.int32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_tensor_outer_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call tensor.outer with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2,), np.float32)
    y1 = generate_random_input((3,), np.float32)
    x2 = generate_random_input((4,), np.float32)
    y2 = generate_random_input((5,), np.float32)
    TEST_OP(
        outer_forward_func_dynamic,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor', 'GradByRequirement'],
        case_config={'disable_input_check': True}
    )
