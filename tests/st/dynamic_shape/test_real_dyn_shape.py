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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def real_forward_func(x):
    return ops.real(x)


@test_utils.run_with_cell
def real_backward_func(x):
    return ops.grad(real_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_real_forward(mode):
    """
    Feature: real ops.
    Description: test ops real.
    Expectation: output the real part of the input.
    """
    context.set_context(mode=mode)
    x = Tensor(np.asarray(np.complex(1.3 + 0.4j)).astype(np.complex64))
    output = real_forward_func(x)
    expect_output = np.asarray(np.complex(1.3)).astype(np.float32)
    np.testing.assert_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_real_backward(mode):
    """
    Feature: real ops.
    Description: test auto grad of ops real.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.asarray(np.complex(1.3 + 0.4j)).astype(np.complex64))
    output = real_backward_func(x)
    expect_output = np.asarray(np.complex(1. + 0j)).astype(np.complex64)
    np.testing.assert_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_real_dynamic(mode):
    """
    Feature: real ops.
    Description: test ops real dynamic tensor input.
    Expectation: output the real part of the input.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.complex64)
    test_cell = test_utils.to_cell_obj(ops.real)
    test_cell.set_inputs(x_dyn)
    x1 = Tensor(np.asarray(np.complex(1.3 + 0.4j)).astype(np.complex64))
    output1 = test_cell(x1)
    expect_output1 = np.asarray(1.3).astype(np.float32)
    np.testing.assert_equal(output1.asnumpy(), expect_output1)
    x2 = Tensor(np.asarray([[np.complex(1.4 + 0.4j), np.complex(2.5 + 0.6j)],
                            [np.complex(3.6 + 0.7j), np.complex(4.7 + 0.8j)]]).astype(np.complex64))
    output2 = test_cell(x2)
    expect_output2 = np.asarray([[1.4, 2.5],
                                 [3.6, 4.7]]).astype(np.float32)
    np.testing.assert_equal(output2.asnumpy(), expect_output2)
