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

"""Tests for mint.hstack concatenation and invalid inputs."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint
from mindspore.graph.api import _pynative_executor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def hstack_forward_func(tensor1, tensor2):
    """Call mint.hstack through a cell in PyNative mode."""
    return mint.hstack((tensor1, tensor2))


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('shapes', [((), ()), ((), (2,)), ((2, 1), (2, 3)), ((2, 1, 3), (2, 2, 3))])
def test_mint_hstack_ascend_concat(shapes):
    """
    Feature: mint.hstack Ascend concatenation.
    Description: Verify scalar promotion and concatenation in PyNative mode.
    Expectation: Values, shape and dtype match NumPy horizontal stacking.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    x = np.ones(shapes[0], dtype=np.float32)
    y = np.full(shapes[1], 2, dtype=np.float32)
    output = hstack_forward_func(ms.Tensor(x), ms.Tensor(y))
    expected = np.hstack((x, y))
    assert output.shape == expected.shape
    assert output.dtype == ms.float32
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('sequence_type', [tuple, list])
def test_mint_hstack_cpu_empty(sequence_type):
    """
    Feature: mint.hstack input validation.
    Description: Pass an empty tensor tuple or list in CPU PyNative mode.
    Expectation: ValueError is raised.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    with pytest.raises(ValueError, match="empty"):
        mint.hstack(sequence_type())
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('shapes', [((2,), (2, 2)), ((2, 1), (3, 1))],
                         ids=['different_ranks', 'incompatible_non_concat_axis'])
def test_mint_hstack_cpu_invalid_shapes(shapes):
    """
    Feature: mint.hstack shape validation.
    Description: Pass tensors with different ranks or incompatible non-concatenation dimensions.
    Expectation: ValueError is raised in CPU PyNative mode.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = ms.Tensor(np.ones(shapes[0], dtype=np.float32))
    y = ms.Tensor(np.ones(shapes[1], dtype=np.float32))
    with pytest.raises(ValueError):
        mint.hstack((x, y))
        _pynative_executor.sync()
