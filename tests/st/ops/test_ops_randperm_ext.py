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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.ops.function.random_func import randperm_ext
from mindspore.common import dtype as mstype
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def randperm_ext_forward_func(n, generator=None, dtype=mstype.int64):
    return randperm_ext(n, generator=generator, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_randperm_ext_forward(mode):
    """
    Feature: randperm_ext ops.
    Description: test ops randperm_ext.
    Expectation: generates random permutation of integers from 0 to n-1 without repeating.
    """
    context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        context.set_context(jit_level="O0")

    generator = ms.Generator()
    output1 = randperm_ext_forward_func(4, generator, mstype.int32).asnumpy()
    np.testing.assert_equal(output1.shape, (4,))
    np.testing.assert_equal(output1.dtype, np.int32)

    state = generator.get_state()
    output2 = randperm_ext_forward_func(4, generator, mstype.int32).asnumpy()
    assert np.any(output1 != output2)
    generator.set_state(state)
    output3 = randperm_ext_forward_func(4, generator, mstype.int32).asnumpy()
    np.testing.assert_allclose(output2, output3)
