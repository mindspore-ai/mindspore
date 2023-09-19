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
from mindspore import Tensor, jit, context, mutable

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_len_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = [1, 2, 3, x, np.array([1, 2, 3, 4])]
        return len(a), len(x.asnumpy())

    out = foo(Tensor([1, 2, 3, 4]))
    assert out[0] == 5, out[1] == 1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_len_numpy_string():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = np.array([[1, 2, 3], [0, 0, 0]])
        return len(x), len("string")

    out = foo()
    assert out[0] == 2, out[1] == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_len_mutable():
    """
    Feature: JIT Fallback
    Description: Test len() in fallback runtime
    Expectation: No exception
    """
    @jit
    def foo():
        return len(mutable(2))

    with pytest.raises(TypeError) as e:
        foo()
    assert "object of type Int64 has no len()." in str(e.value)
