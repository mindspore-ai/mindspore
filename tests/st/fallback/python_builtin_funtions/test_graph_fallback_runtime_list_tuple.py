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
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_list_tuple_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test list() and tuple() in fallback runtime
    Expectation: No exception.
    """

    @jit
    def foo(x):
        a = list((1, x, np.array([5, 6]), x.asnumpy()))
        b = tuple((1, x, np.array([5, 6]), x.asnumpy()))
        return a, b

    out = foo(Tensor([2, 3]))
    assert isinstance(out[0], list)
    assert isinstance(out[1], tuple)

    assert out[0][0] == 1
    assert (out[0][1] == Tensor([2, 3])).all()
    assert (out[0][2] == np.array([5, 6])).all()
    assert (out[0][3] == np.array([2, 3])).all()

    assert out[1][0] == 1
    assert (out[1][1] == Tensor([2, 3])).all()
    assert (out[1][2] == np.array([5, 6])).all()
    assert (out[1][3] == np.array([2, 3])).all()
