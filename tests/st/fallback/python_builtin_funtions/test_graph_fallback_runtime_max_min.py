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
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_runtime_max():
    """
    Feature: JIT Fallback
    Description: Test max() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        return max(Tensor([1, 2, 3]).asnumpy())

    out = foo()
    assert out == 3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_runtime_min():
    """
    Feature: JIT Fallback
    Description: Test min() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        return min(Tensor([1, 2, 3]).asnumpy())

    out = foo()
    assert out == 1


@pytest.mark.skip(reason="Invalid call node")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_runtime_max_min():
    """
    Feature: JIT Fallback
    Description: Test max/min() in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        x = max(Tensor([1, 2, 3]).asnumpy())
        y = min(Tensor([1, 2, 3]).asnumpy())
        return x, y

    out = foo()
    assert out[0] == 3, out[1] == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_runtime_max_min_scalar_tensor():
    """
    Feature: JIT Fallback
    Description: Test max(scalar, tensor) and min(scalar, tensor) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo():
        return max(1, Tensor([2])), min(3, Tensor([4]))

    out = foo()
    assert out == (2, 3)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fallback_runtime_max_scalar_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test max(scalar, tensor.asnumpy()) in fallback runtime
    Expectation: No exception
    """

    @jit
    def foo(x):
        return max(1, x.asnumpy())

    with pytest.raises(TypeError) as info:
        x = Tensor([2])
        foo(x)
    assert "Cannot join the return values of different branches" in str(info.value)
