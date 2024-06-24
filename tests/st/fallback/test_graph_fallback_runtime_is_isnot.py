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
""" test graph JIT Fallback runtime is/is not feature """
import pytest
import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_none_asnumpy():
    """
    Feature: Support is.
    Description: Support is in fallback runtime.
    Expectation: No exception.
    """
    @ms.jit
    def foo():
        input_x = Tensor([1], dtype=mstype.int32).asnumpy()
        is_not_res = input_x is not None
        is_res = input_x is None
        return is_not_res, is_res

    ret1, ret2 = foo()
    assert ret1
    assert not ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_int_and_int():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        m = int(x) is 3
        n = int(x) is 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert ret1
    assert not ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_int_and_int_2():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        a = 0
        i = 0
        while i < x:
            i = i + 1
            a = a + 1
        m = a is 3
        n = a is 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert ret1
    assert not ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_int_and_int_3():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        a = 0
        if x > 1:
            a = a + 1
        m = a is 1
        n = a is 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert ret1
    assert not ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_not_int_and_int():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        m = int(x) is not 3
        n = int(x) is not 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert not ret1
    assert ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_not_int_and_int_2():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        a = 0
        i = 0
        while i < x:
            i = i + 1
            a = a + 1
        m = a is not 3
        n = a is not 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert not ret1
    assert ret2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_is_not_int_and_int_3():
    """
    Feature: Is with variable input will be converted to PyExecute node.
    Description: Graph support is/is not syntax
    Expectation: No Error.
    """

    @ms.jit
    def foo(x):
        a = 0
        if x > 1:
            a = a + 1
        m = a is not 1
        n = a is not 2
        return m, n

    ret1, ret2 = foo(Tensor([3]))
    assert not ret1
    assert ret2
