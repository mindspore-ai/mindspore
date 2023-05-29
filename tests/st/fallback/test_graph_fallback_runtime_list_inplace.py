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
"""Test graph list inplace operation"""
import os
import pytest
import numpy as np

from mindspore import Tensor, jit, context
from mindspore.ops.operations import _sequence_ops as seq

context.set_context(mode=context.GRAPH_MODE)

global_list_1 = [1, 2, 3, 4]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_list_used_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_1

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert id(res) == id(global_list_1)
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


global_numpy_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_numpy_list_used_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_numpy_list

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert id(res) == id(global_numpy_list)
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


global_list_2 = [1, 2, 3, 4, [3, 4], None]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_nested_list_getitem_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2[4]

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert id(res) == id(global_list_2[4])
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_nested_list_return_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert id(res) == id(global_list_2)
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_nested_list_return_in_graph_2():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2, global_list_2[4]

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert len(res) == 2
    assert id(res[0]) == id(global_list_2)
    assert id(res[1]) == id(global_list_2[4])
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


global_list_3 = [1, 2, 3, (4, [3, 4])]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_nested_list_getitem_in_graph_2():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_3[3][1]

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert id(res) == id(global_list_3[3][1])
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_global_nested_list_return_in_graph_3():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_3, global_list_3[3][1]

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo()
    assert len(res) == 2
    assert id(res[0]) == id(global_list_3)
    assert id(res[1]) == id(global_list_3[3][1])
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_local_list():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1, 2, 3, a, b]
        return x

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    input_a = Tensor([1])
    input_b = 2
    ret = foo(input_a, input_b)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), 2]
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.skip(reason="Can not handle value list in make list scene.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_local_list_2():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1, 2, 3, a, b]
        return x

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    input_a = Tensor([1])
    input_b = [1, 2]
    ret = foo(input_a, input_b)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), [1, 2]]
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


global_list_4 = [1, 2]


@pytest.mark.skip(reason="Can not handle value list in make list scene.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_return_local_list_3():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [1, 2, 3, a, global_list_4]
        return x

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    input_a = Tensor([1])
    ret = foo(input_a)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), [1, 2]]
    assert id(ret[4]) == id(global_list_4)
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_local_sequence_used_in_graph_with_operator():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute, should be used as input to ops correctly.
    Expectation: No exception.
    """
    seq_func = seq.SequenceMul()

    @jit
    def foo(x, y):
        list_input = [x, y]
        return seq_func(list_input, 2)

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo(Tensor([1]), Tensor([2]))
    assert isinstance(res, list)
    assert res == [Tensor([1]), Tensor([2]), Tensor([1]), Tensor([2])]
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.skip(reason="Operator do not support nested sequence")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_local_sequence_used_in_graph_with_operator_2():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute, should be used as input to ops correctly.
    Expectation: No exception.
    """
    seq_func = seq.SequenceMul()

    @jit
    def foo(x, y, z):
        list_input = [x, (y, z)]
        return seq_func(list_input, 2)

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    res = foo(Tensor([1]), Tensor([2]), Tensor([3]))
    assert isinstance(res, list)
    assert res == [Tensor([1]), (Tensor([2]), Tensor([3])), Tensor([1]), (Tensor([2]), Tensor([3]))]
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"


@pytest.mark.skip(reason="SequenceCount operator can not handle sequence input with different elements")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_list_count():
    """
    Feature: list count.
    Description: support list count.
    Expectation: No exception.
    """
    @jit
    def list_net_10(aa, bb):
        x = ['a', {'Michael': 1, 'Bob': 'bb', '2': [1, '2']}, aa, bb]
        res = x.count(aa)
        return Tensor(res)

    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "1"
    aa = Tensor(20)
    bb = Tensor(10)
    out = list_net_10(aa, bb)
    assert out == 1
    os.environ["MS_DEV_FALLBACK_SUPPORT_LIST"] = "0"
