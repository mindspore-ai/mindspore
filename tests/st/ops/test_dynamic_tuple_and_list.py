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
import mindspore.ops as ops
from mindspore import jit, mutable, Tensor, dtype as mstype
from tests.mark_utils import arg_mark
import pytest

def hypermap_add_in_dynamic(para_a, para_b, para_c):
    """Add function."""
    return para_a + para_b + para_c

@jit(backend="ms_backend")
def hypermap_in_dynamic(para_a, para_b, para_c, para_d):
    """Multiply tuple by mutable and add again."""
    return ops.HyperMap()(hypermap_add_in_dynamic, para_a * para_d, para_b * para_d, para_c * para_d)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_hypermap_in_dynamic_normal_tuple():
    """
    Feature: hypermap of dynamic tuple
    Description: Test the result of hypermap of dynamic tuple
    Expectation: Expectation: Output is equal to the expected output
    """
    tuple_a = (1, 2)
    tuple_b = (1, 2)
    tuple_c = (1, 2)
    mutable_d = mutable(3)
    output = hypermap_in_dynamic(tuple_a, tuple_b, tuple_c, mutable_d)
    assert output == (3, 6, 3, 6, 3, 6)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_hypermap_in_dynamic_normal_list():
    """
    Feature: hypermap of dynamic list
    Description: Test the hypermap of dynamic list
    Expectation: Expectation: Output is equal to the expected output
    """
    list_a = [1, 2]
    list_b = [1, 2]
    list_c = [1, 2]
    mutable_d = mutable(3)
    output = hypermap_in_dynamic(list_a, list_b, list_c, mutable_d)
    assert output == [3, 6, 3, 6, 3, 6]

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_hypermap_in_dynamic_tensor_tuple():
    """
    Feature: hypermap of dynamic tensor tuple
    Description: Test the hypermap of dynamic tuple composed of tensors
    Expectation: Expectation: Output is equal to the expected output
    """
    tuple_a = (Tensor(1, mstype.float32), Tensor(2, mstype.float32))
    tuple_b = (Tensor(1, mstype.float32), Tensor(2, mstype.float32))
    tuple_c = (Tensor(1, mstype.float32), Tensor(2, mstype.float32))
    mutable_d = mutable(3)
    output = hypermap_in_dynamic(tuple_a, tuple_b, tuple_c, mutable_d)
    assert output == (Tensor(3, mstype.float32), Tensor(6, mstype.float32),
                      Tensor(3, mstype.float32), Tensor(6, mstype.float32),
                      Tensor(3, mstype.float32), Tensor(6, mstype.float32))

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_hypermap_in_dynamic_tensor_list():
    """
    Feature: hypermap of dynamic tensor list
    Description: Test the hypermap of dynamic list composed of tensors
    Expectation: Expectation: Output is equal to the expected output
    """
    list_a = [Tensor(1, mstype.float32), Tensor(2, mstype.float32)]
    list_b = [Tensor(1, mstype.float32), Tensor(2, mstype.float32)]
    list_c = [Tensor(1, mstype.float32), Tensor(2, mstype.float32)]
    mutable_d = mutable(3)
    output = hypermap_in_dynamic(list_a, list_b, list_c, mutable_d)
    assert output == [Tensor(3, mstype.float32), Tensor(6, mstype.float32),
                      Tensor(3, mstype.float32), Tensor(6, mstype.float32),
                      Tensor(3, mstype.float32), Tensor(6, mstype.float32)]

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1.00
        )
def test_hypermap_in_dynamic_tuple_tuple():
    """
    Feature: hypermap of dynamic tuple tuple
    Description: Test the hypermap of dynamic tuple composed of tuples
    Expectation: Error is raised as expected
    """
    with pytest.raises(TypeError) as raise_info:
        tuple_a = ((1, 2), (1, 2))
        tuple_b = ((1, 2), (1, 2))
        tuple_c = ((1, 2), (1, 2))
        mutable_d = mutable(3)
        hypermap_in_dynamic(tuple_a, tuple_b, tuple_c, mutable_d)
    assert "The HyperMap does not support scenarios involving nested dynamic" in str(raise_info.value)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_hypermap_in_dynamic_list_list():
    """
    Feature: hypermap of dynamic list list
    Description: Test the hypermap of dynamic list composed of lists
    Expectation: Error is raised as expected
    """
    with pytest.raises(TypeError) as raise_info:
        tuple_a = [[1, 2], [1, 2]]
        tuple_b = [[1, 2], [1, 2]]
        tuple_c = [[1, 2], [1, 2]]
        mutable_d = mutable(3)
        hypermap_in_dynamic(tuple_a, tuple_b, tuple_c, mutable_d)
    assert "The HyperMap does not support scenarios involving nested dynamic" in str(raise_info.value)
