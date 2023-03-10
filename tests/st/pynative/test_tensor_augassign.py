# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test_tensor_setitem """
import numpy as np
import pytest

from mindspore import Tensor, context
from mindspore import dtype as mstype


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_slice():
    input_np_3d = np.arange(120).reshape(4, 5, 6).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)
    index_slice_1 = slice(1, None, None)
    index_slice_2 = slice(None, 4, None)
    index_slice_3 = slice(-3, 4, None)
    index_slice_4 = slice(2, -1, None)
    index_slice_7 = slice(1, 5, None)
    index_slice_8 = slice(-5, 3, None)

    value_number = 3
    value_list_1_ele = [2]
    value_list_mul_ele = [10, 20, 30, 40, 50, 60]
    value_list_much_ele = [10, 20, 30, 40, 50, 60, 70]

    input_tensor_3d[index_slice_1] += value_number
    input_np_3d[index_slice_1] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_slice_2] -= value_list_1_ele
    input_np_3d[index_slice_2] -= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_slice_3] *= value_list_mul_ele
    input_np_3d[index_slice_3] *= value_list_mul_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_slice_4] /= value_number
    input_np_3d[index_slice_4] /= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_slice_7] /= value_number
    input_np_3d[index_slice_7] /= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_slice_8] += value_number
    input_np_3d[index_slice_8] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    with pytest.raises(ValueError):
        input_tensor_3d[index_slice_8] /= value_list_much_ele


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_ellipsis():
    input_np_3d = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    value_number_1, value_number_2 = 1, 2.0

    value_np_1 = np.array([1])
    value_np_2 = np.array([1, 2, 3, 4])
    value_np_3 = np.arange(12).reshape(3, 4)
    value_tensor_1 = Tensor(value_np_1)
    value_tensor_2 = Tensor(value_np_2)
    value_tensor_3 = Tensor(value_np_3)

    value_tuple_1_ele = (0.5,)
    value_tuple_4_ele = (0.1, 0.2, 0.3, 0.4)

    value_list_1_ele = [1.5]
    value_list_4_ele = [1.1, 1.2, 1.3, 1.4]

    input_tensor_3d[...] += value_number_1
    input_np_3d[...] += value_number_1
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] -= value_number_2
    input_np_3d[...] -= value_number_2
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] *= value_tensor_1
    input_np_3d[...] *= value_np_1
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] /= value_tensor_2
    input_np_3d[...] /= value_np_2
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] /= value_tensor_3
    input_np_3d[...] /= value_np_3
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] -= value_tuple_1_ele
    input_np_3d[...] -= value_tuple_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] *= value_tuple_4_ele
    input_np_3d[...] *= value_tuple_4_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] -= value_list_1_ele
    input_np_3d[...] -= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[...] *= value_list_4_ele
    input_np_3d[...] *= value_list_4_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_bool():
    input_np_3d = np.arange(120).reshape(4, 5, 6).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    index_bool_1 = True
    index_bool_2 = False

    value_number = 1

    value_np_1 = np.array([1], np.float32)
    value_np_2 = np.array([1, 2, 3, 4, 5, 6], np.float32)
    value_np_3 = np.arange(1, 31).astype(np.float32).reshape(5, 6)
    value_np_4 = np.arange(1, 121).astype(np.float32).reshape(4, 5, 6)
    value_tensor_1 = Tensor(value_np_1, mstype.float32)
    value_tensor_2 = Tensor(value_np_2, mstype.float32)
    value_tensor_3 = Tensor(value_np_3, mstype.float32)
    value_tensor_4 = Tensor(value_np_4, mstype.float32)

    value_tuple_1_ele = (0.5,)
    value_tuple_6_ele = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    value_list_1_ele = [1.5]
    value_list_6_ele = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    input_tensor_3d[index_bool_1] += value_number
    input_np_3d[index_bool_1] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] -= value_tensor_1
    input_np_3d[index_bool_1] -= value_np_1
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] *= value_tensor_2
    input_np_3d[index_bool_1] *= value_np_2
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] -= value_tensor_3
    input_np_3d[index_bool_1] -= value_np_3
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] //= value_tensor_4
    input_np_3d[index_bool_1] //= value_np_4
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] %= value_tuple_1_ele
    input_np_3d[index_bool_1] %= value_tuple_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] %= value_tuple_6_ele
    input_np_3d[index_bool_1] %= value_tuple_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] %= value_list_1_ele
    input_np_3d[index_bool_1] %= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_bool_1] -= value_list_6_ele
    input_np_3d[index_bool_1] -= value_list_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    with pytest.raises(IndexError):
        input_tensor_3d[index_bool_2] *= value_tensor_2


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_number():
    input_np_1d = np.arange(4).astype(np.float32)
    input_tensor_1d = Tensor(input_np_1d, mstype.float32)
    input_np_3d = np.arange(80).reshape(4, 5, 4).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    number_index_1, number_index_2, number_index_3, number_index_4 = 0, 3, 4, 3.4

    value_number = 2

    value_np_scalar = np.array(5)
    value_np_1_ele = np.array([1])
    value_np_1d = np.array([1, 2, 3, 4])
    value_np_2d = np.arange(20).reshape(5, 4)
    value_tensor_scalar = Tensor(value_np_scalar, mstype.float32)
    value_tensor_1_ele = Tensor(value_np_1_ele, mstype.float32)
    value_tensor_1d = Tensor(value_np_1d, mstype.float32)
    value_tensor_2d = Tensor(value_np_2d, mstype.float32)

    value_tuple_1_ele = (100,)
    value_tuple_mul_ele = (10, 20, 30, 40)
    value_tuple_much_ele = (10, 20, 30, 40, 10)
    value_tuple_empty = ()

    value_list_1_ele = [101]
    value_list_mul_ele = [11, 21, 31, 41]
    value_list_much_ele = [12, 22, 33, 43, 18]
    value_list_empty = []

    input_tensor_1d[number_index_1] += value_number
    input_np_1d[number_index_1] += value_number
    assert np.allclose(input_tensor_1d.asnumpy(), input_np_1d, 0.0001, 0.0001)

    input_tensor_1d[number_index_2] -= value_number
    input_np_1d[number_index_2] -= value_number
    assert np.allclose(input_tensor_1d.asnumpy(), input_np_1d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] *= value_number
    input_np_3d[number_index_1] *= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_2] /= value_number
    input_np_3d[number_index_2] /= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_1d[number_index_1] //= value_tensor_scalar
    input_np_1d[number_index_1] //= value_np_scalar
    assert np.allclose(input_tensor_1d.asnumpy(), input_np_1d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] *= value_tensor_scalar
    input_np_3d[number_index_1] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_2] %= value_tensor_1_ele
    input_np_3d[number_index_2] %= value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] += value_tensor_1d
    input_np_3d[number_index_1] += value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_2] -= value_tensor_2d
    input_np_3d[number_index_2] -= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_1d[number_index_1] += value_tuple_1_ele
    input_np_1d[number_index_1] += value_tuple_1_ele
    assert np.allclose(input_tensor_1d.asnumpy(), input_np_1d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] -= value_tuple_1_ele
    input_np_3d[number_index_1] -= value_tuple_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] *= value_tuple_mul_ele
    input_np_3d[number_index_1] *= value_tuple_mul_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_1d[number_index_2] += value_list_1_ele
    input_np_1d[number_index_2] += value_list_1_ele
    assert np.allclose(input_tensor_1d.asnumpy(), input_np_1d, 0.0001, 0.0001)

    input_tensor_3d[number_index_1] -= value_list_1_ele
    input_np_3d[number_index_1] -= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[number_index_2] *= value_list_mul_ele
    input_np_3d[number_index_2] *= value_list_mul_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    with pytest.raises(IndexError):
        input_tensor_1d[number_index_3] += value_number
    with pytest.raises(IndexError):
        input_tensor_3d[number_index_3] -= value_number
    with pytest.raises(IndexError):
        input_tensor_1d[number_index_4] *= value_number
    with pytest.raises(IndexError):
        input_tensor_3d[number_index_4] /= value_number

    with pytest.raises(ValueError):
        input_tensor_1d[number_index_1] *= value_tuple_mul_ele
    with pytest.raises(ValueError):
        input_tensor_3d[number_index_1] *= value_tuple_much_ele
    with pytest.raises(RuntimeError):
        input_tensor_1d[number_index_1] /= value_tuple_empty

    with pytest.raises(ValueError):
        input_tensor_3d[number_index_2] //= value_list_much_ele
    with pytest.raises(ValueError):
        input_tensor_3d[number_index_2] *= value_list_empty


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_tensor():
    input_np_3d = np.arange(120).reshape(4, 5, 6).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    index_np_1d_1ele = np.random.randint(4, size=1)
    index_np_1d = np.random.randint(4, size=6)
    index_np_2d = np.random.randint(4, size=(5, 6))
    index_np_3d = np.random.randint(4, size=(4, 5, 6))

    index_tensor_1d_1ele = Tensor(index_np_1d_1ele, mstype.int32)
    index_tensor_1d = Tensor(index_np_1d, mstype.int32)
    index_tensor_2d = Tensor(index_np_2d, mstype.int32)
    index_tensor_3d = Tensor(index_np_3d, mstype.int32)

    value_number = 1

    value_np_1 = np.array([1])
    value_np_2 = np.array([1, 2, 3, 4, 5, 6])
    value_np_3 = np.arange(1, 31).reshape(5, 6)
    value_np_4 = np.arange(1, 181).reshape(6, 5, 6)
    value_tensor_1 = Tensor(value_np_1)
    value_tensor_2 = Tensor(value_np_2)
    value_tensor_3 = Tensor(value_np_3)
    value_tensor_4 = Tensor(value_np_4)

    value_tuple_1_ele = (0.5,)
    value_tuple_6_ele = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    value_list_1_ele = [1.5]
    value_list_6_ele = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

    input_tensor_3d[index_tensor_1d_1ele] += value_number
    input_np_3d[index_np_1d_1ele] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d_1ele] -= value_tensor_2
    input_np_3d[index_np_1d_1ele] -= value_np_2
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d_1ele] /= value_tuple_6_ele
    input_np_3d[index_np_1d_1ele] /= value_tuple_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d_1ele] *= value_list_1_ele
    input_np_3d[index_np_1d_1ele] *= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d] += value_number
    input_np_3d[index_np_1d] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d] -= value_tensor_1
    input_np_3d[index_np_1d] -= value_np_1
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d] /= value_tuple_1_ele
    input_np_3d[index_np_1d] /= value_tuple_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_1d] += value_list_6_ele
    input_np_3d[index_np_1d] += value_list_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_2d] -= value_number
    input_np_3d[index_np_2d] -= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_2d] *= value_tensor_2
    input_np_3d[index_np_2d] *= value_np_2
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_2d] /= value_tensor_4
    input_np_3d[index_np_2d] /= value_np_4
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_2d] += value_tuple_6_ele
    input_np_3d[index_np_2d] += value_tuple_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_2d] -= value_list_1_ele
    input_np_3d[index_np_2d] -= value_list_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_3d] *= value_number
    input_np_3d[index_np_3d] *= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_3d] /= value_tensor_1
    input_np_3d[index_np_3d] /= value_np_1
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_3d] += value_tensor_3
    input_np_3d[index_np_3d] += value_np_3
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_3d] /= value_tuple_1_ele
    input_np_3d[index_np_3d] /= value_tuple_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tensor_3d] -= value_list_6_ele
    input_np_3d[index_np_3d] -= value_list_6_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)


# GPU: does not supported op "FloorMod"
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_list():
    input_np_3d = np.arange(120).reshape(4, 5, 6).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    list_index_empty = []
    list_index_int_1 = [2]
    list_index_int_2 = [3, 1]
    list_index_int_overflow = [4, 2]
    list_index_bool_1 = [False, False, False, False]
    list_index_bool_2 = [True, True, True, True]
    list_index_bool_3 = [True, False, True, False]
    list_index_mix_1 = [True, 0]
    list_index_mix_2 = [3, False]

    value_number = 2

    value_np_scalar = np.array(100)
    value_np_1_ele = np.array([1])
    value_np_1d = np.array([1, 2, 3, 4, 5, 6])
    value_np_2d = np.arange(1, 31).reshape(5, 6)
    value_np_3d = np.arange(1, 61).reshape(2, 5, 6)
    value_tensor_scalar = Tensor(value_np_scalar, mstype.float32)
    value_tensor_1_ele = Tensor(value_np_1_ele, mstype.float32)
    value_tensor_1d = Tensor(value_np_1d, mstype.float32)
    value_tensor_2d = Tensor(value_np_2d, mstype.float32)
    value_tensor_3d = Tensor(value_np_3d, mstype.float32)

    input_tensor_3d[list_index_int_1] += value_number
    input_np_3d[list_index_int_1] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_1] += value_tensor_scalar
    input_np_3d[list_index_int_1] += value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_1] -= value_tensor_1_ele
    input_np_3d[list_index_int_1] -= value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_1] *= value_tensor_1d
    input_np_3d[list_index_int_1] *= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_1] /= value_tensor_2d
    input_np_3d[list_index_int_1] /= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] += value_number
    input_np_3d[list_index_int_2] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] //= value_tensor_scalar
    input_np_3d[list_index_int_2] //= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] *= value_tensor_1_ele
    input_np_3d[list_index_int_2] *= value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] %= value_tensor_1d
    input_np_3d[list_index_int_2] %= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] += value_tensor_2d
    input_np_3d[list_index_int_2] += value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_int_2] -= value_tensor_3d
    input_np_3d[list_index_int_2] -= value_np_3d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_2] += value_number
    input_np_3d[list_index_bool_2] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_2] *= value_tensor_scalar
    input_np_3d[list_index_bool_2] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_2] /= value_tensor_1_ele
    input_np_3d[list_index_bool_2] /= value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_2] //= value_tensor_1d
    input_np_3d[list_index_bool_2] //= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_2] %= value_tensor_2d
    input_np_3d[list_index_bool_2] %= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] += value_number
    input_np_3d[list_index_bool_3] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] *= value_tensor_scalar
    input_np_3d[list_index_bool_3] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] += value_tensor_1_ele
    input_np_3d[list_index_bool_3] += value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] -= value_tensor_1d
    input_np_3d[list_index_bool_3] -= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] *= value_tensor_2d
    input_np_3d[list_index_bool_3] *= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_bool_3] /= value_tensor_3d
    input_np_3d[list_index_bool_3] /= value_np_3d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] += value_number
    input_np_3d[list_index_mix_1] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] *= value_tensor_scalar
    input_np_3d[list_index_mix_1] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] += value_tensor_1_ele
    input_np_3d[list_index_mix_1] += value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] -= value_tensor_1d
    input_np_3d[list_index_mix_1] -= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] *= value_tensor_2d
    input_np_3d[list_index_mix_1] *= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_1] /= value_tensor_3d
    input_np_3d[list_index_mix_1] /= value_np_3d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] += value_number
    input_np_3d[list_index_mix_2] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] *= value_tensor_scalar
    input_np_3d[list_index_mix_2] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] += value_tensor_1_ele
    input_np_3d[list_index_mix_2] += value_np_1_ele
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] -= value_tensor_1d
    input_np_3d[list_index_mix_2] -= value_np_1d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] *= value_tensor_2d
    input_np_3d[list_index_mix_2] *= value_np_2d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[list_index_mix_2] /= value_tensor_3d
    input_np_3d[list_index_mix_2] /= value_np_3d
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    with pytest.raises(IndexError):
        input_tensor_3d[list_index_empty] += value_number
    with pytest.raises(IndexError):
        input_tensor_3d[list_index_int_overflow] += value_number
    with pytest.raises(IndexError):
        input_tensor_3d[list_index_bool_1] += value_number


# GPU: does not supported op "FloorMod"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tesnsor_augassign_by_tuple():
    input_np_3d = np.arange(120).reshape(4, 5, 6).astype(np.float32)
    input_tensor_3d = Tensor(input_np_3d, mstype.float32)

    index_tuple_1 = (slice(1, 3, 1), ..., [1, 3, 2])
    index_tuple_2 = (2, 3, 4)
    index_tuple_4 = ([2, 3], True)
    index_tuple_5 = (False, 3)
    index_tuple_6 = (False, slice(3, 1, -1))
    index_tuple_7 = (..., slice(None, 6, 2))

    value_number = 2

    value_np_scalar = np.array(100)
    value_tensor_scalar = Tensor(value_np_scalar, mstype.float32)

    input_tensor_3d[index_tuple_1] += value_number
    input_np_3d[index_tuple_1] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tuple_1] -= Tensor(np.ones((2, 5, 3)), mstype.float32)
    input_np_3d[index_tuple_1] -= np.ones((2, 5, 3))
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tuple_2] *= value_tensor_scalar
    input_np_3d[index_tuple_2] *= value_np_scalar
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tuple_4] //= value_number
    input_np_3d[index_tuple_4] //= value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    input_tensor_3d[index_tuple_7] += value_number
    input_np_3d[index_tuple_7] += value_number
    assert np.allclose(input_tensor_3d.asnumpy(), input_np_3d, 0.0001, 0.0001)

    with pytest.raises(IndexError):
        input_tensor_3d[index_tuple_5] *= value_number

    with pytest.raises(IndexError):
        input_tensor_3d[index_tuple_6] %= value_number
