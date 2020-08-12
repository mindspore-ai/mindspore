# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test implicit conversion """
import numpy as np

from mindspore import Tensor


def test_float_tensor_and_int_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = 2
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]], dtype=np.float32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_tensor_and_float_add():
    x = Tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
    y = 3.3
    ret_actual = x + y
    ret_expect = Tensor(np.array([[4.3, 3.3], [3.3, 4.3]], dtype=np.float32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_tensor_and_int_add():
    x = Tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
    y = 3
    ret_actual = x + y
    ret_expect = Tensor(np.array([[4, 3], [3, 4]], dtype=np.int32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_bool_and_int_tensor_add():
    x = True
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_int_tensor_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_float_tensor_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64))
    y = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_int_tensor_and_int_tensor_add():
    x = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16))
    y = Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[2, 4, 6], [8, 10, 12]], dtype=np.int32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()


def test_float_tensor_and_bool_tensors_add():
    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[True, True, True], [False, False, False]], dtype=np.bool_))
    ret_actual = x + y
    ret_expect = Tensor(np.array([[1.1, 1.2, 1.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    assert (ret_actual.asnumpy() == ret_expect.asnumpy()).all()
