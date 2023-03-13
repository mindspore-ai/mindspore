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
""" test check parameter """
import pytest
import numpy as np
from mindspore._checkparam import twice
from mindspore import _checkparam as Validator

kernel_size = 5
kernel_size1 = twice(kernel_size)
assert kernel_size1 == (5, 5)

def test_check_int1():
    a = np.random.randint(-100, 100)
    assert Validator.check_is_int(a) == a

def test_check_int2():
    with pytest.raises(TypeError):
        Validator.check_is_int(3.3)

def test_check_int3():
    with pytest.raises(TypeError):
        Validator.check_is_int("str")

def test_check_int4():
    with pytest.raises(TypeError):
        Validator.check_is_int(True)

def test_check_is_int5():
    with pytest.raises(TypeError):
        Validator.check_is_int(True)
    with pytest.raises(TypeError):
        Validator.check_is_int(False)

def test_check_positive_int1():
    a = np.random.randint(1, 100)
    assert Validator.check_positive_int(a) == a

def test_check_positive_int2():
    a = np.random.randint(-100, 0)
    with pytest.raises(ValueError):
        Validator.check_positive_int(a)

def test_check_positive_int3():
    with pytest.raises(TypeError):
        Validator.check_positive_int(3.3)

def test_check_positive_int4():
    with pytest.raises(TypeError):
        Validator.check_positive_int("str")

def test_check_negative_int1():
    a = np.random.randint(-100, -1)
    assert Validator.check_negative_int(a) == a

def test_check_negative_int2():
    a = np.random.randint(0, 100)
    with pytest.raises(ValueError):
        Validator.check_negative_int(a)

def test_check_negative_int3():
    with pytest.raises(TypeError):
        Validator.check_negative_int(3.3)

def test_check_negative_int4():
    with pytest.raises(TypeError):
        Validator.check_negative_int("str")

def test_check_non_positive_int1():
    a = np.random.randint(-100, 0)
    assert Validator.check_non_positive_int(a) == a

def test_check_non_positive_int2():
    a = np.random.randint(1, 100)
    with pytest.raises(ValueError):
        Validator.check_non_positive_int(a)

def test_check_non_positive_int3():
    with pytest.raises(TypeError):
        Validator.check_non_positive_int(3.3)

def test_check_non_positive_int4():
    with pytest.raises(TypeError):
        Validator.check_non_positive_int("str")

def test_check_bool_1():
    assert Validator.check_bool(True)


def test_check_bool_2():
    assert Validator.check_bool(False) is not True


def test_check_bool_3():
    with pytest.raises(TypeError):
        Validator.check_bool("str")


def test_check_bool_4():
    with pytest.raises(TypeError):
        Validator.check_bool(1)


def test_check_bool_5():
    with pytest.raises(TypeError):
        Validator.check_bool(3.5)


def test_twice_1():
    assert twice(3) == (3, 3)


def test_twice_2():
    assert twice((3, 3)) == (3, 3)


def test_twice_3():
    with pytest.raises(TypeError):
        twice(0.5)


def test_twice_4():
    with pytest.raises(TypeError):
        twice("str")


def test_twice_5():
    with pytest.raises(TypeError):
        twice((1, 2, 3))


def test_twice_6():
    with pytest.raises(TypeError):
        twice((3.3, 4))
