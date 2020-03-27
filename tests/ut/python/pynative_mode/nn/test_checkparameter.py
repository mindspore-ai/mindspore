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
""" test_checkparameter """
import pytest

from mindspore._checkparam import check_int, check_int_positive, \
    check_bool, check_input_format, _expand_tuple


once = _expand_tuple(1)
twice = _expand_tuple(2)
triple = _expand_tuple(3)
kernel_size = 5
kernel_size1 = twice(kernel_size)
assert kernel_size1 == (5, 5)


def test_check_int_1():
    assert check_int(3) == 3


def check_int_positive_1():
    with pytest.raises(ValueError):
        check_int_positive(-1)


def test_NCHW1():
    assert check_input_format("NCHW") == "NCHW"


def test_NCHW3():
    with pytest.raises(ValueError):
        check_input_format("rt")


def test_check_int_2():
    with pytest.raises(TypeError):
        check_int(3.3)


def test_check_int_3():
    with pytest.raises(TypeError):
        check_int("str")


def test_check_int_4():
    with pytest.raises(TypeError):
        check_int(True)


def test_check_bool_1():
    assert check_bool(True)


def test_check_bool_2():
    assert check_bool(False) is not True


def test_check_bool_3():
    with pytest.raises(TypeError):
        check_bool("str")


def test_check_bool_4():
    with pytest.raises(TypeError):
        check_bool(1)


def test_check_bool_5():
    with pytest.raises(TypeError):
        check_bool(3.5)


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
