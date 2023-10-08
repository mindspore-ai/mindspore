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

from mindspore import Tensor


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


# pylint: disable=unused-variable
def test_compare(tag):
    """
    Feature: Boost parse.
    Description: Parse the network witch has comparison statement.
    Expectation: The false branch should be folded.
    """
    fns = FnDict()

    @fns
    def is_none(x, y):
        z = None
        if z is None:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def is_not_none(x, y):
        z = 1
        if z is not None:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def equal_num(x, y):
        z = 1
        if z == 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def equal_str(x, y):
        z = "abc"
        if z == "abc":
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def equal_tensor(x, y):
        z = Tensor(2)
        if z == Tensor(2):
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def equal_tensor_with_comment(x, y):
        z = Tensor(2)
        if z == Tensor(2):  # @jit.cond: True
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def not_equal_num1(x, y):
        z = 2
        if z != 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def not_equal_num2(x, y):
        z = None
        if z != 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def greater(x, y):
        z = 2
        if z > 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def greater_equal(x, y):
        z = 2
        if z >= 2:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def less(x, y):
        z = 2
        if z < 3:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def less_equal(x, y):
        z = 2
        if z <= 2:
            x = x + y
        else:
            x = x - y
        return x + x

    return fns[tag]


def test_if_name(tag):
    """
    Feature: Boost parse.
    Description: Parse the network witch has "if var:" statement.
    Expectation: The false branch should be folded.
    """
    fns = FnDict()

    @fns
    def if_name(x, y):
        z = True
        if z:
            x = x + y
        else:
            x = x - y
        return x + x

    return fns[tag]


def test_unary_op(tag):
    """
    Feature: Boost parse.
    Description: Parse the network witch has UnaryOp statement.
    Expectation: The false branch should be folded.
    """
    fns = FnDict()

    @fns
    def if_not_name(x, y):
        z = False
        if not z:
            x = x + y
        else:
            x = x - y
        return x + x

    return fns[tag]


def test_bool_op(tag):
    """
    Feature: Boost parse.
    Description: Parse the network witch has BoolOp statement.
    Expectation: The false branch should be folded.
    """
    fns = FnDict()

    @fns
    def name_or_equal(x, y):
        z = True
        one = 1
        if z or one > 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def unary_op_or_equal(x, y):
        z = True
        one = 1
        if not z or z == 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def name_and_equal(x, y):
        z = True
        one = 1
        if z and one == 1:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def unary_op_and_equal(x, y):
        z = False
        one = 1
        if not z and one == 1:
            x = x + y
        else:
            x = x - y
        return x + x

    return fns[tag]
