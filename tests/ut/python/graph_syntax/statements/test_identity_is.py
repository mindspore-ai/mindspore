# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
""" test syntax for logic expression """
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore import dtype as mstype
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE)


class IdentityIs(nn.Cell):
    def __init__(self, x, y):
        super(IdentityIs, self).__init__()
        self.x = x
        self.y = y

    def construct(self):
        in_v = self.x is self.y
        return in_v


def test_ms_syntax_operator_int_is_int():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(1, 1)
    ret = net()
    assert ret


def test_ms_syntax_operator_int_is_int_2():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(1, 3)
    ret = net()
    assert not ret


def test_ms_syntax_operator_int_is_none():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(1, None)
    ret = net()
    assert not ret


def test_ms_syntax_operator_int_is_true():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(1, True)
    ret = net()
    assert not ret


def test_ms_syntax_operator_bool_is_none():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(True, None)
    ret = net()
    assert not ret


def test_ms_syntax_operator_bool_is_false():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(True, False)
    ret = net()
    assert not ret


def test_ms_syntax_operator_type_is_type():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    x = Tensor(0, ms.int32)
    net = IdentityIs(x.dtype, mstype.bool_)
    ret = net()
    assert not ret


def test_ms_syntax_operator_scalar_is_type():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    x = 1
    net = IdentityIs(x, mstype.bool_)
    ret = net()
    assert not ret


def test_ms_syntax_operator_is_int():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(int, int)
    ret = net()
    assert ret


def test_ms_syntax_operator_is_float():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(int, float)
    ret = net()
    assert not ret


def test_ms_syntax_operator_is_tensor():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs(ms.Tensor, ms.Tensor)
    ret = net()
    assert ret


def test_ms_syntax_operator_str_is_str():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs("aaa", "aaa")
    ret = net()
    assert ret


def test_ms_syntax_operator_str_is_str_2():
    """
    Feature: simple expression
    Description: test is operator.
    Expectation: No exception
    """
    net = IdentityIs("aaa", "bbb")
    ret = net()
    assert not ret
