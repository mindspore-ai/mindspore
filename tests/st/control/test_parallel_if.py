# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype, Parameter
from mindspore.nn import Cell
import pytest


def setup_module():
    context.set_context(mode=context.GRAPH_MODE)


def test_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in while loop requires that the after-if func graph should not
                 be called.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    return bias
                x = x - 1
                y = y + 1
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_break_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    return bias
                break
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    return bias
                return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_break_else_return_in_while_in_else_take_break():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: take the break branch, success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    break
                return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_break_else_return_in_while_in_else_take_return():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: take the return branch, success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y < 0:
                    break
                return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([4], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_while_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer while to
                 the else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer while to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        return bias
                    x = x - 1
                    y = y + 1
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        return bias
                    return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_while_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            while x > 0:
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                return bias
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The inner first if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            if x > 0:
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The inner second if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > 0:
                return bias
            if x > y:
                x = x + y
            else:
                x = x - y
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_while_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            while x > 0:
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_by_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The second if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                return bias
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in else of the second if/else requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y
        if bias > y:
            y = x + y
        else:
            if x > 0:
                return bias

        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_by_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The second first if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > 0:
                return bias
        if x > y:
            x = x + y
        else:
            x = x - y

        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_in_if_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else in the first if can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            while x > 0:
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_in_if_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in else of the first if/else requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The if/else inside the first if can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            if x > 0:
                return bias

        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in for loop requires that the after-if func graph should not
                 be called.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return bias
                x = x - 1
                y = y + 1
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_break_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return bias
                break
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return bias
                return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_for_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer for to
                 the else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer for to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        return bias
                    x = x - 1
                    y = y + 1
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_return_else_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        return bias
                    return x
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_for_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            for _ in range(5):
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                return bias
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_for_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_after_by_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The second if/else can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                return bias
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_else_in_if_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else in the first if can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            for _ in range(5):
                return bias
        return x + y

    x = Tensor([4], mstype.int32)
    y = Tensor([1], mstype.int32)
    bias = Tensor([-5], mstype.int32)
    expect = Tensor([-5], mstype.int32)
    ret = foo(x, y, bias)
    assert ret == expect


def test_if_by_if_break_in_if_in_while():
    """
    Feature: Parallel if transformation.
    Description: break in if in while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer if part. The by if can be transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, z):
        out = z
        while x < y:
            if y > 2 * x:
                out = out + out
                if y > 3 * x:
                    y = y - 1
                if y == 3 * x:
                    break
        out = out + out
        return out

    x = Tensor(2, mstype.int32)
    y = Tensor(8, mstype.int32)
    z = Tensor([5], mstype.int32)
    expect = Tensor([40], mstype.int32)
    ret = foo(x, y, z)
    assert ret == expect


def test_if_raise_raise():
    """
    Feature: Parallel if transformation.
    Description: raise in if requires that the after-if func graph should not
                 be called, so it cannot be transformed. The outer if can be
                 transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, z):
        out = z
        if x >= y:
            if x > y:
                raise ValueError("x is bigger y")
        else:
            out = out * 2
        out = out + out
        return out

    x = 3
    y = 2
    z = Tensor([5], mstype.int32)
    with pytest.raises(ValueError):
        foo(x, y, z)


def test_if_raise_not_raise():
    """
    Feature: Parallel if transformation.
    Description: raise in if requires that the after-if func graph should not
                 be called, so it cannot be transformed. The outer if can be
                 transformed.
    Expectation: success
    """

    @jit
    def foo(x, y, z):
        out = z
        if x >= y:
            if x > y:
                raise ValueError("x is bigger y")
        else:
            out = out * 2
        out = out + out
        return out

    x = 2
    y = 2
    z = Tensor([5], mstype.int32)
    expected = Tensor([10], mstype.int32)
    ret = foo(x, y, z)
    assert ret == expected


def test_if_assert_success():
    """
    Feature: Parallel if transformation.
    Description: assert in if will not affect the inner and outer if transformation.
    Expectation: success
    """

    @jit
    def foo(x, y, z):
        out = z
        out = z
        if x >= y:
            if x > y:
                assert x > y
                out = out * 3
        else:
            out = out * 2
        out = out + out
        return out

    x = 3
    y = 2
    z = Tensor([5], mstype.int32)
    expected = Tensor([30], mstype.int32)
    ret = foo(x, y, z)
    assert ret == expected


def test_if_assert_failure():
    """
    Feature: Parallel if transformation.
    Description: assert in if will not affect the inner and outer if transformation.
    Expectation: success
    """

    @jit
    def foo(x, y, z):
        out = z
        if x >= y:
            if x > y:
                assert x == y
                out = out * 3
        else:
            out = out * 2
        out = out + out
        return out

    x = 3
    y = 2
    z = Tensor([5], mstype.int32)
    with pytest.raises(Exception):
        foo(x, y, z)


def test_weight_multiple_one_in_if():
    """
    Feature: Parallel if transformation.
    Description: If the return value of the subgraph is Load, need to insert TensorMove.
                 "x = self.w * 1" mean that x is a copy of self.w, x and self.w are not the same object.
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([4], mstype.int32), name='weight')

        def construct(self, x, y):
            if y != self.w:
                x = self.w * 1
                self.w = self.w - 1
            return x + y

    x = Tensor([2], mstype.int32)
    y = Tensor([3], mstype.int32)
    expect = Tensor([7], mstype.int32)
    ret = Net()(x, y)
    assert ret == expect


def test_weight_in_if():
    """
    Feature: Parallel if transformation.
    Description: "x = self.w" mean that x is another name for self.w, x and self.w are the same object.
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.w = Parameter(Tensor([4], mstype.int32), name='weight')

        def construct(self, x, y):
            if y != self.w:
                x = self.w
                self.w = self.w - 1
            return x + y

    x = Tensor([2], mstype.int32)
    y = Tensor([3], mstype.int32)
    expect = Tensor([6], mstype.int32)
    ret = Net()(x, y)
    assert ret == expect


def test_weight_tuple_in_if():
    """
    Feature: Parallel if transformation.
    Description: If the return value of the subgraph is Tuple(Load), need to insert TensorMove to each load.
    Expectation: success
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter(Tensor([4], mstype.int32), name='weight1')
            self.w2 = Parameter(Tensor([5], mstype.int32), name='weight2')

        def construct(self, x, y):
            if y != self.w1:
                x = self.w1 * 1
                self.w1 = self.w1 - 1
                y = self.w2 / 1
                self.w2 = self.w2 - 1
            return x + y
    input_x = Tensor([2], mstype.int32)
    input_y = Tensor([3], mstype.int32)
    expect = Tensor([9], mstype.int32)
    net = Net()
    ret = net(input_x, input_y)
    assert ret == expect
