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
"""
file: parallel_if.py
"""
from mindspore.ops import functional as F
from mindspore._extends.parse import standard_method


class FnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        return self.fn_dict.get(name)


# pylint: disable=unused-variable
# disable pylint unused variable for basic/manual which are used by decorator.
def test_simple_if(tag):
    """
    Feature: Parallel if transformation
    Description: simple if with single if/else
    Expectation: funcgraph parsed and manual constructed should be isomorphic.
    """
    fns = FnDict()

    @fns
    def basic(x, y):
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + x

    @fns
    def manual(x, y):
        def after(a_x):
            return a_x + a_x

        def true_branch():
            return x + y

        def false_branch():
            return x - y

        cond = standard_method.bool_(x > y)

        switch_node = F.switch(cond, true_branch, false_branch)
        result = switch_node()
        return after(result)

    return fns[tag]


def test_if_by_if(tag):
    """
    Feature: Parallel if transformation
    Description: if/else after if/else
    Expectation: funcgraph parsed and manual constructed should be isomorphic.
    """
    fns = FnDict()

    @fns
    def basic(x, y):
        if x > y:
            x = x + y
        else:
            x = x - y
        if x < y:
            y = x * y
        else:
            y = x + y
        return x + y

    @fns
    def manual(x, y):
        # first if
        def true_branch1():
            return x + y

        def false_branch1():
            return x - y

        cond1 = standard_method.bool_(x > y)
        switch_node = F.switch(cond1, true_branch1, false_branch1)
        result1 = switch_node()

        cond2 = standard_method.bool_(result1 < y)

        # second if
        def true_branch2():
            return result1 * y

        def false_branch2():
            return result1 + y

        def after2(a_x, a_y):
            return a_x + a_y

        def after1():
            switch_node = F.switch(cond2, true_branch2, false_branch2)
            result2 = switch_node()
            return after2(result1, result2)

        return after1()

    return fns[tag]


def test_if_in_if(tag):
    """
    Feature: Parallel if transformation
    Description: if/else in if
    Expectation: funcgraph parsed and manual constructed should be isomorphic.
    """
    fns = FnDict()

    @fns
    def basic(x, y):
        if x >= y:
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            x = x * y
        return x + y

    @fns
    def manual(x, y):
        # inner if/else
        def true_branch2():
            return x + y

        def false_branch2():
            return x - y

        def after2(a_x):
            return a_x

        # outer if/else
        def after1(a_x):
            return a_x + y

        def true_branch1():
            cond2 = standard_method.bool_(x > y)
            switch_node = F.switch(cond2, true_branch2, false_branch2)
            result2 = switch_node()
            return after2(result2)

        def false_branch1():
            return x * y

        cond1 = standard_method.bool_(x >= y)
        switch_node = F.switch(cond1, true_branch1, false_branch1)
        result1 = switch_node()
        return after1(result1)

    return fns[tag]


def test_if_elif_else(tag):
    """
    Feature: Parallel if transformation
    Description: if/elif/else which can be treated as if/else{if/else}.
    Expectation: funcgraph parsed and manual constructed should be isomorphic.
    """
    fns = FnDict()

    @fns
    def basic(x, y):
        if x > y:
            out = x + y
        elif x == y:
            out = x - y
        else:
            out = x * y
        return out + out

    @fns
    def manual(x, y):
        # elif/else part
        def true_branch2():
            return x - y

        def false_branch2():
            return x * y

        def after2(out):
            return out

        # if part
        def after1(out):
            return out + out

        def true_branch1():
            return x + y

        def false_branch1():
            cond2 = standard_method.bool_(x == y)
            switch_node = F.switch(cond2, true_branch2, false_branch2)
            result2 = switch_node()
            return after2(result2)

        cond1 = standard_method.bool_(x > y)
        switch_node = F.switch(cond1, true_branch1, false_branch1)
        result1 = switch_node()
        return after1(result1)

    return fns[tag]


#  The following test cases are Combination of
#      Three kinds of statements: return, break, continue
#      Two kinds of loop: while loop, for loop
#      Location of additional if/else: if/else parallel with loop,
#                                      if/else parallel with if/else, if/else inside if.
def test_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in while loop requires that the after-if func graph should not
                 be called.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                return bias
        return x + y

    return foo


def test_if_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_if_return_else_break_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    return bias
                break
        return x + y

    return foo


def test_if_return_else_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    return bias
                return x
        return x + y

    return foo


def test_while_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer while to
                 the else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    return bias
        return x + y

    return foo


def test_if_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer while to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_if_return_else_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_while_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The inner first if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The inner second if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_while_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_by_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The second if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in else of the second if/else requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_by_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner second if requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The second first if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_in_if_while_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part. The inner if/else in the first if can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_in_if_if_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in else of the first if/else requires that the after-if func graph should not
                 be called, and this information should be propagated through if to
                 the outer else part. The if/else inside the first if can be transformed.
    Expectation: success
    """

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

    return foo


def test_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in for loop requires that the after-if func graph should not
                 be called.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                return bias
        return x + y

    return foo


def test_if_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_if_return_else_break_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return bias
                break
        return x + y

    return foo


def test_if_return_else_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return bias
                return x
        return x + y

    return foo


def test_for_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer for to
                 the else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    return bias
        return x + y

    return foo


def test_if_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through outer for to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_if_return_else_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part.
    Expectation: success
    """

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

    return foo


def test_for_return_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_for_return_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The first if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_after_by_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The second if/else can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_else_in_if_for_return_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer else part. The inner if/else in the first if can be transformed.
    Expectation: success
    """

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

    return foo


## Similar test cases but replace return to break. These test cases may not runnable as it just replace
## return bias to break.
def test_while_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so the if-else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                break
        return x + y

    return foo


def test_if_break_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: if inside the while loop cannot be transformed, but the outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    break
                x = x - 1
                y = y + 1
        return x + y

    return foo


def test_if_break_else_break_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: if inside the while loop cannot be transformed, but the outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    break
                break
        return x + y

    return foo


def test_if_break_else_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    break
                return x
        return x + y

    return foo


def test_while_break_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in inner while loop cannot flow beyond the loop, so the if-else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    break
        return x + y

    return foo


def test_if_break_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in inner while loop cannot flow beyond the while loop, so the outer
                 if/else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        break
                    x = x - 1
                    y = y + 1
        return x + y

    return foo


def test_if_break_else_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        break
                    return x
        return x + y

    return foo


def test_while_break_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            while x > 0:
                break
        return x + y

    return foo


def test_if_else_after_while_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                break
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    return foo


def test_while_break_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            while x > 0:
                break
        return x + y

    return foo


def test_if_else_after_by_while_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                break
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    return foo


def test_if_else_in_if_while_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            while x > 0:
                break
        return x + y

    return foo


def test_for_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop cannot flow beyond the loop, so if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                break
        return x + y

    return foo


def test_if_break_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop cannot flow beyond the loop, so outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    break
                x = x - 1
                y = y + 1
        return x + y

    return foo


def test_if_break_else_return_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in for loop will flow beyond the loop, so no if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    break
                return x
        return x + y

    return foo


def test_if_break_else_break_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop cannot flow beyond the loop, so outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    break
                break
        return x + y

    return foo


def test_for_break_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in inner for loop cannot flow beyond the loop, so if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    break
        return x + y

    return foo


def test_if_break_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in inner for loop will not flow beyond the loop, so outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        break
                    x = x - 1
                    y = y + 1
        return x + y

    return foo


def test_if_break_else_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop will flow beyond the loop, so no if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        break
                    return x
        return x + y

    return foo


def test_for_break_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            for _ in range(5):
                break
        return x + y

    return foo


def test_if_else_after_for_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                break
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    return foo


def test_for_break_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                break
        return x + y

    return foo


def test_if_else_after_by_for_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                break
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    return foo


def test_if_else_in_if_for_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: break in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            for _ in range(5):
                break
        return x + y

    return foo


## Similar test cases but replace break to continue. These test cases may not runnable as it just replace
## break to continue.
def test_while_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so the if-else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                continue
        return x + y

    return foo


def test_if_continue_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: if inside the while loop cannot be transformed, but the outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    continue
                x = x - 1
                y = y + 1
        return x + y

    return foo


def test_if_continue_else_continue_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: if inside the while loop cannot be transformed, but the outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    continue
                continue
        return x + y

    return foo


def test_if_continue_else_return_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                if y > 0:
                    continue
                return x
        return x + y

    return foo


def test_while_continue_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in inner while loop cannot flow beyond the loop, so the if-else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    continue
        return x + y

    return foo


def test_if_continue_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in inner while loop cannot flow beyond the while loop, so the outer
                 if/else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        continue
                    x = x - 1
                    y = y + 1
        return x + y

    return foo


def test_if_continue_else_return_in_while_in_while_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through while to
                 the outer else part.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                while x > 0:
                    if y > 0:
                        continue
                    return x
        return x + y

    return foo


def test_while_continue_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            while x > 0:
                continue
        return x + y

    return foo


def test_if_else_after_while_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                continue
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    return foo


def test_while_continue_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            while x > 0:
                continue
        return x + y

    return foo


def test_if_else_after_by_while_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            while x > 0:
                continue
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    return foo


def test_if_else_in_if_while_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in while loop cannot flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            while x > 0:
                continue
        return x + y

    return foo


def test_for_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop cannot flow beyond the loop, so if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                continue
        return x + y

    return foo


def test_if_continue_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop cannot flow beyond the loop, so outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    continue
                x = x - 1
                y = y + 1
        return x + y

    return foo


def test_if_return_else_continue_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in for loop will flow beyond the loop, so no if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    return x
                continue
        return x + y

    return foo


def test_if_continue_else_continue_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in for loop will flow beyond the loop, so no if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                if y > 0:
                    continue
                continue
        return x + y

    return foo


def test_for_continue_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in inner for loop cannot flow beyond the loop, so if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    continue
        return x + y

    return foo


def test_if_continue_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in inner for loop will not flow beyond the loop, so outer if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        continue
                    x = x - 1
                    y = y + 1
        return x + y

    return foo


def test_if_continue_else_return_in_for_in_for_in_else():
    """
    Feature: Parallel if transformation.
    Description: return in inner for loop will flow beyond the loop, so no if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                for _ in range(5):
                    if y > 0:
                        continue
                    return x
        return x + y

    return foo


def test_for_continue_after_if_else_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            if x > y:
                x = x + y
            else:
                x = x - y

            for _ in range(5):
                continue
        return x + y

    return foo


def test_if_else_after_for_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                continue
            if x > y:
                x = x + y
            else:
                x = x - y

        return x + y

    return foo


def test_for_continue_in_else_after_if_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if x > y:
            x = x + y
        else:
            x = x - y

        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                continue
        return x + y

    return foo


def test_if_else_after_by_for_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
        else:
            for _ in range(5):
                continue
        if x > y:
            x = x + y
        else:
            x = x - y
        return x + y

    return foo


def test_if_else_in_if_for_continue_in_else():
    """
    Feature: Parallel if transformation.
    Description: continue in for loop will not flow beyond the loop, so both if can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        if bias > y:
            y = x + y
            if x > y:
                x = x + y
            else:
                x = x - y
        else:
            for _ in range(5):
                continue
        return x + y

    return foo


def test_func_call_in_if_while_break_in_else():
    """
    Feature: Parallel if transformation.
    Description: return inside def func should not propagate to caller of that func, so the if-else can be transformed.
    Expectation: success
    """

    def foo(x, y, bias):
        def bar(x, y):
            return x + y

        if bias > y:
            y = bar(x, y)
        else:
            while x > 0:
                break
        return x + y

    return foo


def test_if_by_if_break_in_if_in_while():
    """
    Feature: Parallel if transformation.
    Description: break in if in while loop requires that the after-if func graph should not
                 be called, and this information should be propagated through for to
                 the outer if part. The by if can be transformed.
    Expectation: success
    """

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

    return foo


def test_if_raise_raise():
    """
    Feature: Parallel if transformation.
    Description: raise in if requires that the after-if func graph should not
                 be called, so it cannot be transformed. The outer if can be
                 transformed.
    Expectation: success
    """

    def foo(x, y, z):
        out = z
        if x >= y:
            if x > y:
                raise ValueError("x is bigger y")
        else:
            out = out * 2
        out = out + out
        return out

    return foo


def test_if_assert_failure():
    """
    Feature: Parallel if transformation.
    Description: assert in if will not affect the inner and outer if transformation.
    Expectation: success
    """

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

    return foo
