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
