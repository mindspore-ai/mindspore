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
""" order_enforce_test """
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Parameter


# pylint: disable=unused-variable

class TestOrderEnforceFnDict:
    def __init__(self):
        self.fn_dict = {}

    def __call__(self, fn):
        self.fn_dict[fn.__name__] = fn

    def __getitem__(self, name):
        try:
            return self.fn_dict[name]
        except KeyError:
            return None


# Test ut for file: order_enforce.cc.
def test_order_enforce(tag):
    """
    Feature: Side Effect.
    Description: Insert `TensorMove` nodes afters some special `Load` nodes.
    Expectation: The `TensorMove`nodes are inserted after `Load` nodes as expected.
    """
    fns = TestOrderEnforceFnDict()

    param_a = Parameter(Tensor([1], mstype.int32), name='a')
    param_b = Parameter(Tensor([2], mstype.int32), name='b')
    param_c = Parameter(Tensor([3], mstype.int32), name='c')
    param_d = Parameter(Tensor([4], mstype.int32), name='d')

    x = Tensor([5], mstype.int32)
    y = Tensor([6], mstype.int32)

    load = P.Load()
    pow_op = P.Pow()

    @fns
    def test_two_loads():
        load1 = load(param_a)
        load2 = load(param_a)
        load3 = load(param_b)
        return load1, load2, load3

    @fns
    def test_partial_load_arg():
        def partial_func(parameter_0, parameter_1):
            return parameter_0, parameter_1

        func = F.partial(partial_func, load(param_a))
        output = func(load(param_b))
        return output

    @fns
    def test_partial_load_arg_call_out_as_arg():
        def partial_func(parameter_0, parameter_1):
            return parameter_0, parameter_1

        def arg_func():
            return load(param_a)

        func = F.partial(partial_func, arg_func())
        output = func(load(param_b))
        return output

    @fns
    def test_call_out_load():
        def partial_func(parameter_0, parameter_1):
            return parameter_0, parameter_1

        func = F.partial(partial_func, param_a)
        output = func(param_b)
        return load(F.tuple_getitem(output, 0)), load(F.tuple_getitem(output, 1))

    @fns
    def load_ref_same_to_call_out():
        def true_func():
            return (param_a, param_d), (param_c, param_c)

        def false_func():
            return (param_b, param_d), (param_c, param_c)

        cond = pow_op(x, x) < pow_op(y, y)
        func = F.switch(cond, true_func, false_func)
        switch_call_output = func()
        tuple_getitem_0_0 = F.tuple_getitem(F.tuple_getitem(switch_call_output, 0), 0)
        tuple_getitem_0_1 = F.tuple_getitem(F.tuple_getitem(switch_call_output, 0), 1)
        output = F.make_tuple(load(param_a), load(param_b), load(param_c), load(tuple_getitem_0_0),
                              load(tuple_getitem_0_1))
        return output

    @fns
    def test_switch_switch_call():
        def true_func():
            def true_true_func():
                return param_a

            def true_false_func():
                return param_b

            cond = pow_op(x, x) < pow_op(y, y)
            return F.switch(cond, true_true_func, true_false_func)()

        def false_func():
            def false_true_func():
                return param_c

            def false_false_func():
                return param_c

            cond = pow_op(x, x) < pow_op(y, y)
            return F.switch(cond, false_true_func, false_false_func)()

        def func():
            cond = pow_op(x, x) < pow_op(y, y)
            return F.switch(cond, true_func, false_func)()

        output = F.make_tuple(load(param_a), load(param_b), load(param_c), load(param_d), load(func()))
        return output

    return fns[tag]
