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
""" test_remove_unnecessary_phi """
# coding: utf-8

from numpy.random import normal

from mindspore import Tensor
from mindspore import context
from mindspore.common.api import jit


def setup_module(module):
    context.set_context(mode=context.PYNATIVE_MODE)


# If phi node (fv_func) in for body is not replaced by fv_func,
# this test will failed as phi node as a parameter will be inferred
# as POLY.
def test_remove_phi_and_fv():
    """ test_remove_phi_and_fv """

    @jit
    def loop(x, input_data):
        def fv_func(y):
            return x * y

        ret = ()
        for inp in input_data:
            ele = fv_func(inp)
            ret += (ele,)
        return ret

    input_data = (Tensor(normal(0, 0.1, (3, 3))), Tensor(normal(0, 0.1, (3))))
    input1 = Tensor(normal(0, 0.1, (3, 3)))
    print(loop(input1, input_data))


# Multiple phi nodes should be replaced.
# mul Φ0 (mul, Φ0); Φ0 will be replaced by mul;
# x   Φ1 (x, Φ1); Φ1 will be replaced by x;
# ret Φ2 (1, ret{[0]: Φ0, [1]: Φ1, [2]: inp}), Φ2 will not be replaced.

# Φ0 and Φ1 in Φ2 node should be replaced with mul and x.
def test_remove_multiple_phi():
    """ test_remove_multiple_phi """

    @jit
    def loop(x):
        def mul(a, b):
            return a * b

        ret = 1
        for inp in range(3):
            ret = mul(x, inp)
        return ret

    print(loop(2))


# replace phi nodes recursively.
# x as phi Φ5 (Φ3, Φ3) in graph ↓⥁loop
# x as phi Φ3 (x, Φ5) in graph ⤾loop
# one predecessor of ⤾loop is ↓⥁loop.
# Φ5 will be resolved first, it can be replace by Φ3, then Φ3 (x, Φ5) become Φ3 (x, Φ3), so Φ3 can be replaced by x.
# recursively, Φ5 also should be replaced by x.
def test_remove_multiple_phi_recursive():
    """ test_remove_multiple_phi_recursive """

    @jit
    def loop(x):
        def mul(a, b):
            return a * b

        ret = 1
        for inp in range(3):
            if inp % 2 == 0:
                ret = mul(inp, inp)
        return ret * x

    print(loop(2))
