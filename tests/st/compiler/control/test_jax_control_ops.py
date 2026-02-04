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
from tests.mark_utils import arg_mark
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype
from mindspore.ops import composite as C
from mindspore import mutable
import numpy as np

context.set_context(jit_config={"jit_level": "O0"})
grad_by_list = C.GradOperation(get_by_list=True)
grad_all = C.GradOperation(get_all=True)


def no_inline(func):
    func.no_inline = True
    return func


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    val = mutable(val)
    while cond_fun(val):
        val = body_fun(val)
    return val


def while_cond(val):
    return val < 10


def while_body_fun(val):
    val = val * 3 - 1
    return val


@jit(backend="ms_backend")
def call_while_loop(x):
    val = while_loop(while_cond, while_body_fun, x)
    return val


@jit(backend="ms_backend")
def grad_while_loop(x):
    x = grad_all(call_while_loop)(x)
    return x


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    upper = mutable(upper)
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        x = mutable(x)
        carry, y = f(carry, x)  # carry is the carryover
        ys.append(y)  # the `y`s get accumulated into a stacked array
    return carry, ys


def cumsum(res, el):
    res = res + el
    return res, res  # ("carryover", "accumulated")


@jit(backend="ms_backend")
def call_scan(a):
    result_init = 0
    return scan(cumsum, result_init, a)


def for_body_fun(i, val):
    x = i * 3
    x = x * val * val
    return x


@jit(backend="ms_backend")
def call_fori_loop(x):
    x = fori_loop(1, 100, for_body_fun, x)
    return x


@jit(backend="ms_backend")
def grad_for_loop(x):
    x = grad_all(call_fori_loop)(x)
    return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_grad_for_loop():
    """
    Feature: control flow function.
    Description: test gad of for_loop.
    Expectation: Null.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = Tensor([1], mstype.int32)
    x = grad_for_loop(x)
    print(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_fori_loop():
    """
    Feature: control flow function.
    Description: test fori_loop.
    Expectation: Null.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = Tensor([1], mstype.int32)
    x = call_fori_loop(x)
    print(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_scan():
    """
    Feature: control flow function.
    Description: test scap.
    Expectation: Null.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = np.array([1, 2, 3, 5, 7, 11, 13, 17])
    x, _ = call_scan(x)
    print(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_while_loop():
    """
    Feature: control flow function.
    Description: test while_loop.
    Expectation: Null.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = Tensor([1], mstype.int32)
    x = call_while_loop(x)
    print(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_grad_while_loop():
    """
    Feature: control flow function.
    Description: test grad of while_loop.
    Expectation: Null.
    """

    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = Tensor([1], mstype.int32)
    x = grad_while_loop(x)
    print(x)


@no_inline
def no_inline_fun(val):
    x = val * 3 + 2
    return x


@jit(backend="ms_backend")
def call_no_inline_fun(val):
    for _ in range(100):
        val = no_inline_fun(val)
    return val


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_no_inline_fun():
    """
    Feature: control flow function.
    Description: test no inline function.
    Expectation: Null.
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./ir")
    x = Tensor([1], mstype.int32)
    x = call_no_inline_fun(x)
    print(x)
