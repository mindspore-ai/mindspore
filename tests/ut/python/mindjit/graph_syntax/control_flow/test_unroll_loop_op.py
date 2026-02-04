# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import jit, context
import mindspore.ops as ops

context.set_context(mode=context.GRAPH_MODE)


def test_while_simple_loop():
    """
    Feature: control flow
    Description: Using WhileLoopEvaluator to handle ops.WhileLoop operation
    Expectation: No exception.
    """

    def simple_loop_func(init_value):
        init_value = init_value + 1
        return init_value


    def simple_cond_func(init_value):
        return init_value < 100

    @jit
    def test_simple_while_loop_inner(init_state):
        whileop = ops.WhileLoop()
        result = whileop(simple_cond_func, simple_loop_func, init_state)
        return result

    init = 0
    result = test_simple_while_loop_inner(init)
    assert result == 100

def test_scan_simple_loop():
    """
    Feature: control flow
    Description: Using ScanEvaluator to handle ops.Scan operation
    Expectation: No exception.
    """

    def simple_loop_func(res, el):
        res = res + el
        return res, res

    @jit
    def test_simple_scan_inner(array):
        result_init = 0
        scan_op = ops.Scan()
        return scan_op(simple_loop_func, result_init, array, len(array), False)

    array = [1, 2, 3, 4]
    result = test_simple_scan_inner(array)
    assert result == (10, [1, 3, 6, 10])

def test_fori_loop():
    """
    Feature: control flow
    Description: Using ForiLoopEvaluator to handle ops.ForiLoop operation
    Expectation: No exception.
    """

    def loop_func(index, val):
        return index + val

    @jit
    def test_fori_loop_inner(result_init):
        fori_loop = ops.ForiLoop()
        return fori_loop(0, 10, loop_func, result_init)

    result_init = 0
    result = test_fori_loop_inner(result_init)
    assert result == 45
