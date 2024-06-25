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
from tests.st.compiler.control.cases_register import case_register
import mindspore.context as context
from mindspore import Tensor, jit
from mindspore.common import dtype as mstype


@jit
def hof(x):
    def f(x):
        return x + 3

    def k(x):
        return x - 1

    def g(x):
        if x < 5:
            return f
        return k

    ret = g(x)(x)
    return ret


@case_register.level1
@case_register.target_ascend
def test_fun_fun():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([10], mstype.int32)
    ret = hof(x)
    expect = Tensor([9], mstype.int32)
    assert ret == expect


if __name__ == "__main__":
    test_fun_fun()
