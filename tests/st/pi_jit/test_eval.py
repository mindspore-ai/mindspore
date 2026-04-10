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
"""test eval function"""
from mindspore import jit
from tests.mark_utils import arg_mark


def func2(x):
    return x + x

@arg_mark(plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
def test_pijit_catch_func():
    """
    Feature: Test eval global func
    Description: Test eval.
    Expectation: process eval success.
    """

    def func(x):
        out = eval(f"func{x}(x)")
        return out

    got = jit(function=func, capture_mode="bytecode")(2)
    expect = func(2)
    assert got == expect
