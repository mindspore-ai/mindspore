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
"""Test graph sequence operation with nested or irregular input/output"""
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sequence_compare_with_operation():
    """
    Feature: Enable sequence operations with nested or irregular inputs.
    Description: Sequence operations with nested or irregular inputs should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit(mode="PIJit")
    def foo(x, y):
        m = ((x, x+1), x+2)
        n = ((y, y-1), y+2)
        return m < n, m <= n, m > n, m >= n

    context.set_context(mode=context.PYNATIVE_MODE)
    a1, a2, a3, a4 = foo(Tensor([1]), Tensor([3]))
    assert a1
    assert a2
    assert not a3
    assert not a4
