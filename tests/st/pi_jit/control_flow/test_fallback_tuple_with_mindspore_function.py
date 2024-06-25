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
''' test fallback tuple with mindspore function '''
import numpy as np
import mindspore.nn as nn
from mindspore import jit, Tensor, context
from mindspore.ops import Primitive
import pytest
from tests.mark_utils import arg_mark


@pytest.mark.skip
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_tuple_with_mindspore_function():
    """
    Feature: JIT Fallback
    Description: Test fallback when local input has tuple with mindspore function type, such as Cell, Primitive.
    Expectation: No exception.
    """
    def isinstance_fn(a, base_type):
        mro = type(a).mro()
        for i in base_type:
            if i in mro:
                return True
        return False

    @jit(mode="PIJit")
    def foo():
        return isinstance_fn(np.array(1), (np.ndarray, nn.Cell, Primitive))

    context.set_context(mode=context.PYNATIVE_MODE)
    assert foo()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_prune_if_in_while():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(mode="PIJit")
    def convert_list(list_of_tensor):
        if isinstance(list_of_tensor, list):
            tuple_of_tensor = ()
            for tensor in list_of_tensor:
                tuple_of_tensor += (tensor,)
            return tuple_of_tensor
        return list_of_tensor
    context.set_context(mode=context.PYNATIVE_MODE)
    res = convert_list([Tensor(0), Tensor(0)])
    assert len(res) == 2
