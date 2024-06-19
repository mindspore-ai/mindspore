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
"""run dynamic shape test"""
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, jit, context, Symbol
from mindspore.nn import Cell
from mindspore._c_expression import get_code_extra
from .share.utils import match_array

s=Symbol(max=10,min=1)
g_relu=nn.ReLU()

class SignatureNet(Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    @jit(mode="PIJit", input_signature=(Tensor(shape=(s,None), dtype=ms.float32)))
    def construct(self, a):
        return self.relu(a)

@jit(mode="PIJit", input_signature=(Tensor(shape=(None, s), dtype=ms.float32)))
def signature_test(a):
    return g_relu(a)

@jit(mode="PIJit", input_signature=((Tensor(shape=(None, s), dtype=ms.float32), Tensor(shape=(None, s), dtype=ms.float32)), None))
def signature_tuple_test(a, b):
    return g_relu(a[0])

@jit(mode="PIJit", jit_config={"enable_dynamic_shape": True, "limit_graph_count": 1})
def dynamic_shape_test(a, b):
    return a + b

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dynamic_shape_case():
    """
    Feature: Method DynamicShape Testing
    Description: Test dyanmicshape function to check whether it works.
    Expectation: The result of the case should dump the dynamic shape ir at last.
                 'enable_dynamic_shape' flag is used to enable dynamic shape when calling 3 times for different shape.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1])
    b = Tensor([2])
    expect = Tensor([3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1])
    b = Tensor([2, 2])
    expect = Tensor([3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)
    a = Tensor([1, 1, 1])
    b = Tensor([2, 2, 2])
    expect = Tensor([3, 3, 3])
    c = dynamic_shape_test(a, b)
    assert all(c == expect)


@pytest.mark.skip(reason="adapter later")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_signature_case():
    """
    Feature: Method DynamicShape DynamicSymbolic In Signature Testing
    Description: Test dynamicshape and dynamicsymbolic in signature function to check whether it works.
    Expectation: The result of the case should compile the graph no more than once.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    t1 = Tensor([[1.1, 1.1],[2.2,2.2]], dtype=ms.float32)
    t2 = Tensor([[1.1],[2.2]], dtype=ms.float32)
    res1 = signature_test(t1)
    match_array(res1, t1)
    res2 = signature_test(t2)
    match_array(res2, t2)
    res1 = signature_tuple_test((t1, t2), 1)
    match_array(res1, t1)
    res2 = signature_tuple_test((t2, t1), 1)
    match_array(res2, t2)
    res1 = SignatureNet()(t1)
    match_array(res1, t1)
    res2 = SignatureNet()(t2)
    match_array(res2, t2)
    jcr1 = get_code_extra(signature_test)
    assert(jcr1["compile_count_"] == 1)
    jcr2 = get_code_extra(SignatureNet().construct)
    assert(jcr2["compile_count_"] == 1)
