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
''' test resolve of listcomp, dictcomp, setcomp, genexpr code in pijit '''
import pytest
import types
from mindspore import jit


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_listcomp():
    """
    Feature: Generator expression unrolling
    Description: Test code <listcomp> unrolling
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config={"interpret_captured_code": True})
    def func(a, b, c):
        x = [a, b, c]
        a = [(k if k is None else (i for i in k))
             for k in x if not isinstance(k, tuple)]
        return a

    x = [None, "123", tuple()]
    x = func(*x)
    assert len(x) == 2
    assert x[0] is None
    assert type(x[1]) is types.GeneratorType
