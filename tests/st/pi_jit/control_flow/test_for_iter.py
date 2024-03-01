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
''' test FOR_ITER for pijit '''
import pytest
import dis
import sys
from mindspore import jit, Tensor
from mindspore._c_expression import get_code_extra


def for_range(x):
    res = 0
    for i in range(x):
        res = res + i
    return res


def for_enumerate(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(x):
        res = res + i
        res = res + v
    return x


def for_zip(x):
    x = [x, x, x]
    v = None
    for v in zip(x, x, x, x):
        pass
    return v


def for_mix(x):
    x = [x, x, x]
    res = 0
    for i, v in enumerate(list(zip(x, x, x, x))):
        res = res + i
        res = res + v[0]
    return res


def for_mix_with_sideeffect(x):
    x = [x, x, x]
    z = zip(list(enumerate(x)))
    for i in z:
        if i[0] == 1:
            break
    return list(z)


@pytest.mark.level0
@pytest.mark.skip
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [for_range, for_enumerate, for_mix, for_zip])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_for_iter_unrolling(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    config = {"loop_unrolling": True}
    excepted = func(param)
    result = jit(fn=func, mode="PIJit", jit_config=config)(param)
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]

    # just unrolling loop in python 3.9
    if sys.version_info.major == 3 and sys.version_info.minor == 9:
        for i in dis.get_instructions(new_code):
            assert i.opname != "FOR_ITER"

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result


@pytest.mark.level0
@pytest.mark.skip
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func', [for_mix_with_sideeffect])
@pytest.mark.parametrize('param', [1, Tensor([1])])
def test_not_implement_for_iter(func, param):
    """
    Feature: Test loop unrolling
    Description: Test loop unrolling
    Expectation: No exception.
    """
    config = {"loop_unrolling": True}
    excepted = func(param)
    result = jit(fn=func, mode="PIJit", jit_config=config)(param)
    jcr = get_code_extra(func)

    assert jcr["stat"] == "GRAPH_CALLABLE"
    assert jcr["code"]["call_count_"] > 0
    assert excepted == result
