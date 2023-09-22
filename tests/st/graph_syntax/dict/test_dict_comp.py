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
""" test_dict_get """
import pytest
from mindspore import context, jit

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_generated_from_list():
    """
    Feature: dict comp.
    Description: support dict comp, which is generated from list.
    Expectation: No exception.
    """
    @jit
    def dict_generate():
        x = [('a', 1), ('b', 2), ('c', 3)]
        res = {k: v for (k, v) in x if v > 1}
        return res

    out = dict_generate()
    assert out == {'b': 2, 'c': 3}


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dict_generated_from_dict():
    """
    Feature: dict comp.
    Description: support dict comp, which is generated from dict.
    Expectation: No exception.
    """
    @jit
    def dict_generate():
        x = {'a': 1, 'b': 2, 'c': 3}
        res = {k: v for (k, v) in x.items() if v > 1}
        return res

    out = dict_generate()
    assert out == {'b': 2, 'c': 3}


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dictcomp_with_multiple_generator():
    """
    Feature: dict comp.
    Description: support dict comp, which is generated from dict.
    Expectation: No exception.
    """
    @jit
    def dict_generate():
        x = ({'a': 1, 'b': 2}, {'d': 1, 'e': 2}, {'g': 1, 'h': 2})
        res = {k: v for y in x for (k, v) in y.items()}
        return res

    with pytest.raises(TypeError) as ex:
        dict_generate()
    assert "The 'generators' supports 1 'comprehension' in DictComp/GeneratorExp" in str(
        ex.value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_dictcomp_with_pyexecute_input():
    """
    Feature: dict comp.
    Description: support dict comp, which is generated from dict.
    Expectation: No exception.
    """
    @jit
    def dict_generate():
        d = {'a': 1, 'b': 2, 'c': 3, 'A': 4, 'B': 5, 'D': 6}
        res = {i.lower(): d.get(i.lower(), 0) + d.get(i.upper(), 0) for i in d}
        return res

    out = dict_generate()
    assert out == {'a': 5, 'b': 7, 'c': 3, 'd': 6}
