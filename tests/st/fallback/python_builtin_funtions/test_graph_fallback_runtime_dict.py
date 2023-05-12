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

import pytest
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_dict_runtime():
    """
    Feature: JIT Fallback
    Description: Test dict() in fallback runtime
    Expectation:No exception
    """

    @jit
    def foo(x1, x2, x3, x4):
        dict_x1 = dict(zip(['one', 'two', 'three'], [1, 2, x1]))
        dict_x1['two'] = [1, 2, 3, x2]
        dict_x2 = dict([("one", 1), ("two", x3)])
        dict_x3 = dict(one=1, two=x4)
        return dict_x1, dict_x2, dict_x3

    x1 = Tensor([3])
    x2 = Tensor([4])
    x3 = Tensor([5])
    x4 = Tensor([6])
    out1, out2, out3 = foo(x1, x2, x3, x4)
    assert out1 == {'one': 1, 'two': [1, 2, 3, Tensor([4])], 'three': Tensor([3])}
    assert out2 == {"one": 1, "two": Tensor([5])}
    assert out3 == {"one": 1, "two": Tensor([6])}
