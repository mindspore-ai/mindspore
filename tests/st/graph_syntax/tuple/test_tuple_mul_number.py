# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test tuple mul number """

import pytest
from mindspore import context, jit

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tuple_mul_non_integer_number():
    """
    Feature: tuple multiple non-integer number.
    Description: tuple can only multiply integer number.
    Expectation: Raise TypeError.
    """
    @jit
    def foo():
        x = (1, 2, 3, 4)
        return x * 2.0
    with pytest.raises(TypeError) as error_info:
        foo()
    assert "can't multiply sequence by non-int of type" in str(error_info)
