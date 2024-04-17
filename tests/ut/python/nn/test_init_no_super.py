# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import context, nn


def test_init_no_super():
    """
    Feature: Support use Cell attribute.
    Description: Test method init no super in instance
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)

    class InitNoSuper(nn.Cell):
        def __init__(self):
            self.a = 1

        def construct(self):
            return self.a

    net = InitNoSuper()
    with pytest.raises(AttributeError) as info:
        net()
    assert "super().__init__()" in str(info.value)
