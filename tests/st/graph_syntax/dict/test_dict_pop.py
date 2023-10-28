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
""" test_dict_pop """
import pytest
from mindspore import context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dict_pop():
    """
    Feature: dict get.
    Description: support dict get, which the return is None.
    Expectation: No exception.
    """
    class Net(Cell):
        def construct(self, **kwargs):
            kwargs.pop('label1')
            return kwargs

    net = Net()
    dict_input = {'label1': 1, 'label2': 2}
    out = net(**dict_input)
    assert out == {'label2': 2}
