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
""" test jit forbidden api in graph mode. """
import pytest

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_wrong_attr():
    """
    Feature: Syntax invalid attr
    Description: Graph syntax object's invalid attribute and method.
    Expectation: No Expectation
    """
    class Net(nn.Cell):
        def construct(self, x):
            x.abcd()
            return x

    x = Tensor([1, 2, 3], dtype=mstype.float32, const_arg=True)
    net = Net()
    with pytest.raises(AttributeError) as ex:
        net(x)
    assert "'Tensor' object has no attribute 'abcd'" or\
           "Tensor object has no attribute abcd" in str(ex.value)
