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
""" test graph fallback """
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, ms_function, context
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


class ControlNet(nn.Cell):
    def inner_function_1(self, a, b):
        return a + b

    def inner_function_2(self, a, b):
        return a - b

    def construct(self, x):
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        if a + b > x:
            return self.inner_function_1(a, b)
        return self.inner_function_2(a, b)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_control_sink_tensor():
    """
    Feature: Fallback feature: support define Tensor in Class construct.
    Description: Fallback feature: support define Tensor in Class construct.
    Expectation: Fallback feature: support define Tensor in Class construct.
    """
    x = Tensor(np.array(1), mstype.int32)
    net = ControlNet()
    output = net(x)
    output_expect = Tensor(9, mstype.int32)
    assert output == output_expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_np_tensor_list():
    """
    Feature: Fallback feature
    Description: support Basic method of Tensor list.
    Expectation: No exception.
    """
    @ms_function
    def np_tensor_list():
        a = Tensor(np.array(4), mstype.int32)
        b = Tensor(np.array(5), mstype.int32)
        c = Tensor(np.array(6), mstype.int32)
        tensor_list = [a, b]
        for tensor in tensor_list:
            print(tensor)
        tensor_list.append(tensor_list[-1] + c)
        return tensor_list

    tensor_list = np_tensor_list()
    print("tensor_list:", tensor_list)
    assert len(tensor_list) == 3
