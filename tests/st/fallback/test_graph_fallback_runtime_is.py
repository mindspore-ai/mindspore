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
from mindspore import Tensor, context
import mindspore.nn as nn
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_is_none_asnumpy():
    """
    Feature: Support is.
    Description: Support is in fallback runtime.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            input_x = Tensor([1], dtype=mstype.int32).asnumpy()
            is_not_res = input_x is not None
            is_res = input_x is None
            return is_not_res, is_res

    net = Net()
    res = net()
    assert res[0], not res[1]
