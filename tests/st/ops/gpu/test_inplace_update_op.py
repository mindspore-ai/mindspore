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

# This example should be run with multiple processes.

# Please refer to the Programming Guide > Distributed Training -> Distributed Parallel Usage Example

# on mindspore.cn and focus on the contents of these three parts: Configuring Distributed Environment

# Variables, Calling the Collective Communication Library, Running the Script.

import pytest

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetInplaceUpdate(nn.Cell):
    def __init__(self, indices):
        super(NetInplaceUpdate, self).__init__()
        self.indices = indices
        self.inplace_update = P.InplaceUpdate(self.indices)

    def construct(self, input_x1, input_x2):
        output = self.inplace_update(input_x1, input_x2)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inplace_update_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for InplaceUpdate
    Expectation: the result match to expect result
    """
    inplace_update = NetInplaceUpdate(indices=(0, 1))
    x1 = Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float16)
    x2 = Tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float16)
    output = inplace_update(x1, x2)
    expect = Tensor([[0.5, 1.0], [1.0, 1.5], [5, 6]], mindspore.float16)
    assert (output.asnumpy() == expect).all()
