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

# This example should be run with multiple processes.

# Please refer to the Programming Guide > Distributed Training -> Distributed Parallel Usage Example

# on mindspore.cn and focus on the contents of these three parts: Configuring Distributed Environment

# Variables, Calling the Collective Communication Library, Running the Script.

import pytest

import mindspore
from mindspore import context
from mindspore import Tensor, ops
from mindspore.ops import operations as P
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetInplaceUpdateV2(nn.Cell):
    def __init__(self, x, v):
        super(NetInplaceUpdateV2, self).__init__()
        self.x = x
        self.v = v
        self.inplace_update_v2 = P.InplaceUpdateV2()

    def construct(self, indices):
        output = self.inplace_update_v2(self.x, indices, self.v)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inplace_update_fp16():
    """
    Feature: InplaceUpdateV2
    Description: test cases for InplaceUpdateV2
    Expectation: the result match to expect result
    """
    x = Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float16)
    v = Tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float16)
    inplace_update_v2 = NetInplaceUpdateV2(x, v)
    indices = Tensor(shape=[None], dtype=mindspore.int32)
    inplace_update_v2.set_inputs(indices)
    real_indices = Tensor([0, 1], dtype=mindspore.int32)

    output = inplace_update_v2(real_indices)
    expect = Tensor([[0.5, 1.0], [1.0, 1.5], [5, 6]], mindspore.float16)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inplace_update_tensor():
    """
    Feature: InplaceUpdateV2
    Description: test tensor interface for InplaceUpdateV2
    Expectation: the result match to expect result
    """
    x = Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float16)
    v = Tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float16)
    real_indices = Tensor([0, 1], dtype=mindspore.int32)

    output = x.inplace_update(v, real_indices)
    expect = Tensor([[0.5, 1.0], [1.0, 1.5], [5, 6]], mindspore.float16)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inplace_update_functional():
    """
    Feature: InplaceUpdateV2
    Description: test function interface for InplaceUpdateV2
    Expectation: the result match to expect result
    """
    x = Tensor([[1, 2], [3, 4], [5, 6]], mindspore.float16)
    v = Tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float16)
    real_indices = Tensor([0, 1], dtype=mindspore.int32)

    output = ops.inplace_update(x, v, real_indices)
    expect = Tensor([[0.5, 1.0], [1.0, 1.5], [5, 6]], mindspore.float16)
    assert (output.asnumpy() == expect).all()
