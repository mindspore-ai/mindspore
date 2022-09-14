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

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import operations as P


def test_half_tensor_api():
    """
    Feature: test half tensor API.
    Description: test half dtype tensor conversion.
    Expectation: the input and output shape should be same. output dtype should be float16.
    """
    dtype_op = P.DType()
    x = Tensor(np.ones([2, 3, 1]), ms.float32)
    output = x.half()
    assert x.shape == output.shape
    assert dtype_op(output) == ms.float16


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_half_tensor_api_modes():
    """
    Feature: test half tensor API for different modes.
    Description: test half dtype tensor conversion.
    Expectation: the input and output shape should be same. output dtype should be float16.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_half_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_half_tensor_api()


if __name__ == '__main__':
    test_half_tensor_api_modes()
