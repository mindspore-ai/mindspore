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


def test_to_tensor_api(dtype):
    """
    Feature: test to tensor API.
    Description: test to API for dtype tensor conversion.
    Expectation: the input and output shape should be same. output dtype should be same as op arg.
    """
    dtype_op = P.DType()
    x = Tensor(np.ones([2, 3, 1]))
    output = x.to(dtype)
    assert x.shape == output.shape
    assert dtype_op(output) == dtype


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_to_tensor_api_modes():
    """
    Feature: test to tensor API for different modes.
    Description: test to API for dtype tensor conversion.
    Expectation: the input and output shape should be same. output dtype should be same as op arg.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_to_tensor_api(ms.bool_)
    test_to_tensor_api(ms.float16)
    test_to_tensor_api(ms.float32)
    test_to_tensor_api(ms.float64)
    test_to_tensor_api(ms.int8)
    test_to_tensor_api(ms.uint8)
    test_to_tensor_api(ms.int16)
    test_to_tensor_api(ms.uint16)
    test_to_tensor_api(ms.int32)
    test_to_tensor_api(ms.int64)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_to_tensor_api(ms.bool_)
    test_to_tensor_api(ms.float16)
    test_to_tensor_api(ms.float32)
    test_to_tensor_api(ms.float64)
    test_to_tensor_api(ms.int8)
    test_to_tensor_api(ms.uint8)
    test_to_tensor_api(ms.int16)
    test_to_tensor_api(ms.uint16)
    test_to_tensor_api(ms.int32)
    test_to_tensor_api(ms.int64)
