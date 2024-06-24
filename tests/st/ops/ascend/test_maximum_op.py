# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np

import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_maximum_tensor_api_modes(mode):
    """
    Feature: Test maximum tensor api.
    Description: Test maximum tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor([1.0, 5.0, 3.0], mstype.float32)
    y = Tensor([4.0, 2.0, 6.0], mstype.float32)
    output = x.maximum(y)
    expected = np.array([4., 5., 6.], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_maximum_tensor_with_bfloat16():
    """
    Feature: test maximum on Ascend
    Description: used two Tensor with type bfloat16.
    Expectation: result match to numpy result.
    """
    x_np = np.random.randn(3, 10).astype(np.float32)
    y_np = np.random.randn(3, 10).astype(np.float32)
    input_x_ms = Tensor(x_np, ms.bfloat16)
    input_y_ms = Tensor(y_np, ms.bfloat16)
    output = input_x_ms.maximum(input_y_ms)
    print(output.float().asnumpy())
    output_np = np.maximum(input_x_ms.float().asnumpy(), input_y_ms.float().asnumpy())
    assert np.allclose(output.float().asnumpy(), output_np, rtol=0.004, atol=0.004)
