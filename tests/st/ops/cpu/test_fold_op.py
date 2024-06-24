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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

from mindspore import Tensor
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F


def test_fold_functional_api():
    """
    Feature: test fold functional API.
    Description: test case for fold functional API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones([16, 64, 25]), mstype.float32)
    output_size = Tensor([8, 8], mstype.int32)
    output = F.fold(x, output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
    expected_shape = (16, 16, 8, 8)
    assert output.dtype == x.dtype
    assert output.shape == expected_shape


def test_fold_tensor_api():
    """
    Feature: test fold tensor API.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    x = Tensor(np.ones([16, 64, 25]), mstype.float32)
    output_size = Tensor([8, 8], mstype.int32)
    output = x.fold(output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])
    expected_shape = (16, 16, 8, 8)
    assert output.dtype == x.dtype
    assert output.shape == expected_shape

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fold_functional_api_with_invalid_output_size():
    """
    Feature: test fold tensor API with invalid output size.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = Tensor(np.ones([16, 64, 4]), mstype.float32)
    output_size = Tensor([6, -1], mstype.int32)
    with pytest.raises(ValueError, match=r"the value of 'output_size' must not be negative"):
        F.fold(x, output_size, kernel_size=[2, 2], dilation=[2, 2], padding=[2, 2], stride=[2, 2])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fold_tensor_functional_api_modes():
    """
    Feature: test fold tensor and functional APIs for different modes.
    Description: test case for fold tensor API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_fold_functional_api()
    test_fold_tensor_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_fold_functional_api()
    test_fold_tensor_api()


if __name__ == '__main__':
    test_fold_tensor_functional_api_modes()
    test_fold_functional_api_with_invalid_output_size()
