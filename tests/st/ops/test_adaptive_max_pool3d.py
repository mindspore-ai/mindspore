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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nn_adaptivemaxpool3d_with_int_oputput_size():
    """
    Feature: test nn.AdaptiveMaxPool3d
    Description: verify the result of AdaptiveMaxPool3d
    Expectation: assertion success
    """
    values = Tensor(np.arange(0, 64).reshape(1, 4, 4, 4).astype(np.int32))
    expected_result = np.array([[[[21, 23], [29, 31]], [[53, 55], [61, 63]]]], np.int32)
    output_size = 2
    net = nn.AdaptiveMaxPool3d(output_size)
    output = net(values)
    assert np.allclose(output.asnumpy(), expected_result)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_f_adaptivemaxpool3d_with_int_oputput_size():
    """
    Feature: test ops.adaptive_max_pool3d
    Description: verify the result of adaptive_max_pool3d
    Expectation: assertion success
    """
    values = Tensor(np.arange(0, 64).reshape(1, 4, 4, 4).astype(np.int32))
    expected_result = np.array([[[[21, 23], [29, 31]], [[53, 55], [61, 63]]]], np.int32)
    output_size = 2
    output = ops.adaptive_max_pool3d(values, output_size)
    assert np.allclose(output.asnumpy(), expected_result)
