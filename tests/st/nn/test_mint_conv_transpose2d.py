# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.mint.nn as nn
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=398.0)
def test_conv2d_transpose_dyn():
    """
    Feature: Dynamic shape of BatchNorm1d
    Description: test BatchNorm1d with dynamic rank/shape.
    Expectation: success
    """
    in_channels = 16
    out_channels = 16
    kernel_size = 3
    stride = 2
    padding = 1
    net = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    input_0 = ms.Tensor(np.random.rand(20, 16, 50, 100).astype(np.float32))
    output_size_0 = (100, 200)
    inputs_0 = [input_0, output_size_0]

    input_1 = ms.Tensor(np.random.rand(16, 20, 40).astype(np.float32))
    output_size_1 = (40, 80)
    inputs_1 = [input_1, output_size_1]

    TEST_OP(net, [inputs_0, inputs_1],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True})
