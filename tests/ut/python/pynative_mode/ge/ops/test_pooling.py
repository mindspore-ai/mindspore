# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
test pooling api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ....ut_filter import non_graph_engine


@non_graph_engine
def test_avgpool():
    """ test_avgpool """
    kernel_size = 3
    stride = 2
    avg_pool = nn.AvgPool2d(kernel_size, stride)
    assert avg_pool.kernel_size == 3
    assert avg_pool.stride == 2
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 6, 6]).astype(np.float32))
    output = avg_pool(input_data)
    output_np = output.asnumpy()
    print(output_np)
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


@non_graph_engine
def test_maxpool2d():
    """ test_maxpool2d """
    kernel_size = 3
    stride = 3

    max_pool = nn.MaxPool2d(kernel_size, stride)
    assert max_pool.kernel_size == 3
    assert max_pool.stride == 3
    input_data = Tensor(np.random.randint(0, 255, [1, 3, 6, 6]).astype(np.float32))
    output = max_pool(input_data)
    output_np = output.asnumpy()
    print(output_np)
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))
