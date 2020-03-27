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
""" test Activations """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ....ut_filter import non_graph_engine


# test activation
@non_graph_engine
def test_relu():
    relu = nn.ReLU()
    input_data = Tensor(np.random.rand(1, 3, 4, 4).astype(np.float32) - 0.5)
    output = relu.construct(input_data)
    output_np = output.asnumpy()
    print(output_np)
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


@non_graph_engine
def test_softmax_axis_none():
    layer = nn.Softmax()
    x = Tensor(np.random.rand(1, 3, 4, 4).astype(np.float32))
    output = layer.construct(x)
    output_np = output.asnumpy()
    print(output_np)
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))
