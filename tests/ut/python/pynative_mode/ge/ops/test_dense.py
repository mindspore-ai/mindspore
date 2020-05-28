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
""" test_dense """
import numpy as np

from mindspore import Tensor
from mindspore.nn import Dense
from ....ut_filter import non_graph_engine


@non_graph_engine
def test_dense():
    input_data = Tensor(np.ones([16, 8]).astype(np.float32))
    kernel = Tensor(np.ones([8, 8]).astype(np.float32))
    bias = Tensor(np.ones([8]).astype(np.float32))
    fc = Dense(8, 8, kernel, bias)
    output = fc(input_data)
    output_np = output.asnumpy()
    print(output_np)


@non_graph_engine
def test_dense_nobias():
    input_data = Tensor(np.ones([16, 8]).astype(np.float32))
    kernel = Tensor(np.ones([8, 8]).astype(np.float32))
    fc = Dense(8, 8, kernel, has_bias=False)
    output = fc(input_data)
    output_np = output.asnumpy()
    print(output_np)
