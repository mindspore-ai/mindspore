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
"""ut for batchnorm layer"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ....ut_filter import non_graph_engine


@non_graph_engine
def test_bn2d():
    """ut of nn.BatchNorm2d"""
    gamma = Tensor(np.random.randn(64).astype(np.float32) * 0.01)
    beta = Tensor(np.random.randn(64).astype(np.float32) * 0.01)
    moving_mean = Tensor(np.random.randn(64).astype(np.float32) * 0.01)
    moving_var = Tensor(np.random.randn(64).astype(np.float32) * 0.01)

    bn = nn.BatchNorm2d(num_features=64,
                        eps=1e-5,
                        momentum=0.1,
                        gamma_init=gamma,
                        beta_init=beta,
                        moving_mean_init=moving_mean,
                        moving_var_init=moving_var)

    # 3-channel RGB
    input_data = Tensor(np.random.randint(0, 10, [1, 64, 56, 56]).astype(np.float32))
    # for test in infer lib
    output = bn.construct(input_data)

    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))
