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
import numpy as np
from mindspore.nn import LayerNorm
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.train.model import Model
from mindspore import log as logger
import pytest
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(Cell):
    def __init__(self, input_shape, begin_norm_axis, begin_params_axis, gamma, beta):
        super(Net, self).__init__()
        self.layernorm = LayerNorm(input_shape, begin_norm_axis, begin_params_axis, gamma, beta)

    def construct(self, input):
        x = self.layernorm(input)
        return x

def pt_me_layernorm(input_data, normalized_shape, gamma, beta, axis):
    net = Net(normalized_shape, begin_norm_axis=axis,
              begin_params_axis=axis,
              gamma=Tensor(gamma),
              beta=Tensor(beta))
    net.set_train()
    model = Model(net)
    out_me = model.predict(Tensor(input_data))
    logger.info("Check me result:")
    logger.info(out_me.asnumpy())

@pytest.mark.lower_bs
def test_normal_layernorm_1_128_1024_axis_2():
    """
    2 input[1, 128, 1024],normalized_shape=[128, 1024]
    """
    input_data = np.random.randn(1, 128, 1024).astype(np.float32)
    gamma = np.random.randn(1024).astype(np.float32)
    gamma.fill(1.1)
    beta = np.random.randn(1024).astype(np.float32)
    beta.fill(0.1)
    pt_me_layernorm(input_data, (1024, ), gamma, beta, 2)
