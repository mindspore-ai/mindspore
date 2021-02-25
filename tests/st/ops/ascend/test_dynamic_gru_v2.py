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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class DynamicGRUV2(nn.Cell):
    def __init__(self):
        super(DynamicGRUV2, self).__init__()
        self.dynamic_gru = P.DynamicGRUV2()

    def construct(self, x, weight_i, weight_h, bias_i, bias_h, init_h):
        return self.dynamic_gru(x, weight_i, weight_h, bias_i, bias_h, None, init_h)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_dynamic_gru_v2():
    x = Tensor(np.random.rand(2, 8, 64).astype(np.float16))
    weight_i = Tensor(np.random.rand(64, 48).astype(np.float16))
    weight_h = Tensor(np.random.rand(16, 48).astype(np.float16))
    bias_i = Tensor(np.random.rand(48).astype(np.float16))
    bias_h = Tensor(np.random.rand(48).astype(np.float16))
    init_h = Tensor(np.random.rand(8, 16).astype(np.float16))
    gru_net = DynamicGRUV2()
    output = gru_net(x, weight_i, weight_h, bias_i, bias_h, init_h)
    assert output[0].shape == (2, 8, 16)
