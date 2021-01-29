# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test cases for gamma distribution"""

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import dtype as ms
import pytest

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class GammaMean(nn.Cell):
    def __init__(self, concentration, rate, seed=10, dtype=ms.float32, name='Gamma'):
        super().__init__()
        self.b = msd.Gamma(concentration, rate, seed, dtype, name)

    def construct(self):
        out1 = self.b.mean()
        out2 = self.b.mode()
        out3 = self.b.var()
        out4 = self.b.entropy()
        out5 = self.b.sd()
        return out1, out2, out3, out4, out5



@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_probability_gamma_mean_cdoncentration_rate_rand_2_ndarray():
    concentration = np.random.uniform(0.0001, 100, size=(1024, 512, 7, 7)).astype(np.float32)
    rate = np.random.uniform(0.0001, 100, size=(1024, 512, 7, 7)).astype(np.float32)
    net = GammaMean(concentration, rate)
    net()
