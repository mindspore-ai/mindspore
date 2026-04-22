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

from tests.mark_utils import arg_mark
import mindspore
from mindspore import Tensor, nn
import numpy as np


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_pynative_implicit_cast():
    """
    Feature: Pynative implicit cast
    Description: Test pynative implicit cast
    Expectation: run success
    """

    net = nn.Conv1dTranspose(3, 64, 4, has_bias=False, weight_init='normal', pad_mode='pad')
    x = Tensor(np.ones([1, 3, 50]), mindspore.float32)
    output = net(x).asnumpy().shape
    assert output == (1, 64, 53)
