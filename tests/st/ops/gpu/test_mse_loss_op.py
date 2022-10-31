# Copyright 2022 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np

from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mse_loss_functional_api_modes(mode):
    """
    Feature: Test mse_loss functional api.
    Description: Test mse_loss functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    logits = Tensor([1, 2, 3], mstype.float32)
    labels = Tensor([[1, 1, 1], [1, 2, 2]], mstype.float32)
    output = F.mse_loss(logits, labels, reduction='none')
    expected = np.array([[0., 1., 4.], [0., 0., 1.]], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)
