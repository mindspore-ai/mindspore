# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype

skip_flag = ms.get_context("device_target") == "Ascend"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.skipif(skip_flag, reason="I84LSH/I892G7")
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_fold(mode):
    """
    Feature: nn.Fold
    Description: Verify the result of Fold
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(input_data=np.random.rand(16, 64, 25), dtype=mstype.float32)
    output_size = Tensor(input_data=[8, 8], dtype=mstype.int32)
    fold_op = nn.Fold(output_size, [2, 2], [2, 2], [2, 2], [2, 2])
    output = fold_op(x)
    expect_shape = (16, 16, 8, 8)
    assert np.allclose(output.shape, expect_shape)
