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

import mindspore as ms
from mindspore.ops import Primitive
from mindspore.common import Tensor
import numpy as np
import pytest

tensor_move = Primitive('TensorMove')

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_tensor_move():
    """
    Feature: test tensor_move.
    Description: Operation selects input is Tensor with bfloat16 type.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.random.randn(2, 1, 120), ms.bfloat16)
    tensor_move_data = tensor_move(input_x)
    input_x_np = input_x.float().asnumpy()
    tensor_move_data_np = tensor_move_data.float().asnumpy()
    np.allclose(input_x_np, tensor_move_data_np, 0)
