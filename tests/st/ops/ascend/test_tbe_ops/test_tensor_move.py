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
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore.ops import Primitive
from mindspore.common import Tensor
import numpy as np
import pytest

tensor_move = Primitive('TensorMove')

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
