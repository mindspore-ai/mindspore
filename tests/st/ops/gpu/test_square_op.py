# Copyright 2020-2022 Huawei Technologies Co., Ltd
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


class SquareNet(nn.Cell):
    def __init__(self):
        super(SquareNet, self).__init__()
        self.ops = P.Square()

    def construct(self, x):
        return self.ops(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_square_normal(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Square
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.random.rand(2, 3, 4, 4).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(2, 3, 1, 5, 4, 4).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
    x_np = np.random.rand(2).astype(np.float32)
    output_ms = P.Square()(Tensor(x_np))
    output_np = np.square(x_np)
    assert np.allclose(output_ms.asnumpy(), output_np)
