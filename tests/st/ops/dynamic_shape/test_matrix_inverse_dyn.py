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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either matrix_inverseress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""test matrix_inverse dynamic shape"""

import numpy as np
import pytest

from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import operations as P

np.random.seed(1)


class NetMatrixInverse(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matrix_inverse = P.MatrixInverse()

    def construct(self, x):
        return self.matrix_inverse(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_matrix_inverse():
    """
    Feature: test matrix_inverse op
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    x_np = np.random.uniform(-2, 2, (3, 4, 4)).astype(np.float32)
    x = Tensor(x_np)
    x_dyn = Tensor(shape=[None, None, None], dtype=x.dtype)

    context.set_context(device_target="GPU")
    matrix_inverse = NetMatrixInverse()
    matrix_inverse.set_inputs(x_dyn)
    output0 = matrix_inverse(x)
    assert output0.asnumpy().shape == (3, 4, 4)
