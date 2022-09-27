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


class NetLerp(nn.Cell):
    def __init__(self):
        super(NetLerp, self).__init__()
        self.lerp = P.Lerp()

    def construct(self, x, y, z):
        return self.lerp(x, y, z)


def lerp_compute(x, y, z):
    return x + z * (y - x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lerp_fp32():
    """
    Feature: Lerp function.
    Description:  The Tensor of float16, float32, float64, the scalar of float.
    Expectation: Returns the lerp sparse tensor of the input.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    lerp = NetLerp()
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float16)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float16)
    z = 0.7
    assert np.allclose(lerp(Tensor(x), Tensor(y), z).asnumpy(), lerp_compute(x, y, z), 1e-03, 1e-03)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float32)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float32)
    z = 0.7
    assert np.allclose(lerp(Tensor(x), Tensor(y), z).asnumpy(), lerp_compute(x, y, z), 1e-04, 1e-04)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float64)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float64)
    z = 0.7
    assert np.allclose(lerp(Tensor(x), Tensor(y), z).asnumpy(), lerp_compute(x, y, z), 1e-05, 1e-05)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float16)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float16)
    z = np.array([[1., -1.2, 0.9], [0.1, 2., 1.]], dtype=np.float16)
    assert np.allclose(lerp(Tensor(x), Tensor(y), Tensor(z)).asnumpy(), lerp_compute(x, y, z), 1e-03, 1e-03)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float32)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float32)
    z = np.array([[1., -1.2, 0.9], [0.1, 2., 1.]], dtype=np.float32)
    assert np.allclose(lerp(Tensor(x), Tensor(y), Tensor(z)).asnumpy(), lerp_compute(x, y, z), 1e-04, 1e-04)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float64)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float64)
    z = np.array([[1., -1.2, 0.9], [0.1, 2., 1.]], dtype=np.float64)
    assert np.allclose(lerp(Tensor(x), Tensor(y), Tensor(z)).asnumpy(), lerp_compute(x, y, z), 1e-05, 1e-05)
