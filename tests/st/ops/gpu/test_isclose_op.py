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


class NetIsClose(nn.Cell):
    def __init__(self, rtol=1e-05, atol=1e-08, equal_nan=False):
        super(NetIsClose, self).__init__()
        self.isclose = P.IsClose(rtol, atol, equal_nan)

    def construct(self, x, y):
        return self.isclose(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_isclose_fp32():
    """
    Feature: IsClose function.
    Description:  The Tensor of float16, float32, float64, int8, int16, int32 and uint8.
    Expectation: Returns the isclosed sparse tensor of the input.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    isclose = NetIsClose(equal_nan=False)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float16)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float16)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y), 1e-03, 1e-03)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float32)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float32)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y), 1e-04, 1e-04)
    x = np.array([[1., -1., 2.], [3.1, 2, 1.]], dtype=np.float64)
    y = np.array([[1.2, -1., 2.1], [3., 2., 1.1]], dtype=np.float64)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y), 1e-05, 1e-05)
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int8)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int8)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y))
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int16)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int16)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y))
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int32)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int32)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y))
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.int64)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.int64)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y))
    x = np.array([[1, -1, 2], [3, 2, 1]], dtype=np.uint8)
    y = np.array([[6, -1., 2], [3, 3, 2]], dtype=np.uint8)
    assert np.allclose(isclose(Tensor(x), Tensor(y)).asnumpy(), np.isclose(x, y))
