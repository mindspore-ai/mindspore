# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

import numpy as np
import pytest

from mindspore import ops
from mindspore import context
from mindspore import Tensor


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_pinv(dtype):
    """
    Feature: test ops.pinv functional interface.
    Description: test cases for pinv for float32 and float64
    Expectation: the result match with numpy result.
    """
    x0 = np.array([[3., 8.], [2., 2.]], dtype=dtype)
    x1 = np.array([[2., 3.], [4., 6.]], dtype=dtype)
    x2 = np.array([[0., 1.], [1., 1.], [1., 0.]], dtype=dtype)

    if dtype == np.float32:
        loss = 1e-4
    else:
        loss = 1e-5

    ms_res0 = ops.pinv(Tensor(x0)).asnumpy()
    ms_res1 = ops.pinv(Tensor(x1)).asnumpy()
    ms_res2 = ops.pinv(Tensor(x2)).asnumpy()

    np_res0 = np.linalg.pinv(x0)
    np_res1 = np.linalg.pinv(x1)
    np_res2 = np.linalg.pinv(x2)

    assert np.allclose(np_res0, ms_res0, loss)
    assert np.allclose(np_res1, ms_res1, loss)
    assert np.allclose(np_res2, ms_res2, loss)
