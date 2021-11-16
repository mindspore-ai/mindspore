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
from typing import Generic
import mindspore.context as context
from mindspore import Tensor
from mindspore.scipy.linalg import cholesky
import numpy as np
import scipy as scp
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('lower', [True])
@pytest.mark.parametrize('dtype', [np.float64])
def test_scipy_cholesky(lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for new scipy cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(dtype=dtype)
    tensor_a = Tensor(a)
    output = cholesky(tensor_a, lower=lower)
    expect = scp.linalg.cholesky(a, lower=lower)
    assert np.allclose(expect, output.asnumpy())
