# Copyright 2021 Huawei Technologies Co., Ltd
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
import scipy as scp
import numpy as np

import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
np.random.seed(0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('lower', [True])
@pytest.mark.parametrize('dtype', [np.float64])
def test_cholesky(lower: bool, dtype: Generic):
    """
    Feature: ALL TO ALL
    Description:  test cases for cholesky [N,N]
    Expectation: the result match scipy cholesky
    """
    a = np.array([[4, 12, -6], [12, 37, -43], [-16, -43, 98]], dtype=dtype)
    tensor_a = Tensor(a)
    scp_c = scp.linalg.cholesky(a, lower=lower)
    mscp_c = cholesky(tensor_a, lower=lower)
    assert np.allclose(scp_c, mscp_c.asnumpy())
