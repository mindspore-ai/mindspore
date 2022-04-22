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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from scipy import special


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.level0
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
def test_erf(mode, shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for erf
    Expectation: the result match to scipy
    """
    context.set_context(mode=mode, device_target="CPU")
    erf = P.Erf()
    input_x = np.random.randn(*shape).astype(dtype)
    output = erf(Tensor(input_x))
    expect_output = special.erf(input_x)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)
