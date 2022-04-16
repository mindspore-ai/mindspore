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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetInv(nn.Cell):
    def __init__(self):
        super(NetInv, self).__init__()
        self.inv = P.Inv()

    def construct(self, x):
        return self.inv(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol',
                         [(np.int32, 1.0e-7), (np.float16, 1.0e-5), (np.float32, 1.0e-5)])
def test_inv(mode, shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for inv
    Expectation: the result match to numpy
    """
    context.set_context(mode=mode, device_target="GPU")
    inv = NetInv()
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(*shape).astype(dtype) * prop
    output = inv(Tensor(x))
    expect_output = (1. / x).astype(dtype)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)
