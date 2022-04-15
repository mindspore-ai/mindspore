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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

class CdistTEST(nn.Cell):
    def __init__(self, p):
        super(CdistTEST, self).__init__()
        self.cdist = P.Cdist(p)

    def construct(self, x1, x2):
        return self.cdist(x1, x2)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_CdistP2_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 2.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistTEST(2.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_CdistP0_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 0.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistTEST(0.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2.0, 2.0], [2.0, 2.0]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_CdistP1_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 1.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistTEST(1.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[4.0, 4.0], [2.0, 2.0]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_CdistP8_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = 8.0.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistTEST(8.)
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2.1810155, 2.1810155], [1.0905077, 1.0905077]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_CdistPinf_float32():
    """
    Feature: Cdist cpu kernel
    Description: test the cdist p = inf.
    Expectation: the output[0] is same as numpy
    """
    cdist = CdistTEST(float('inf'))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist(x1, x2)
    expect = np.array([[[2., 2.], [1., 1.]]]).astype(np.float32)
    print(output)
    assert (output.asnumpy() == expect).all()
