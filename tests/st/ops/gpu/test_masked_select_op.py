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


def maskedselect():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    net = P.MaskedSelect()
    return net(Tensor(x), Tensor(mask))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedselect():
    """
    Feature: MaskedSelect
    Description:  test cases for MaskedSelect operator.
    Expectation: the result match expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    y = maskedselect()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()
