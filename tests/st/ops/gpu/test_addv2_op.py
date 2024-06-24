# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
from mindspore.common.tensor import Tensor
from mindspore.ops.operations import math_ops as P


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_addv2_fp32():
    """
    Feature: Returns the add of the  tensor
    Description: 1D x, float32, 1D y, float32
    Expectation: success
    """
    x = np.array([1]).astype(np.float32)
    y = np.array([1]).astype(np.float32)

    output = P.AddV2()(Tensor(x), Tensor(y))
    expect_result = np.array([2])

    assert (output.asnumpy() == expect_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_addv2_complex64():
    """
    Feature: Returns the add of the  tensor
    Description: 1D x, complex64, 1D y, complex64
    Expectation: success
    """
    x = np.array([1]).astype(np.float32) + \
        np.array([1]).astype(np.float32) * 1j
    y = np.array([1]).astype(np.float32) + \
        np.array([1]).astype(np.float32) * 1j

    output = P.AddV2()(Tensor(x), Tensor(y))
    expect_result = x + y

    assert (output.asnumpy() == expect_result).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_addv2_uint8():
    """
    Feature: Returns the add of the  tensor
    Description: 1D x, uint8, 7D y, uint8
    Expectation: success
    """
    x = np.array([1]).astype(np.uint8)
    y = np.array([2, 2, 2, 2, 2, 2, 2]).astype(np.uint8)

    output = P.AddV2()(Tensor(x), Tensor(y))
    expect_result = np.array([3, 3, 3, 3, 3, 3, 3])

    assert (output.asnumpy() == expect_result).all()
