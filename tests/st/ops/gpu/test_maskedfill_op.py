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
from mindspore.ops import functional as F

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class MaskedFillNet(nn.Cell):
    def __init__(self):
        super(MaskedFillNet, self).__init__()
        self.maskedfill = P.MaskedFill()

    def construct(self, inputs, mask, value):
        return self.maskedfill(inputs, mask, value)


def maskedfill_fun(ntype):
    maskedfill_net = MaskedFillNet()
    inputs = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(ntype))
    mask = Tensor(np.array([[True, True, False, True], [False, False, True, False]]).astype(np.bool))
    value = Tensor(np.array(22).astype(ntype))
    expect = np.array([[22, 22, 3, 22], [5, 6, 22, 8]]).astype(ntype)
    output = maskedfill_net(inputs, mask, value)
    assert (output.asnumpy() == expect).all()

    mask = Tensor(np.array([[True, True, True, True], [True, True, True, True]]).astype(np.bool))
    value = Tensor(np.array(1).astype(ntype))
    expect = np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).astype(ntype)
    output = maskedfill_net(inputs, mask, value)
    assert (output.asnumpy() == expect).all()

    mask = Tensor(np.array([[False, False, False, False], [False, False, False, False]]).astype(np.bool))
    value = Tensor(np.array(22).astype(ntype))
    expect = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(ntype)
    output = maskedfill_net(inputs, mask, value)
    assert (output.asnumpy() == expect).all()

    # BroadCast
    mask = Tensor(np.array([True, True, False, True]).astype(np.bool))
    value = Tensor(np.array(22).astype(ntype))
    expect = np.array([[22, 22, 3, 22], [22, 22, 7, 22]]).astype(ntype)
    output = maskedfill_net(inputs, mask, value)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_float():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with float input.
    Expectation: The result match to expect.
    """
    maskedfill_fun(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_float16():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with float16 input.
    Expectation: The result match to expect.
    """
    maskedfill_fun(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_int():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with int input.
    Expectation: The result match to expect.
    """
    maskedfill_fun(np.int32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_int8():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with int8 input.
    Expectation: The result match to expect.
    """
    maskedfill_fun(np.int8)


def maskedfill_value(value):
    maskedfill_net = MaskedFillNet()
    inputs = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    mask = Tensor(np.array([True, True, False, True]).astype(np.bool))
    expect = np.array([0.5, 0.5, 3, 0.5]).astype(np.float32)
    output = maskedfill_net(inputs, mask, value)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_float_value():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with float value.
    Expectation: The result match to expect.
    """
    maskedfill_value(0.5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_func_masked_fill_float():
    """
    Feature: Test func masked_fill.
    Description: Test func masked_fill api with float value.
    Expectation: The result match to expect.
    """
    inputs = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float16))
    mask = Tensor(np.array([[True, True, False, True], [False, False, True, False]]).astype(np.bool))
    value = 22
    expect = np.array([[22, 22, 3, 22], [5, 6, 22, 8]]).astype(np.float16)
    output = F.masked_fill(inputs, mask, value)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_masked_fill_float():
    """
    Feature: Test Tensor masked_fill.
    Description: Test Tensor masked_fill api with float value.
    Expectation: The result match to expect.
    """
    inputs = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float16))
    mask = Tensor(np.array([[True, True, False, True], [False, False, True, False]]).astype(np.bool))
    value = 22
    output = inputs.masked_fill(mask, value)
    expect = np.array([[22, 22, 3, 22], [5, 6, 22, 8]]).astype(np.float16)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_maskedfill_tensor_value():
    """
    Feature: Test MaskedFill op.
    Description: Test MaskedFill with tensor input.
    Expectation: The result match to expect.
    """
    maskedfill_value(Tensor(0.5))
