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
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class TrilNet(nn.Cell):
    def __init__(self, nptype, diagonal):
        super(TrilNet, self).__init__()
        self.tril = P.Tril(diagonal=diagonal)
        self.x_np = np.random.randn(2, 3, 4).astype(nptype)
        self.x_ms = Tensor(self.x_np)

    @ms_function
    def construct(self):
        return self.tril(self.x_ms)


class TriuNet(nn.Cell):
    def __init__(self, nptype, diagonal):
        super(TriuNet, self).__init__()
        self.triu = P.Triu(diagonal=diagonal)
        self.x_np = np.random.randn(2, 3, 4).astype(nptype)
        self.x_ms = Tensor(self.x_np)

    @ms_function
    def construct(self):
        return self.triu(self.x_ms)


def tril_triu(nptype, diagonal):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    tril_ = TrilNet(nptype, diagonal)
    triu_ = TriuNet(nptype, diagonal)
    tril_output = tril_()
    triu_output = triu_()
    tril_expect = np.tril(tril_.x_np, diagonal).astype(nptype)
    triu_expect = np.triu(triu_.x_np, diagonal).astype(nptype)
    assert (tril_output.asnumpy() == tril_expect).all()
    assert (triu_output.asnumpy() == triu_expect).all()


def tril_triu_pynative(nptype, diagonal):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    tril_ = TrilNet(nptype, diagonal)
    triu_ = TriuNet(nptype, diagonal)
    tril_output = tril_()
    triu_output = triu_()
    tril_expect = np.tril(tril_.x_np, diagonal).astype(nptype)
    triu_expect = np.triu(triu_.x_np, diagonal).astype(nptype)
    assert (tril_output.asnumpy() == tril_expect).all()
    assert (triu_output.asnumpy() == triu_expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_uint8():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.uint8, -5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_uint16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.uint16, -4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_uint32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.uint32, -3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_uint64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.uint64, -2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_int8():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.int8, -1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_int16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.int16, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_int32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.int32, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_int64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.int64, 2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_float16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.float16, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.float32, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_float64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.float64, 5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_graph_bool():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu(np.bool, 6)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_uint8():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.uint8, -5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_uint16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.uint16, -4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_uint32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.uint32, -3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_uint64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.uint64, -2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_int8():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.int8, -1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_int16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.int16, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_int32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.int32, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_int64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.int64, 2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_float16():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.float16, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_float32():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.float32, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_float64():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.float64, 5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tril_triu_pynative_bool():
    """
    Feature: ALL To ALL
    Description: test cases for Tril and Triu
    Expectation: the result match to numpy
    """
    tril_triu_pynative(np.bool, 6)
