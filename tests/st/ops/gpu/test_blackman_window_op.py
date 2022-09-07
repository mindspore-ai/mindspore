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
import torch
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import functional as F
import mindspore.ops.operations.other_ops as P
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import ms_function


class BlackmanWindowNet(nn.Cell):
    def __init__(self, periodic=True, dtype=mstype.float32):
        super(BlackmanWindowNet, self).__init__()
        self.blackmanwindow = P.BlackmanWindow(periodic=periodic, dtype=dtype)

    @ms_function
    def construct(self, input_x):
        return self.blackmanwindow(input_x)


def get_dtype(dtype="float16"):
    if dtype == "float16":
        nptype = np.float16
        msptype = mstype.float16
        pttype = torch.float32
    elif dtype == "float32":
        nptype = np.float32
        msptype = mstype.float32
        pttype = torch.float32
    elif dtype == "float64":
        nptype = np.float64
        msptype = mstype.float64
        pttype = torch.float64
    else:
        print("The attr 'dtype' must in [float16, float32, float64]")
    return nptype, msptype, pttype


def blackman_window(periodic, dtype, loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    nptype, msptype, pttype = get_dtype(dtype)
    input_x_np = np.array(200, dtype=np.int32)
    input_x_ms = Tensor(input_x_np)
    input_x_torch = torch.tensor(input_x_np)
    blackman_window_net = BlackmanWindowNet(periodic, msptype)
    blackman_window_output = blackman_window_net(input_x_ms)
    blackman_window_expect = torch.blackman_window(input_x_torch, periodic=periodic, dtype=pttype)
    assert np.allclose(blackman_window_output.asnumpy(), blackman_window_expect.numpy().astype(nptype), loss, loss)


def blackman_window_pynative(periodic, dtype, loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    nptype, msptype, pttype = get_dtype(dtype)
    input_x_np = np.array(200, dtype=np.int64)
    input_x_ms = Tensor(input_x_np)
    input_x_torch = torch.tensor(input_x_np)
    blackman_window_net = BlackmanWindowNet(periodic, msptype)
    blackman_window_output = blackman_window_net(input_x_ms)
    blackman_window_expect = torch.blackman_window(input_x_torch, periodic=periodic, dtype=pttype)
    assert np.allclose(blackman_window_output.asnumpy(), blackman_window_expect.numpy().astype(nptype), loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_blackman_window_graph_int32_true_float32():
    """
    Feature: ALL To ALL
    Description: test cases for BlackmanWindow
    Expectation: the result match to torch
    """
    blackman_window(periodic=True, dtype="float32", loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_blackman_window_pynative_int64_false_float64():
    """
    Feature: ALL To ALL
    Description: test cases for BlackmanWindow
    Expectation: the result match to torch
    """
    blackman_window_pynative(periodic=False, dtype="float64", loss=1.0e-5)


def test_blackman_window_functional():
    """
    Feature: test blackman_window functional API.
    Description: test case for blackman_window functional API.
    Expectation: the result match with expected result.
    """
    window_length = Tensor(10, mstype.int32)
    output = F.blackman_window(window_length, periodic=True, dtype=mstype.float32)
    expected = np.array([-2.9802322e-08, 4.0212840e-02, 2.0077014e-01, 5.0978714e-01,
                         8.4922993e-01, 1.0000000e+00, 8.4922981e-01, 5.0978690e-01,
                         2.0077008e-01, 4.0212870e-02]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_blackman_window_functional_modes():
    """
    Feature: test blackman_window functional API in PyNative and Graph modes.
    Description: test case for blackman_window functional API.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_blackman_window_functional()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_blackman_window_functional()
