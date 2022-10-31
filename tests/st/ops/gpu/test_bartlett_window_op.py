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
import mindspore.ops.operations.other_ops as P
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops import functional as F


class BartlettWindowNet(nn.Cell):
    def __init__(self, periodic=True, dtype=mstype.float32):
        super(BartlettWindowNet, self).__init__()
        self.bartlettwindow = P.BartlettWindow(periodic=periodic, dtype=dtype)

    @jit
    def construct(self, input_x):
        return self.bartlettwindow(input_x)


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


def bartlett_window(periodic, dtype, loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    nptype, msptype, pttype = get_dtype(dtype)
    input_x_np = np.array(200, dtype=np.int32)
    input_x_ms = Tensor(input_x_np)
    input_x_torch = torch.tensor(input_x_np)
    bartlett_window_net = BartlettWindowNet(periodic, msptype)
    bartlett_window_output = bartlett_window_net(input_x_ms)
    bartlett_window_expect = torch.bartlett_window(input_x_torch, periodic=periodic, dtype=pttype)
    assert np.allclose(bartlett_window_output.asnumpy(), bartlett_window_expect.numpy().astype(nptype), loss, loss)


def bartlett_window_pynative(periodic, dtype, loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    nptype, msptype, pttype = get_dtype(dtype)
    input_x_np = np.array(200, dtype=np.int64)
    input_x_ms = Tensor(input_x_np)
    input_x_torch = torch.tensor(input_x_np)
    bartlett_window_net = BartlettWindowNet(periodic, msptype)
    bartlett_window_output = bartlett_window_net(input_x_ms)
    bartlett_window_expect = torch.bartlett_window(input_x_torch, periodic=periodic, dtype=pttype)
    assert np.allclose(bartlett_window_output.asnumpy(), bartlett_window_expect.numpy().astype(nptype), loss, loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bartlett_window_graph_int32_true_float32():
    """
    Feature: ALL To ALL
    Description: test cases for BartlettWindow
    Expectation: the result match to torch
    """
    bartlett_window(periodic=True, dtype="float32", loss=1.0e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bartlett_window_pynative_int64_false_float64():
    """
    Feature: ALL To ALL
    Description: test cases for BartlettWindow
    Expectation: the result match to torch
    """
    bartlett_window_pynative(periodic=False, dtype="float64", loss=1.0e-5)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_bartlett_window_functional_api(mode):
    """
    Feature: test bartlett_window functional api for PyNative and Graph modes.
    Description: test bartlett_window functional api and compare with expected output.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="GPU")
    window_length = Tensor(5, mstype.int32)
    output = F.bartlett_window(window_length, periodic=True, dtype=mstype.float32)
    expected = np.array([0, 0.4, 0.8, 0.8, 0.4], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)
