# Copyright 2025 Huawei Technoelu_grad_exties Co., Ltd
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
import os
import glob
import time
import numpy as np
import pytest
import mindspore as ms
from mindspore import context, Tensor, ops, nn
from tests.mark_utils import arg_mark

def check_npy_exist(dir_name, file_name):
    full_relative_path = os.path.join(dir_name, file_name)
    pattern = f"{full_relative_path}*"
    file_candidates = glob.glob(pattern)
    if file_candidates:
        return True
    return False

class SingleInputSingleOutputWithDumpGradientNet(nn.Cell):
    def __init__(self, case_name='default_case'):
        super().__init__()
        self.case = case_name
        self.dg = ops.DumpGradient()
    def construct(self, x):
        x = self.dg("dout_to_x.npy", x, 'out')
        return x ** 3


class SingleInputMultipleOutputsWithDumpGradientNet(nn.Cell):
    def __init__(self, case_name='default_case'):
        super().__init__()
        self.case = case_name
        self.dg = ops.DumpGradient()
    def construct(self, x):
        x = self.dg("dout_to_x.npy", x, 'out')
        return x ** 3, 2 * x


class MultipleInputsSingleOutputWithDumpGradientNet(nn.Cell):
    def __init__(self, case_name='default_case'):
        super().__init__()
        self.dg = ops.DumpGradient()
        self.case = case_name
    def construct(self, x, y, z):
        x = self.dg(f"{self.case}/dout_to_x.npy", x, 'out')
        y = self.dg(f"{self.case}/dout_to_y.npy", y, 'out')
        z = self.dg(f"{self.case}/dout_to_z.npy", z, 'out')
        return x * y * z


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_single_input_single_output(mode):
    """
    Features: Grad With DumpGradient operator.
    Description: Test DumpGradient in net
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputSingleOutputWithDumpGradientNet()
    grad_fn = ops.value_and_grad(net, grad_position=0, weights=None)
    real_output, real_x_grad = grad_fn(x)
    expect_output = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    assert np.allclose(real_output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(real_x_grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_single_input_multiple_outputs(mode):
    """
    Features: Grad With DumpGradient operator.
    Description: Test DumpGradient in net
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    net = SingleInputMultipleOutputsWithDumpGradientNet()
    grad_fn = ops.value_and_grad(net, grad_position=0, weights=None)
    real_output, real_x_grad = grad_fn(x)
    expect_output0 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_output1 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_grad = Tensor(np.array([[5, 14], [29, 50]]).astype(np.float32))
    assert np.allclose(real_output[0].asnumpy(), expect_output0.asnumpy())
    assert np.allclose(real_output[1].asnumpy(), expect_output1.asnumpy())
    assert np.allclose(real_x_grad.asnumpy(), expect_grad.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multiple_input_single_output(mode):
    """
    Features: Grad With DumpGradient operator.
    Description: Test DumpGradient in net
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    run_mode = "graph" if mode == ms.GRAPH_MODE else "pynative"
    case_name = f"{test_single_input_multiple_outputs.__name__}_{run_mode}"
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
    z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
    net = MultipleInputsSingleOutputWithDumpGradientNet(case_name=case_name)
    grad_fn = ops.value_and_grad(net, grad_position=(0, 1, 2), weights=None)
    real_output, real_input_grads = grad_fn(x, y, z)
    expect_output = Tensor(np.array([[0, 18], [-15, -8]]).astype(np.float32))
    expect_grad0 = Tensor(np.array([[0, 9], [-5, -2]]).astype(np.float32))
    expect_grad1 = Tensor(np.array([[0, 6], [15, -4]]).astype(np.float32))
    expect_grad2 = Tensor(np.array([[-2, 6], [-3, 8]]).astype(np.float32))
    assert np.allclose(real_output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(real_input_grads[0].asnumpy(), expect_grad0.asnumpy())
    assert np.allclose(real_input_grads[1].asnumpy(), expect_grad1.asnumpy())
    assert np.allclose(real_input_grads[2].asnumpy(), expect_grad2.asnumpy())
    time.sleep(1) # In GraphMode, Dump is asynchronous
    assert check_npy_exist(case_name, "dout_to_x")
    assert check_npy_exist(case_name, "dout_to_y")
    assert check_npy_exist(case_name, "dout_to_z")
