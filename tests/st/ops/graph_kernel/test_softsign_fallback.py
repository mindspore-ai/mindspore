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
import platform
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.softsign = P.Softsign()

    def construct(self, x):
        return self.softsign(x)


def get_output(x, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    opt = Net()
    output = opt(Tensor(x))
    return output


def softsign_compare(shape, dtype):
    np.random.seed(0)
    x = np.random.normal(0, 1, shape).astype(dtype)

    expect = get_output(x, True)
    output = get_output(x, False)
    rtol = 1.e-4
    atol = 1.e-4
    if dtype == "float16":
        rtol = 1.e-3
        atol = 1.e-3

    assert np.allclose(expect.asnumpy(), output.asnumpy(), rtol, atol, equal_nan=True)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sofsign_cpu_pynative_mode():
    """
    /// Feature: softsign op expand fallback
    /// Description: softsign op on cpu set pynative mode test expand fallback
    /// Expectation: open graph kernel result equal to close graph kernel
    """
    if platform.system().lower() != 'linux':
        return
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    softsign_compare([2, 3, 2], np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sofsign_cpu_graph_mode():
    """
    /// Feature: softsign op expand fallback
    /// Description: softsign op on cpu set graph mode test expand fallback
    /// Expectation: open graph kernel result equal to close graph kernel
    """
    if platform.system().lower() != 'linux':
        return
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    softsign_compare([2, 3, 2], np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sofsign_gpu_pynative_mode():
    """
    /// Feature: softsign op expand fallback
    /// Description: softsign op on gpu set pynative mode test expand fallback
    /// Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    softsign_compare([2, 3, 2], np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sofsign_gpu_graph_mode():
    """
    /// Feature: softsign op expand fallback
    /// Description: softsign op on gpu set graph mode test expand fallback
    /// Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    softsign_compare([2, 3, 2], np.float32)
