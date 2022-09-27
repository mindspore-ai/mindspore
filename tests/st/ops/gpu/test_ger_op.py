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

# This example should be run with multiple processes.

# Please refer to the Programming Guide > Distributed Training -> Distributed Parallel Usage Example

# on mindspore.cn and focus on the contents of these three parts: Configuring Distributed Environment

# Variables, Calling the Collective Communication Library, Running the Script.
import numpy as np
import pytest

import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class GerNet(nn.Cell):
    def __init__(self):
        super(GerNet, self).__init__()
        self.ger = P.Ger()

    def construct(self, x1, x2):
        return self.ger(x1, x2)


def ger_graph(x1, x2, ms_type, nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    ger_ = GerNet()
    ms_x1 = Tensor(x1, ms_type)
    ms_x2 = Tensor(x2, ms_type)
    ger_output = ger_(ms_x1, ms_x2)
    ger_expect = np.outer(x1, x2).astype(nptype)
    assert (ger_output.asnumpy() == ger_expect).all()


def ger_pynative(x1, x2, ms_type, nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    ger_ = GerNet()
    ms_x1 = Tensor(x1, ms_type)
    ms_x2 = Tensor(x2, ms_type)
    ger_output = ger_(ms_x1, ms_x2)
    ger_expect = np.outer(x1, x2).astype(nptype)
    assert (ger_output.asnumpy() == ger_expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_pynative_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_pynative(x1, x2, mindspore.float16, np.float16)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_pynative_fp32():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_pynative(x1, x2, mindspore.float32, np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_pynative_fp64():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_pynative(x1, x2, mindspore.float64, np.float64)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_graph_fp16():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_graph(x1, x2, mindspore.float16, np.float16)



@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_graph_fp32():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_graph(x1, x2, mindspore.float32, np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ger_graph_fp64():
    """
    Feature: ALL To ALL
    Description: test cases for Ger
    Expectation: the result match to numpy
    """
    x1 = np.random.randint(-100, 100, size=10)
    x2 = np.random.randint(-100, 100, size=10)
    ger_graph(x1, x2, mindspore.float64, np.float64)
