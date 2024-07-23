# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops import auto_generate as P
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark


class LogNet(nn.Cell):
    def __init__(self):
        super(LogNet, self).__init__()
        self.log = P.Log()

    def construct(self, input_x):
        return self.log(input_x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(sens_param=True)
        self.network = network

    @jit
    def construct(self, input_x, dout):
        return self.grad(self.network)(input_x, dout)


def op_log_forward_testcase():
    input_x = Tensor(np.arange(1, 7, 1).reshape(2, 3), mstype.float32)
    expect = np.array([[0, 0.693147, 1.0986121], [1.3862944, 1.609438, 1.7917595]], dtype=np.float32)
    net = LogNet()
    output = net(input_x)
    assert np.allclose(output.asnumpy(), expect, 1e-04, 1e-04)


def op_log_backward_testcase():
    input_x = Tensor(np.arange(1, 7, 1).reshape(2, 3), mstype.float32)
    dout = Tensor(np.ones((2, 3)), mstype.float32)
    expect = np.array([[1, 0.5, 0.33333334], [0.25, 0.2, 0.16666667]], dtype=np.float32)
    net = LogNet()
    grad_net = Grad(net)
    output = grad_net(input_x, dout)
    assert np.allclose(output.asnumpy(), expect, 1e-04, 1e-04)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_op_log_cpu():
    """
    Feature: Log cpu kernel
    Description: test the log.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    op_log_forward_testcase()
    op_log_backward_testcase()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_op_log_gpu():
    """
    Feature: Log gpu kernel
    Description: test the log.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    op_log_forward_testcase()
    op_log_backward_testcase()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@test_utils.run_test_with_On
def test_op_log_ascend():
    """
    Feature: Log ascend kernel
    Description: test the log.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    op_log_forward_testcase()
    op_log_backward_testcase()
