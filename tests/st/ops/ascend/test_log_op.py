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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


context.set_context(device_target="Ascend")


class NetLog(nn.Cell):
    def __init__(self):
        super(NetLog, self).__init__()
        self.log = P.Log()

    def construct(self, x):
        return self.log(x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_log(data_type, mode):
    """
    Feature: Log
    Description: test cases for Log
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(1, 2, (2, 3, 4, 4)).astype(data_type)
    x1_np = np.random.uniform(1, 2, 1).astype(data_type)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)

    expect0 = np.log(x0_np)
    expect1 = np.log(x1_np)

    log = NetLog()
    output0 = log(x0)
    output1 = log(x1)

    np.allclose(output0.asnumpy(), expect0, 0.001, 0.001)
    np.allclose(output1.asnumpy(), expect1, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [mindspore.bfloat16])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_log_bf16(data_type, mode):
    """
    Feature: Log
    Description: test cases for Log
    Expectation: the result match to torch
    """
    x0 = Tensor([1.0, 2.0, 3.0], data_type)
    x0_np = np.array([1.0, 2.0, 3.0], np.float32)

    expect0 = np.log(x0_np)

    log = NetLog()
    output0 = log(x0)

    np.allclose(output0.float().asnumpy(), expect0, 0.004, 0.004)
