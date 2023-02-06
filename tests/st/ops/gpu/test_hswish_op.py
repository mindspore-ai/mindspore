# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops.composite import GradOperation


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_x, dout):
        return self.grad(self.network)(input_x, dout)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.hswish = P.HSwish()

    def construct(self, x):
        return self.hswish(x)


def expect_hswish_forward_result(x):
    return np.where(x <= -3, 0, np.where(x >= 3, x, x * (x + 3) / 6))


def expect_hswish_backward_result(x, dout):
    return np.where(x <= -3, 0, np.where(x >= 3, 1, (x * 2 + 3) / 6)) * dout


def judge_result_correct(result, expect, loss):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect, loss, loss)


def generate_test_cases(np_type, mode, loss):
    context.set_context(mode=mode, device_target="GPU")
    x = np.array([-1, -2, 0, 4, 5]).astype(np_type)
    net = Net()
    output = net(Tensor(x))
    expect = expect_hswish_forward_result(x)
    judge_result_correct(output.asnumpy(), expect, loss)

    sens = np.array([-1.45, 0.63, 0.34, 6.43, 34.6]).astype(np_type)
    backward_net = Grad(Net())
    output = backward_net(Tensor(x), Tensor(sens))
    expect = expect_hswish_backward_result(x, sens)
    judge_result_correct(output[0].asnumpy(), expect, loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hswish_forward_and_backward():
    modes = (context.GRAPH_MODE, context.PYNATIVE_MODE)
    dtypes = (np.float32, np.float16)
    for mode in modes:
        for dtype in dtypes:
            loss = 1e-4 if (dtype == np.float32) else 1e-3
            generate_test_cases(dtype, mode, loss)
