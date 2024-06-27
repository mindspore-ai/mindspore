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
from tests.mark_utils import arg_mark
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


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_sofsign_pynative_mode():
    """
    Feature: softsign op expand fallback
    Description: softsign op set pynative mode test expand fallback
    Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    softsign_compare([2, 3, 2], np.float32)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_sofsign_graph_mode():
    """
    Feature: softsign op expand fallback
    Description: softsign op set graph mode test expand fallback
    Expectation: open graph kernel result equal to close graph kernel
    """
    context.set_context(mode=context.GRAPH_MODE)
    softsign_compare([2, 3, 2], np.float32)
