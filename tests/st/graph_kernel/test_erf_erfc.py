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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class ErfNet(nn.Cell):
    def __init__(self):
        super(ErfNet, self).__init__()
        self.erf = P.Erf()

    def construct(self, x):
        return self.erf(x)


class ErfcNet(nn.Cell):
    def __init__(self):
        super(ErfcNet, self).__init__()
        self.erfc = P.Erfc()

    def construct(self, x):
        return self.erfc(x)


def get_output(net, inp, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    output = net()(inp)
    return output


def basic_test(net, datatype):
    inp = Tensor(np.random.random((2, 3)).astype(datatype))
    expect = get_output(net, inp, False)
    output = get_output(net, inp, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)

    inp = Tensor(np.random.random((2, 3, 3, 4, 5)).astype(datatype))
    expect = get_output(net, inp, False)
    output = get_output(net, inp, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_fp16():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    basic_test(ErfNet, np.float16)
    basic_test(ErfcNet, np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_fp32():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    basic_test(ErfNet, np.float32)
    basic_test(ErfcNet, np.float32)
