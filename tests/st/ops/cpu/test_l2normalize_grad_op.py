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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self, axis=0, epsilon=1e-4):
        super(Net, self).__init__()
        self.ops = G.L2NormalizeGrad(axis, epsilon)

    def construct(self, input_x, output, dout):
        return self.ops(input_x, output, dout)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net01():
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    axis = 1
    net = Net(axis)
    input_x = np.arange(24).astype(np.float32).reshape((2, 3, 4))
    dout = np.arange(24, 48).astype(np.float32).reshape((2, 3, 4))
    output = input_x / np.sqrt(np.sum(input_x**2, axis=axis, keepdims=True))
    except_asn = (dout - output * np.sum(output * dout, axis=axis, keepdims=True)
                  ) / np.sqrt(np.sum(input_x**2, axis=axis, keepdims=True))
    input_x = Tensor(input_x, mstype.float32)
    output = Tensor(output, mstype.float32)
    dout = Tensor(dout, mstype.float32)
    net_output = net(input_x, output, dout).asnumpy()
    precision = np.ones(shape=(2, 3, 4), dtype=np.float32) * 1.0e-5
    absolute_error = np.abs(except_asn-net_output)
    assert np.all(absolute_error < precision)
